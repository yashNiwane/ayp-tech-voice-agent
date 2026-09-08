import asyncio
from datetime import datetime
import os
import sqlite3
import sys
import time
from typing import Any, Mapping

import av
from dotenv import load_dotenv
from loguru import logger
import numpy as np

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    LLMRunFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.gemini_live.llm import (
    ContextWindowCompressionParams,
    GeminiLiveLLMService,
)
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loan_leads.db")

# Protection thresholds
MAX_CALL_DURATION_SECONDS = 120  # 2 minutes maximum
INITIAL_SILENCE_TIMEOUT_SECONDS = 15  # 15 seconds if customer is silent after pickup
INACTIVITY_TIMEOUT_SECONDS = 25  # 25 seconds of silence mid-call
VOICE_ENERGY_THRESHOLD = 250.0  # RMS threshold for detecting actual user speech


def init_db():
    """Initialize SQLite database for storing collected loan applicant leads and call logs."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS loan_leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT,
                phone_number TEXT,
                loan_purpose TEXT,
                loan_amount REAL,
                employment_type TEXT,
                monthly_income REAL,
                tenure_years INTEGER,
                existing_emi REAL,
                interest_rate REAL DEFAULT 10.0,
                interest_level TEXT,
                is_interested BOOLEAN DEFAULT 1,
                status TEXT DEFAULT 'COLLECTED',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Add columns if migrating from earlier schema
        cursor.execute("PRAGMA table_info(loan_leads)")
        columns = [col[1] for col in cursor.fetchall()]
        if "interest_level" not in columns:
            cursor.execute("ALTER TABLE loan_leads ADD COLUMN interest_level TEXT")
        if "is_interested" not in columns:
            cursor.execute("ALTER TABLE loan_leads ADD COLUMN is_interested BOOLEAN DEFAULT 1")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS call_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                duration_seconds REAL,
                disconnect_reason TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    logger.info(f"SQLite database initialized at {DB_PATH}")


def insert_loan_lead(data: Mapping[str, Any]) -> int:
    """Inserts a lead in the SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO loan_leads (
                customer_name,
                phone_number,
                loan_purpose,
                loan_amount,
                employment_type,
                monthly_income,
                tenure_years,
                existing_emi,
                interest_rate,
                interest_level,
                is_interested,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("customer_name"),
                data.get("phone_number"),
                data.get("loan_purpose"),
                data.get("loan_amount"),
                data.get("employment_type"),
                data.get("monthly_income"),
                data.get("tenure_years"),
                data.get("existing_emi", 0.0),
                10.0,
                data.get("interest_level", "HIGH"),
                1 if data.get("is_interested", True) else 0,
                "DETAILS_RECORDED",
            ),
        )
        conn.commit()
        lead_id = cursor.lastrowid
        logger.info(f"Saved loan lead #{lead_id} to SQLite: {data}")
        return lead_id


def log_call_termination(duration: float, reason: str, notes: str = ""):
    """Records call completion or termination reason into SQLite."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO call_logs (duration_seconds, disconnect_reason, notes)
                VALUES (?, ?, ?)
                """,
                (round(duration, 2), reason, notes),
            )
            conn.commit()
            logger.info(f"Recorded call log: {reason} after {duration:.1f}s")
    except Exception as e:
        logger.error(f"Failed to record call log: {e}")


class CallProtectionMonitor(FrameProcessor):
    """Monitors live call health and terminates calls exceeding boundaries:

    1. Initial silence: User picks up but does not speak within 15 seconds.
    2. Hard cap: Total call time cannot exceed 2 minutes (prevents playing/time wasting).
    3. Mid-call prolonged silence: No speech for 25 seconds.
    """

    def __init__(self, on_disconnect_callback):
        super().__init__()
        self._on_disconnect_callback = on_disconnect_callback
        self._call_start_time: float | None = None
        self._last_speech_time: float | None = None
        self._has_user_spoken = False
        self._monitor_task: asyncio.Task | None = None
        self._terminated = False

    def start_monitoring(self):
        self._call_start_time = time.time()
        self._last_speech_time = time.time()
        self._has_user_spoken = False
        self._terminated = False
        self._monitor_task = asyncio.create_task(self._watchdog_loop())
        logger.info(
            f"Call protection monitor started: Max {MAX_CALL_DURATION_SECONDS}s, 15s initial silence detection."
        )

    def stop_monitoring(self):
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()

    async def _trigger_disconnect(self, reason: str, notes: str):
        if self._terminated:
            return
        self._terminated = True
        elapsed = time.time() - (self._call_start_time or time.time())
        logger.warning(f"Safety Protection triggered call disconnect: {reason} ({elapsed:.1f}s)")
        log_call_termination(elapsed, reason, notes)
        await self._on_disconnect_callback(reason)

    async def _watchdog_loop(self):
        try:
            while not self._terminated:
                await asyncio.sleep(1.0)
                now = time.time()
                elapsed = now - (self._call_start_time or now)

                # Rule 1: 2-minute hard cap
                if elapsed >= MAX_CALL_DURATION_SECONDS:
                    await self._trigger_disconnect(
                        "MAX_DURATION_EXCEEDED",
                        f"Call hit {MAX_CALL_DURATION_SECONDS}s hard cap limit.",
                    )
                    break

                # Rule 2: 15-second initial silence if customer picks up and doesn't speak
                if not self._has_user_spoken and elapsed >= INITIAL_SILENCE_TIMEOUT_SECONDS:
                    await self._trigger_disconnect(
                        "INITIAL_SILENCE_TIMEOUT",
                        f"Customer picked up but spoke nothing for {INITIAL_SILENCE_TIMEOUT_SECONDS}s.",
                    )
                    break

                # Rule 3: 25-second mid-call dead silence
                if (
                    self._has_user_spoken
                    and self._last_speech_time
                    and (now - self._last_speech_time) >= INACTIVITY_TIMEOUT_SECONDS
                ):
                    await self._trigger_disconnect(
                        "MID_CALL_INACTIVITY",
                        f"Customer went silent for {INACTIVITY_TIMEOUT_SECONDS}s during active call.",
                    )
                    break
        except asyncio.CancelledError:
            pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            # Calculate RMS energy of incoming audio chunk to detect speech
            audio_data = np.frombuffer(frame.audio, dtype=np.int16)
            if len(audio_data) > 0:
                energy = float(np.mean(np.abs(audio_data)))
                if energy > VOICE_ENERGY_THRESHOLD:
                    self._last_speech_time = time.time()
                    if not self._has_user_spoken:
                        self._has_user_spoken = True
                        logger.info("Customer speech detected for the first time on call.")

        await self.push_frame(frame, direction)


class BackgroundNoiseProcessor(FrameProcessor):
    """Mixes realistic background noise into the agent's speech frames."""

    def __init__(
        self,
        file_path: str,
        volume: float = 0.5,
    ):
        super().__init__()
        self._file_path = file_path
        self._volume = volume
        self._sound_cache: dict[int, np.ndarray] = {}
        self._sound_pos = 0

    def _load_audio(self, target_sample_rate: int):
        try:
            logger.info(
                f"Loading background noise from {self._file_path} at {target_sample_rate} Hz"
            )
            container = av.open(self._file_path)
            resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sample_rate)
            frames = []
            for frame in container.decode(audio=0):
                for resampled_frame in resampler.resample(frame):
                    frames.append(resampled_frame.to_ndarray())
            container.close()
            if frames:
                data = np.concatenate(frames, axis=1).squeeze()
                self._sound_cache[target_sample_rate] = data
                logger.info(
                    f"Background noise loaded: {len(data)} samples ({len(data)/target_sample_rate:.2f}s)"
                )
            else:
                logger.warning(f"No audio frames decoded from {self._file_path}")
        except Exception as e:
            logger.error(f"Failed to load background noise audio file: {e}")

    def _mix_audio(self, audio: bytes, sample_rate: int) -> bytes:
        if sample_rate not in self._sound_cache:
            self._load_audio(sample_rate)
        sound_data = self._sound_cache.get(sample_rate)
        if sound_data is None or len(sound_data) == 0:
            return audio

        audio_np = np.frombuffer(audio, dtype=np.int16)
        chunk_size = len(audio_np)
        if chunk_size == 0:
            return audio

        total_samples = len(sound_data)
        if self._sound_pos + chunk_size > total_samples:
            first_part = sound_data[self._sound_pos :]
            remaining = chunk_size - len(first_part)
            second_part = sound_data[:remaining]
            sound_chunk = np.concatenate([first_part, second_part])
            self._sound_pos = remaining
        else:
            sound_chunk = sound_data[self._sound_pos : self._sound_pos + chunk_size]
            self._sound_pos += chunk_size

        mixed = np.clip(
            audio_np.astype(np.int32)
            + (sound_chunk.astype(np.float32) * self._volume).astype(np.int32),
            -32768,
            32767,
        ).astype(np.int16)

        return mixed.tobytes()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSAudioRawFrame):
            frame.audio = self._mix_audio(frame.audio, frame.sample_rate)

        await self.push_frame(frame, direction)


BG_NOISE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bgnoice.m4a")
BACKGROUND_NOISE_VOLUME = 0.5

transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}

HDFC_LOAN_HINDI_INSTRUCTION = """
Aap Aarav (HDFC Bank) hain. Natural Hindi/Hinglish mein baat karein.

Goal: Name, loan details aur interest level collect karke `save_customer_loan_details` mein save karna.

Rules:
- 2–3 short sentences per turn. Ek time par sirf ek question.
- First message: Sirf greeting + name poochein. LOAN ka zikr bilkul na karein.
- No fake promises, links, ya call transfer. Call 2 min se choti rakhein.

Flow:
1. Start: "Namaste, main HDFC Bank se Aarav baat kar raha hoon. Kya main aapka shubh naam jaan sakta hoon?"
2. Naam milne par: 10% pre-approved personal loan offer batayein aur requirement poochein.
3. Purpose → Amount → Salaried/Business → Monthly income → Tenure.
4. Interest level evaluate karein: HIGH / MEDIUM / LOW / NOT_INTERESTED.
5. Details milte hi `save_customer_loan_details` call karein aur thank you kahein.

Tools & Objections:
- `save_customer_loan_details`: Name, amount, employment, income, interest_level milte hi call karein.
- `end_call`: Customer bole 'Call cut karo', loan reject kare, ya time waste kare toh polite alvida kehkar turant call karein.
- Busy / No loan: "Koi baat nahi sir, thank you" bolkar `end_call` karein.
- Rate high: "Sir 10% hamara lowest tier rate hai."
"""

# Global reference for active worker to trigger disconnects
current_active_worker: PipelineWorker | None = None


async def save_customer_loan_details_handler(params: FunctionCallParams):
    """Tool handler that writes collected lead data into SQLite database."""
    args = params.arguments
    logger.info(f"Executing save_customer_loan_details tool with arguments: {args}")
    try:
        lead_id = insert_loan_lead(args)
        result = {
            "status": "success",
            "lead_id": lead_id,
            "message": "Customer loan application details have been recorded successfully in SQLite database.",
        }
    except Exception as e:
        logger.error(f"Error saving lead to SQLite: {e}")
        result = {"status": "error", "error": str(e)}

    await params.result_callback(result)


async def end_call_handler(params: FunctionCallParams):
    """Tool handler that allows the AI agent to immediately hang up the call."""
    reason = params.arguments.get("reason", "AGENT_ENDED_CALL")
    logger.info(f"AI agent invoked end_call tool: {reason}")
    await params.result_callback({"status": "hanging_up", "reason": reason})

    # Schedule clean termination
    if current_active_worker:
        asyncio.create_task(current_active_worker.queue_frame(EndFrame(reason=reason)))


save_loan_tool = FunctionSchema(
    name="save_customer_loan_details",
    description="Saves customer's personal loan application details including their name and assessed interest level into HDFC database.",
    properties={
        "customer_name": {
            "type": "string",
            "description": "Full name of the customer collected during the call",
        },
        "loan_purpose": {
            "type": "string",
            "description": "Purpose of the loan, e.g. home renovation, wedding, travel, business, personal",
        },
        "loan_amount": {
            "type": "number",
            "description": "Desired loan amount in INR, e.g. 500000",
        },
        "employment_type": {
            "type": "string",
            "description": "Customer employment status, e.g. 'Salaried' or 'Self-Employed'",
        },
        "monthly_income": {
            "type": "number",
            "description": "Customer approximate monthly in-hand income in INR",
        },
        "tenure_years": {
            "type": "integer",
            "description": "Repayment tenure in years (1 to 5)",
        },
        "existing_emi": {
            "type": "number",
            "description": "Any ongoing monthly EMI amount in INR, default 0 if none",
        },
        "interest_level": {
            "type": "string",
            "description": "Customer genuine interest level: 'HIGH' (genuinely interested/urgent), 'MEDIUM' (curious/considering), 'LOW' (lukewarm/hesitant), 'NOT_INTERESTED' (rejecting)",
        },
        "is_interested": {
            "type": "boolean",
            "description": "True if customer is genuinely interested in taking the loan, False if not interested",
        },
    },
    required=["loan_amount", "customer_name", "interest_level", "is_interested"],
    handler=save_customer_loan_details_handler,
)

end_call_tool = FunctionSchema(
    name="end_call",
    description="Terminates and hangs up the live phone call when conversation is finished, customer is not interested, or wasting time.",
    properties={
        "reason": {
            "type": "string",
            "description": "Reason for hanging up: 'DETAILS_COLLECTED', 'CUSTOMER_NOT_INTERESTED', 'CUSTOMER_BUSY', 'IRRELEVANT_CONVERSATION', 'CUSTOMER_PLAYING'",
        }
    },
    required=["reason"],
    handler=end_call_handler,
)


async def run_bot(
    transport: BaseTransport,
    runner_args: RunnerArguments,
):
    global current_active_worker
    logger.info("Starting HDFC Personal Loan Hindi voice bot (Enhanced Details Collection)")

    init_db()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    llm = GeminiLiveLLMService(
        name="AYP Voice Engine",
        api_key=api_key,
        tools=[save_loan_tool, end_call_tool],
        settings=GeminiLiveLLMService.Settings(
            model="gemini-3.1-flash-live-preview",
            voice="Puck",
            language="hi-IN",
            max_tokens=2048,
            thinking={"thinking_budget": 0},
            context_window_compression=ContextWindowCompressionParams(enabled=True),
            system_instruction=HDFC_LOAN_HINDI_INSTRUCTION,
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    # Call Protection Monitor
    async def handle_protection_disconnect(reason: str):
        if current_active_worker:
            logger.info(f"Ending pipeline worker due to protection: {reason}")
            await current_active_worker.queue_frame(EndFrame(reason=reason))

    protection_monitor = CallProtectionMonitor(on_disconnect_callback=handle_protection_disconnect)

    bg_processor = (
        BackgroundNoiseProcessor(BG_NOISE_PATH, volume=BACKGROUND_NOISE_VOLUME)
        if os.path.exists(BG_NOISE_PATH)
        else None
    )

    pipeline_stages = [
        transport.input(),
        protection_monitor,
        user_aggregator,
        llm,
    ]
    if bg_processor:
        pipeline_stages.append(bg_processor)
    pipeline_stages.extend(
        [
            transport.output(),
            assistant_aggregator,
        ]
    )

    pipeline = Pipeline(pipeline_stages)

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=False,
            enable_usage_metrics=False,
        ),
    )
    current_active_worker = worker

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected to HDFC Hindi Loan session")
        protection_monitor.start_monitoring()
        context.add_message(
            {
                "role": "user",
                "content": (
                    "Call connect ho chuka hai aur customer ne phone utha liya hai. "
                    "Aarav ke roop me shuru karein: "
                    "'Namaste, main HDFC Bank se Aarav baat kar raha hoon. Kya main aapka shubh naam jaan sakta hoon?' "
                    "(Pehle sentence mein loan ka koi zikr na karein, sirf naam poochein)."
                ),
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected from HDFC Hindi Loan session")
        protection_monitor.stop_monitoring()
        await worker.cancel()

    logger.info("HDFC Hindi Personal Loan pipeline with multi-layer protection started")

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(
        runner_args,
        transport_params,
    )

    await run_bot(transport, runner_args)


# Custom UI mounting for AYP Tech branding
CUSTOM_UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_ui")
if os.path.isdir(CUSTOM_UI_DIR):
    import pipecat.runner.run as runner_module
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import RedirectResponse

    def _custom_setup_frontend_routes(app):
        app.mount("/client", StaticFiles(directory=CUSTOM_UI_DIR, html=True))

        @app.get("/api/leads")
        async def get_leads():
            """Returns all collected loan applicant leads from SQLite."""
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM loan_leads ORDER BY created_at DESC")
                    rows = cursor.fetchall()
                    return {"status": "success", "count": len(rows), "leads": [dict(r) for r in rows]}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        @app.get("/api/call-logs")
        async def get_call_logs():
            """Returns all call termination logs."""
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM call_logs ORDER BY created_at DESC")
                    rows = cursor.fetchall()
                    return {"status": "success", "count": len(rows), "logs": [dict(r) for r in rows]}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        @app.get("/", include_in_schema=False)
        async def root_redirect():
            return RedirectResponse(url="/client/")

    runner_module._setup_frontend_routes = _custom_setup_frontend_routes
def _start_cloudflare_tunnel():
    """Starts a Cloudflare quick tunnel to expose the AYP Tech UI publicly (works on Windows & Linux/Kaggle)."""
    import shutil
    import subprocess
    import threading
    import time
    import re

    # Find cloudflared binary (system PATH or local executable)
    cloudflared_bin = shutil.which("cloudflared")
    if not cloudflared_bin:
        exe_name = "cloudflared.exe" if sys.platform == "win32" else "cloudflared"
        local_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), exe_name)
        if os.path.exists(local_bin):
            cloudflared_bin = local_bin

    if not cloudflared_bin:
        logger.warning("cloudflared binary not found; skipping automatic tunnel creation.")
        return

    def run_tunnel():
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tunnel.log")
        with open(log_file, "w", encoding="utf-8") as f:
            proc = subprocess.Popen(
                [cloudflared_bin, "tunnel", "--url", "http://localhost:7860"],
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )

        # Poll log file for public tunnel URL and print it prominently
        for _ in range(40):
            time.sleep(1.0)
            if os.path.exists(log_file):
                try:
                    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        match = re.search(r"https://[-a-zA-Z0-9]+\.trycloudflare\.com", content)
                        if match:
                            url = match.group(0)
                            print("\n" + "=" * 70)
                            print(f"🚀 AYP Tech Public Cloudflare Demo URL:")
                            print(f"   {url}")
                            print("=" * 70 + "\n")
                            logger.info(f"Public demo accessible at: {url}")
                            break
                except Exception:
                    pass

    t = threading.Thread(target=run_tunnel, daemon=True)
    t.start()
    logger.info(f"Cloudflare tunnel started using {cloudflared_bin}")


if __name__ == "__main__":
    _start_cloudflare_tunnel()
    from pipecat.runner.run import main

    main()