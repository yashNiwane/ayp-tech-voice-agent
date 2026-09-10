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


def _load_kaggle_secrets():
    """Loads secrets automatically from Kaggle Secrets if running in a Kaggle Notebook."""
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        logger.info("Kaggle environment detected. Loading credentials from Kaggle Secrets...")

        # 1. Google / Gemini API Key
        for key in ["GOOGLE_API_KEY", "GEMINI_API_KEY", "google_api_key", "gemini_api_key"]:
            if not os.environ.get("GOOGLE_API_KEY"):
                try:
                    val = user_secrets.get_secret(key)
                    if val:
                        os.environ["GOOGLE_API_KEY"] = val
                        os.environ["GEMINI_API_KEY"] = val
                        logger.info(f"Loaded '{key}' from Kaggle Secrets.")
                        break
                except Exception:
                    pass

        # 2. Ngrok Authtoken
        for token_key in ["NGROK_AUTHTOKEN", "NGROK_AUTH_TOKEN", "NGROK_API_TOKEN", "ngrok_authtoken", "ngrok_token", "NGROK_TOKEN", "ngrok"]:
            if not os.environ.get("NGROK_AUTHTOKEN"):
                try:
                    val = user_secrets.get_secret(token_key)
                    if val:
                        os.environ["NGROK_AUTHTOKEN"] = val
                        logger.info(f"Loaded '{token_key}' from Kaggle Secrets.")
                        break
                except Exception:
                    pass
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Kaggle secrets check bypassed: {e}")

    # Fallback sync between GOOGLE_API_KEY and GEMINI_API_KEY if one is set
    if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY")


_load_kaggle_secrets()

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loan_leads.db")

# Protection thresholds
MAX_CALL_DURATION_SECONDS = 240  # 4 minutes maximum (allows deep understanding, natural conversation, reprompting, and confirmation)
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


current_call_lead_id: int | None = None


def insert_or_update_loan_lead(data: Mapping[str, Any]) -> int:
    """Inserts or updates a lead in the SQLite database for the active call."""
    global current_call_lead_id
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        if current_call_lead_id is not None:
            # Update existing lead record for this call
            fields_to_update = []
            values = []
            for field in [
                "customer_name",
                "phone_number",
                "loan_purpose",
                "loan_amount",
                "employment_type",
                "monthly_income",
                "tenure_years",
                "existing_emi",
                "interest_level",
            ]:
                if field in data and data[field] is not None:
                    fields_to_update.append(f"{field} = ?")
                    values.append(data[field])

            if "is_interested" in data and data["is_interested"] is not None:
                fields_to_update.append("is_interested = ?")
                values.append(1 if data["is_interested"] else 0)

            if fields_to_update:
                values.append(current_call_lead_id)
                query = f"UPDATE loan_leads SET {', '.join(fields_to_update)}, status = 'DETAILS_UPDATED' WHERE id = ?"
                cursor.execute(query, tuple(values))
                conn.commit()
                logger.info(f"Updated existing loan lead #{current_call_lead_id}: {data}")
                return current_call_lead_id

        # Otherwise insert a new record
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
        current_call_lead_id = cursor.lastrowid
        logger.info(f"Saved new loan lead #{current_call_lead_id} to SQLite: {data}")
        return current_call_lead_id


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
Aap Aarav (HDFC Bank) hain. Ek genuine, empathetic aur trusted financial advisor ki tarah customer se Hindi/Hinglish mein baat karein.

PRIMARY PURPOSE:
Call ka sabse bada maksad customer ke saath ek behtareen, dilchasp baat-cheet (great conversation) karna hai, unki actual financial need ko gehrayi se samajhna hai, aur respectful tarike se saari zaroori jankari collect karke confirm karna hai.

ESSENTIAL DETAILS CHECKLIST (In sabhi ka hona zaroori hai):
1. [Name] - Customer ka shubh naam
2. [Purpose] - Loan kis kaam ke liye chahiye (e.g., home renovation, personal, wedding, medical, business growth)
3. [Amount] - Kitni rashi ki requirement hai (e.g., ₹5,00,000)
4. [Employment] - Salaried hain ya apna business/self-employed
5. [Monthly Income] - Mahine ki in-hand aamdani (approximate monthly income)
6. [Tenure] - Kitne saal mein chukana chahenge (1 se 5 saal)
7. [Interest Level] - Unki dilchaspi ka level (HIGH / MEDIUM / LOW / NOT_INTERESTED)

RE-PROMPTING & MISSING INFO RULE (AGAR CUSTOMER POORI DETAILS NA DE):
- Agar customer kisi sawaal ka aadha-adhura jawab de, gol-mol baat kare, ya koi zaroori information miss kar de:
  - Kabhi bhi chup mat baithiye aur na hi adhoori info ke saath conclusion pe jump karein.
  - Politely aur respectfully dobara poochein:
    * Name miss hone par: "Sir, main aapka shubh naam theek se sun nahi paya, kya aap ek baar dohra sakte hain?"
    * Amount clear na hone par: "Samajh gaya sir! Par approx kitne amount ki requirement rahegi aapko?"
    * Income miss hone par: "Sir, approval ke liye mahine ki lagbhag in-hand income jaan sakte hain? Chahe approximate batayein."
    * Purpose miss hone par: "Sir, loan kisi specific zaroorat ke liye dekh rahe hain, jaise business, ghar ka kaam, ya personal use?"

CONVERSATIONAL STEP-BY-STEP FLOW:
1. Shuruat (Warm, courteous opening):
   "Namaste, main HDFC Bank se Aarav baat kar raha hoon. Kya main aapka shubh naam jaan sakta hoon?"
   (Pehle dialogue mein LOAN ka koi zikr na karein, sirf polite greeting aur naam poochein.)
2. Offer introduction & understanding needs:
   Customer ka naam sunkar aadar dein (e.g. "Shukriya [Name] ji!"). Batayein ki unke account par exclusive 10% rate par pre-approved personal loan ka special offer unlock hua hai. Puchiye kya filhal unhe kisi fund ya loan ki zaroorat hai.
3. Empathetic discovery:
   Customer ki zaroorat ko acknowledge karein ("Wah, bahut badhiya sir", "Bilkul, yeh toh bahut zaroori step hai"). Purpose, required amount, employment type, monthly income aur preferred tenure step-by-step jaan lijiye.
4. Saving details:
   Jaise hi saari complete details mil jayein, turant `save_customer_loan_details` tool call karein.
5. Mandatory Confirmation with Customer:
   Details save karne ke baad customer ko politely confirm karein:
   "Dhanyawad [Name] ji! Main aapki saari details ek baar confirm kar deta hoon:
    - Loan Amount: ₹[Amount] ([Purpose] ke liye)
    - Employment: [Salaried/Business], Monthly Income: ₹[Income]
    - Tenure: [Tenure] saal, Interest Rate: 10%
    Kya yeh saari details bilkul accurate hain, ya aap isme koi change ya update karna chahenge?"
6. Real-time Corrections & Updates:
   Agar customer bole ki "amount 8 lakh kar do" ya "income change karni hai":
   - "Ji zaroor [Name] ji, main abhi isko update kar deta hoon."
   - Turant `update_customer_loan_details` call karein aur updated detail re-confirm karein.
7. Graceful Closing:
   Customer confirm kare "Haan sab sahi hai" -> Batayein ki HDFC branch verification team unse formal process ke liye jald sampark karegi -> Thank you kahein aur `end_call` call karein.

RULES OF CONDUCT:
- 1-2 short, punchy sentences per turn. Ek waqt par sirf EK clear question poochein taaki customer aasaani se jawab de sake.
- Customer ki baat ko active listening ke saath respond karein ("Ji bilkul", "Sahi kaha aapne", "Main samajh gaya").
- Customer agar mana kare ("Mujhe loan nahi chahiye" ya "Main busy hoon"): Zabardasti push mat karein. "Koi baat nahi sir, apna keemti samay dene ke liye shukriya" bolkar `end_call` call karein.
- No fake promises, external links, SMS/WhatsApp transfer.
"""

# Global reference for active worker to trigger disconnects
current_active_worker: PipelineWorker | None = None


async def save_customer_loan_details_handler(params: FunctionCallParams):
    """Tool handler that writes collected lead data into SQLite database."""
    args = params.arguments
    logger.info(f"Executing save_customer_loan_details tool: {args}")
    try:
        lead_id = insert_or_update_loan_lead(args)
        result = {
            "status": "success",
            "lead_id": lead_id,
            "message": "Customer loan application details saved. Now please confirm these details with the customer.",
        }
    except Exception as e:
        logger.error(f"Error saving lead to SQLite: {e}")
        result = {"status": "error", "error": str(e)}

    await params.result_callback(result)


async def update_customer_loan_details_handler(params: FunctionCallParams):
    """Tool handler that updates specific fields when customer requests corrections or changes."""
    args = params.arguments
    logger.info(f"Executing update_customer_loan_details tool: {args}")
    try:
        lead_id = insert_or_update_loan_lead(args)
        result = {
            "status": "success",
            "lead_id": lead_id,
            "message": "Customer loan details updated successfully in database. Confirm the updated detail to customer.",
        }
    except Exception as e:
        logger.error(f"Error updating lead in SQLite: {e}")
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
    description="Saves customer's personal loan application details into HDFC database when initially collected.",
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

update_loan_tool = FunctionSchema(
    name="update_customer_loan_details",
    description="Updates previously recorded customer loan application details when the customer requests a change or correction.",
    properties={
        "customer_name": {
            "type": "string",
            "description": "Updated name of the customer if requested",
        },
        "loan_purpose": {
            "type": "string",
            "description": "Updated purpose of the loan",
        },
        "loan_amount": {
            "type": "number",
            "description": "Updated loan amount in INR (e.g. 500000)",
        },
        "employment_type": {
            "type": "string",
            "description": "Updated employment status, e.g. 'Salaried' or 'Self-Employed'",
        },
        "monthly_income": {
            "type": "number",
            "description": "Updated monthly in-hand income in INR",
        },
        "tenure_years": {
            "type": "integer",
            "description": "Updated repayment tenure in years (1 to 5)",
        },
        "existing_emi": {
            "type": "number",
            "description": "Updated ongoing monthly EMI amount in INR",
        },
        "interest_level": {
            "type": "string",
            "description": "Updated customer interest level: 'HIGH', 'MEDIUM', 'LOW', 'NOT_INTERESTED'",
        },
        "is_interested": {
            "type": "boolean",
            "description": "True if customer is interested, False if they want to cancel",
        },
    },
    required=[],
    handler=update_customer_loan_details_handler,
)

end_call_tool = FunctionSchema(
    name="end_call",
    description="Terminates and hangs up the live phone call when conversation is finished, customer is not interested, or confirmed and wrapped up.",
    properties={
        "reason": {
            "type": "string",
            "description": "Reason for hanging up: 'DETAILS_CONFIRMED', 'DETAILS_COLLECTED', 'CUSTOMER_NOT_INTERESTED', 'CUSTOMER_BUSY', 'IRRELEVANT_CONVERSATION'",
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

    if not os.environ.get("GOOGLE_API_KEY"):
        _load_kaggle_secrets()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set. "
            "Please add it to Kaggle Secrets (Add-ons -> Secrets) or set it in your .env file."
        )

    llm = GeminiLiveLLMService(
        name="AYP Voice Engine",
        api_key=api_key,
        tools=[save_loan_tool, update_loan_tool, end_call_tool],
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
        global current_call_lead_id
        current_call_lead_id = None
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
        from fastapi.responses import StreamingResponse
        import io
        import csv

        @app.get("/api/export-csv")
        async def export_leads_csv():
            """Exports all collected leads from SQLite database as a CSV file."""
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                "ID",
                "Customer Name",
                "Phone Number",
                "Loan Purpose",
                "Loan Amount (INR)",
                "Employment Type",
                "Monthly Income (INR)",
                "Tenure (Years)",
                "Existing EMI (INR)",
                "Interest Rate (%)",
                "Interest Level",
                "Interested?",
                "Status",
                "Recorded Timestamp (UTC)",
            ])
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM loan_leads ORDER BY id DESC")
                rows = cursor.fetchall()
                for row in rows:
                    writer.writerow(row)

            output.seek(0)
            return StreamingResponse(
                io.BytesIO(output.getvalue().encode("utf-8-sig")),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=ayp_tech_collected_leads.csv"},
            )

        @app.get("/api/leads")
        async def get_leads_json():
            """Returns leads as JSON for the web UI."""
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM loan_leads ORDER BY id DESC")
                rows = [dict(r) for r in cursor.fetchall()]
                return {"count": len(rows), "leads": rows}

        app.mount("/client", StaticFiles(directory=CUSTOM_UI_DIR, html=True))

        @app.get("/", include_in_schema=False)
        async def root_redirect():
            return RedirectResponse(url="/client/")

    runner_module._setup_frontend_routes = _custom_setup_frontend_routes
def _start_ngrok_tunnel(port: int = 7860):
    """Starts an ngrok tunnel using the authtoken from Kaggle Secrets or environment variables."""
    token = os.environ.get("NGROK_AUTHTOKEN") or os.environ.get("NGROK_AUTH_TOKEN")
    if not token:
        return False

    try:
        try:
            from pyngrok import ngrok
        except ImportError:
            import subprocess
            logger.info("pyngrok not found. Automatically installing pyngrok...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyngrok>=7.1.0"], check=True)
            from pyngrok import ngrok

        logger.info("Authenticating ngrok with credentials from Kaggle Secrets...")
        ngrok.set_auth_token(token)

        # Kill any stale ngrok tunnels if restarting in notebook
        try:
            ngrok.kill()
        except Exception:
            pass

        tunnel = ngrok.connect(port, "http")
        url = tunnel.public_url
        if url.startswith("http://"):
            url = url.replace("http://", "https://", 1)

        print("\n" + "=" * 70)
        print(f"🚀 AYP Tech Public Demo URL (ngrok):")
        print(f"   {url}")
        print("=" * 70 + "\n")
        logger.info(f"Public demo live at: {url}")
        return True
    except Exception as e:
        logger.error(f"Failed to start ngrok tunnel: {e}")
        return False


def _start_cloudflare_tunnel(port: int = 7860):
    """Starts a Cloudflare quick tunnel to expose the AYP Tech UI publicly as fallback."""
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
        f = open(log_file, "a", encoding="utf-8")
        proc = subprocess.Popen(
            [
                cloudflared_bin,
                "tunnel",
                "--protocol",
                "http2",
                "--url",
                f"http://localhost:{port}",
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Poll log file for public tunnel URL and print it prominently
        for _ in range(40):
            time.sleep(1.0)
            if os.path.exists(log_file):
                try:
                    with open(log_file, "r", encoding="utf-8", errors="ignore") as rf:
                        content = rf.read()
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


def _start_tunnel():
    """Attempts to start an ngrok tunnel first (using Kaggle Secrets).
    If no ngrok token is provided or it fails, falls back to Cloudflare.
    """
    if _start_ngrok_tunnel():
        return
    logger.info("No ngrok token provided or ngrok failed, falling back to Cloudflare tunnel...")
    _start_cloudflare_tunnel()


if __name__ == "__main__":
    # Ensure public STUN servers are configured so WebRTC connections work over tunnels / Kaggle NAT
    if not os.getenv("PIPECAT_ICE_SERVERS"):
        os.environ["PIPECAT_ICE_SERVERS"] = "stun:stun.l.google.com:19302,stun:stun1.l.google.com:19302,stun:stun2.l.google.com:19302"

    _start_tunnel()
    from pipecat.runner.run import main

    main()