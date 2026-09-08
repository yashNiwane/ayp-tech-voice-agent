# AYP Tech - Conversational Voice Agent

AI-powered voice agent for automated customer engagement and data collection with real-time call protection, background office noise synthesis, and SQLite lead recording.

---

## ⚡ Quick Start on Kaggle

You can easily run this voice agent inside a **Kaggle Notebook** with public Cloudflare access:

### Step 1: Clone Repository in Kaggle
In your Kaggle notebook cell:
```bash
!git clone https://github.com/yashNiwane/ayp-tech-voice-agent.git
%cd ayp-tech-voice-agent
```

### Step 2: Install Dependencies & Cloudflare Tunnel
```bash
!pip install -r requirements.txt
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
```

### Step 3: Set Gemini API Key & Run
```bash
import os
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY_HERE"  # Or use Kaggle Secrets

!python main.py
```

The notebook output will print your public demo link:
```text
======================================================================
🚀 AYP Tech Public Cloudflare Demo URL:
   https://xxxx-xxxx-xxxx.trycloudflare.com
======================================================================
```
Share this URL with your client to test the live voice portal directly from any web browser!

---

## 💻 Local Setup

1. **Clone repository:**
   ```bash
   git clone https://github.com/yashNiwane/ayp-tech-voice-agent.git
   cd ayp-tech-voice-agent
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key:**
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   ```

4. **Start the server:**
   ```bash
   python main.py
   ```
   Open `http://localhost:7860` in your browser.

---

## 🛡️ Key Features
- **Branded White-Label UI**: Branded exclusively for AYP Tech with all internal tech stack details stripped.
- **Natural Conversational Flow**: Courteous greeting and structured collection of applicant name, loan requirements, income, and genuine interest level.
- **Built-in Call Protection**:
  - 15s initial silence disconnect.
  - 2-minute total duration hard cap.
  - 25s mid-call inactivity watchdog.
  - Agent-driven disconnect tool for rejection or time wasting.
- **SQLite Database**: Automatically persists leads and call termination logs in `loan_leads.db`.
