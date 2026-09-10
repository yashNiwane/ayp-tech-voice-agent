# AYP Tech - Conversational Voice Agent

AI-powered voice agent for automated customer engagement and data collection with real-time call protection, background office noise synthesis, and SQLite lead recording.

---

## ⚡ Quick Start on Kaggle

You can easily run this voice agent inside a **Kaggle Notebook** with public access via ngrok:

### Step 1: Add Credentials to Kaggle Secrets
In your Kaggle Notebook:
1. Click **Add-ons** > **Secrets** in the top menu bar.
2. Add the following secrets:
   - `GOOGLE_API_KEY`: Your Gemini API key.
   - `NGROK_AUTHTOKEN`: Your free authtoken from [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken).

### Step 2: Clone & Install Dependencies
In a Kaggle notebook code cell:
```bash
!git clone https://github.com/yashNiwane/ayp-tech-voice-agent.git
%cd ayp-tech-voice-agent
!pip install -r requirements.txt
```

### Step 3: Run
```bash
!python main.py
```
*(The script automatically retrieves your `GOOGLE_API_KEY` and `NGROK_AUTHTOKEN` directly from Kaggle Secrets!)*

The notebook output will print your live public demo link:
```text
======================================================================
🚀 AYP Tech Public Demo URL (ngrok):
   https://xxxx-xxxx.ngrok-free.app
======================================================================
```
Share this URL to access and test the live voice portal directly from any web browser!

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
