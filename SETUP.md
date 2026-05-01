# Setup Guide — Run in VSCode & Deploy to Render

---

## 1. Run locally in VSCode (5 minutes)

### Step 1 — Open the project
```
File → Open Folder → select  vera-project/
```

### Step 2 — Create virtual environment
Open the integrated terminal (`` Ctrl+` ``):

```bash
# Create venv
python -m venv venv

# Activate — Mac/Linux:
source venv/bin/activate

# Activate — Windows:
venv\Scripts\activate

# Install dependencies:
pip install -r requirements.txt
```

When VSCode asks "We noticed a new virtual environment was created. Do you want to select it?" → click **Yes**.

### Step 3 — Set your Anthropic API key
```bash
cp .env.example .env
```
Open `.env` and replace the placeholder:
```
ANTHROPIC_API_KEY=sk-ant-YOUR-REAL-KEY-HERE
```
Get your key at: https://console.anthropic.com

### Step 4 — Start the bot

**Option A (recommended) — VSCode Run button:**
Press `F5` → select **"▶ Run Vera Bot (port 8080)"**

**Option B — Terminal:**
```bash
uvicorn bot:app --host 0.0.0.0 --port 8080 --reload
```

Bot is live at: http://localhost:8080

### Step 5 — Verify all 5 endpoints work
Open a new terminal tab:

```bash
export BOT=http://localhost:8080

# 1. Health check
curl $BOT/v1/healthz

# 2. Metadata
curl $BOT/v1/metadata

# 3. Push a category context
curl -X POST -H "Content-Type: application/json" \
  -d @dataset/categories/dentists.json \
  $BOT/v1/context

# 4. Fire a tick
curl -X POST -H "Content-Type: application/json" \
  -d '{"now":"2026-05-01T10:00:00Z","available_triggers":["trg_001_research_digest_dentists"]}' \
  $BOT/v1/tick

# 5. Send a reply
curl -X POST -H "Content-Type: application/json" \
  -d '{"conversation_id":"conv_test","merchant_id":"m_001_drmeera_dentist_delhi","from_role":"merchant","message":"Yes please send the abstract","received_at":"2026-05-01T10:05:00Z","turn_number":2}' \
  $BOT/v1/reply
```

Swagger UI: http://localhost:8080/docs

### Step 6 — Generate submission.jsonl
```bash
# Make sure venv is active and .env is set
python scripts/generate_submission.py
```
This calls Claude for all 30 test pairs (~2-3 min). Output: `submission.jsonl`

### Step 7 — Run the judge simulator
```bash
export BOT_URL=http://localhost:8080

# Copy judge_simulator.py from the challenge zip into this folder first
python judge_simulator.py
```

---

## 2. Deploy to Render (10 minutes) — FOR SUBMISSION

Render gives you a free public HTTPS URL. This is what you submit.

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "vera bot submission"
# Create a new repo on github.com then:
git remote add origin https://github.com/YOUR_USERNAME/vera-bot.git
git push -u origin main
```

### Step 2 — Create Render account
Go to https://render.com → Sign up (free)

### Step 3 — New Web Service
1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub account
3. Select your `vera-bot` repository
4. Click **"Connect"**

### Step 4 — Configure the service
Fill in these fields:

| Field | Value |
|-------|-------|
| **Name** | `vera-bot` (or anything) |
| **Runtime** | **Docker** (Render auto-detects your Dockerfile) |
| **Branch** | `main` |
| **Region** | Singapore (closest to India) |
| **Instance Type** | **Free** |

### Step 5 — Add environment variable
Scroll down to **"Environment Variables"** section:
- Key: `ANTHROPIC_API_KEY`
- Value: `sk-ant-YOUR-REAL-KEY`

Click **"Add"**

### Step 6 — Deploy
Click **"Create Web Service"**

Render will build your Docker image and deploy. Takes 3-5 minutes.

### Step 7 — Get your URL
Once deployed, you'll see a URL like:
```
https://vera-bot-xxxx.onrender.com
```

### Step 8 — Verify it's live
```bash
curl https://vera-bot-xxxx.onrender.com/v1/healthz
# Should return: {"status":"ok","uptime_seconds":...,"contexts_loaded":{...}}
```

### Step 9 — Submit this URL
Submit `https://vera-bot-xxxx.onrender.com` to the challenge portal.

---

## ⚠️ Important: Keep the bot alive

Render free tier **spins down after 15 minutes of inactivity**.

To prevent this during the judge test window (48-72 hours after submission):

**Option A — Upgrade to Render Starter ($7/mo)** — most reliable.

**Option B — Free keep-alive ping** using UptimeRobot:
1. Go to https://uptimerobot.com → free account
2. Add monitor: HTTP(S) → `https://vera-bot-xxxx.onrender.com/v1/healthz`
3. Interval: every 5 minutes
4. This pings your bot every 5 min and keeps it warm

---

## Project structure

```
vera-project/
├── bot.py                       ← Main bot (all 5 endpoints + full composition logic)
├── requirements.txt
├── Dockerfile
├── render.yaml                  ← Render deployment config
├── .env.example                 ← Copy to .env, add your API key
├── .env                         ← Your secrets (gitignored, NEVER commit this)
├── README.md                    ← Submission README
├── SETUP.md                     ← This file
│
├── scripts/
│   ├── generate_submission.py   ← Generates submission.jsonl for all 30 test pairs
│   └── generate_dataset.py      ← Expands seeds → full 50M/200C/100T dataset
│
├── dataset/
│   ├── categories/              ← 5 category JSONs (dentists, salons, restaurants, gyms, pharmacies)
│   └── seeds/                   ← merchants_seed, customers_seed, triggers_seed
│
├── expanded/                    ← Full expanded dataset (auto-generated)
│   ├── merchants/               ← 50 merchant JSONs
│   ├── customers/               ← 200 customer JSONs
│   ├── triggers/                ← 100 trigger JSONs
│   ├── categories/              ← Expanded category JSONs
│   └── test_pairs.json          ← 30 canonical test pairs
│
└── .vscode/
    ├── launch.json              ← F5 run configs
    └── settings.json
```

---

## What gets submitted

1. **Bot URL** → `https://vera-bot-xxxx.onrender.com` (submit via challenge portal)
2. **submission.jsonl** → 30 lines, one per test pair (upload via portal or email)
3. **README.md** → approach + tradeoffs (already written)

That's it. The judge harness calls your live URL — no code upload needed.
