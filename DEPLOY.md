# ShipFuel AI — Complete Deployment Guide

This guide walks you through every step to get the project live:
- **Backend** (FastAPI, Python) → hosted on [Railway](https://railway.app)
- **Frontend** (Next.js 16, React 19) → hosted on [Vercel](https://vercel.com)

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Prerequisites](#2-prerequisites)
3. [Step 1 — Push Code to GitHub](#3-step-1--push-code-to-github)
4. [Step 2 — Deploy Backend on Railway](#4-step-2--deploy-backend-on-railway)
5. [Step 3 — Deploy Frontend on Vercel](#5-step-3--deploy-frontend-on-vercel)
6. [Step 4 — Connect Frontend to Backend](#6-step-4--connect-frontend-to-backend)
7. [Verify the Live App](#7-verify-the-live-app)
8. [Local Development](#8-local-development)
9. [Retraining Models](#9-retraining-models)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Project Structure

```
production/
├── backend/                   ← FastAPI Python backend
│   ├── main.py                   API app (endpoints: /predict, /metrics, /stats, /feature-importance)
│   ├── prepare_and_train.py      Dataset prep + model training script
│   ├── requirements.txt          Python dependencies
│   ├── railway.toml              Railway deployment config (auto-detected)
│   ├── .gitignore
│   ├── data/
│   │   └── ship_data_production.csv   92,567 labelled rows (real + synthetic)
│   └── models/
│       ├── scaler.pkl
│       ├── features.pkl
│       ├── linear_regression.pkl
│       ├── random_forest.pkl
│       ├── xgboost.pkl
│       ├── metrics.pkl / metrics.json
│       ├── feature_importance.pkl
│       └── stats.pkl / stats.json
│
└── frrontend/                 ← Next.js 16 frontend  (note: double-r folder name)
    ├── app/
    │   ├── page.tsx              Main UI — sliders, charts, predictions
    │   ├── layout.tsx
    │   └── globals.css
    ├── .env.local                Local dev env (not committed)
    ├── .env.production           Production env template
    ├── next.config.ts
    └── package.json
```

---

## 2. Prerequisites

Make sure you have accounts on:
- [GitHub](https://github.com) — to host the code (free)
- [Railway](https://railway.app) — to host the Python backend (free tier available)
- [Vercel](https://vercel.com) — to host the Next.js frontend (free tier)

And locally installed:
- **Git** — `git --version`
- **Python 3.10+** — `python --version`
- **Node.js 18+** — `node --version`

---

## 3. Step 1 — Push Code to GitHub

Railway and Vercel both deploy directly from a GitHub repository. You need to push your code there first.

### 3a. Create a new GitHub repository

1. Go to [github.com/new](https://github.com/new)
2. Name it something like `shipfuel-ai`
3. Set it to **Public** (free Vercel/Railway deployments work with public repos)
4. **Do NOT** check "Add a README file" — leave it empty
5. Click **Create repository**

### 3b. Initialize git and push

Open a terminal in `C:\Users\krish\OneDrive\Desktop\iiser\production` and run:

```bash
# Navigate to the production folder
cd "C:\Users\krish\OneDrive\Desktop\iiser\production"

# Initialize a git repo
git init

# Add everything
git add .

# Make the first commit
git commit -m "Initial commit: ShipFuel AI backend + frontend"

# Link to your GitHub repo (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/shipfuel-ai.git

# Push
git branch -M main
git push -u origin main
```

> **Tip:** If git asks for credentials, use your GitHub username and a
> [Personal Access Token](https://github.com/settings/tokens/new)
> (scope: `repo`) instead of your password.

### 3c. Verify on GitHub

Go to `https://github.com/YOUR_USERNAME/shipfuel-ai` — you should see the
`backend/` and `frrontend/` folders listed.

---

## 4. Step 2 — Deploy Backend on Railway

### 4a. Create a new Railway project

1. Go to [railway.app](https://railway.app) and log in (use GitHub login for easiest setup)
2. Click **New Project** (top right)
3. Choose **Deploy from GitHub repo**
4. Authorize Railway to access your GitHub account if prompted
5. Find and select your `shipfuel-ai` repository

### 4b. Configure the root directory

Railway needs to know the backend is inside a subfolder, not the repo root.

1. After selecting the repo, Railway will show a service being created
2. Click on the service card → go to **Settings** tab
3. Under **Source**, find **Root Directory**
4. Set it to: `backend`
5. Click **Save**

> Railway will now only look at `production/backend/` when building and running.

### 4c. Confirm build settings

Railway reads `railway.toml` automatically. Verify these settings in the **Settings → Deploy** section:

| Setting | Value |
|---|---|
| Builder | Nixpacks (auto-detected) |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/health` |
| Health Check Timeout | 30s |

These are already configured in `railway.toml` — you should not need to change anything.

### 4d. Trigger the first deployment

1. Go to the **Deployments** tab
2. Click **Deploy** (or it may auto-deploy when you save settings)
3. Watch the build logs — it will:
   - Detect Python from `requirements.txt`
   - Install all packages (scikit-learn, xgboost, fastapi, etc.)
   - Start the uvicorn server
4. Wait for the status to show **Active** (green dot) — takes 2–4 minutes

### 4e. Get your Railway URL

1. Go to the **Settings** tab → **Networking** section
2. Click **Generate Domain** (under Public Networking)
3. Railway gives you a URL like: `https://shipfuel-ai-production.up.railway.app`

**Copy this URL — you need it for the frontend.**

### 4f. Test the backend

Open these URLs in your browser to confirm it's working:

```
https://YOUR-RAILWAY-URL.railway.app/health
→ Should return: {"status":"ok","models_loaded":true}

https://YOUR-RAILWAY-URL.railway.app/docs
→ Should open the interactive Swagger API documentation

https://YOUR-RAILWAY-URL.railway.app/stats
→ Should return dataset statistics (92,567 rows, etc.)
```

---

## 5. Step 3 — Deploy Frontend on Vercel

### 5a. Create a new Vercel project

1. Go to [vercel.com](https://vercel.com) and log in (use GitHub login)
2. Click **Add New... → Project**
3. Find your `shipfuel-ai` repo in the list and click **Import**

### 5b. Configure the root directory

Just like Railway, Vercel needs to know the frontend lives in a subfolder.

1. In the project setup screen, expand **Root Directory** (under "Configure Project")
2. Click **Edit** and type: `frrontend`
   > Note: the folder has a typo — double-r (`frrontend`) — type it exactly
3. Click **Continue**

### 5c. Confirm framework settings

Vercel auto-detects Next.js. Verify:

| Setting | Value |
|---|---|
| Framework Preset | Next.js (auto-detected) |
| Build Command | `next build` (auto-detected) |
| Output Directory | `.next` (auto-detected) |
| Install Command | `npm install` (auto-detected) |

Leave these as-is.

### 5d. Do NOT deploy yet

Before clicking Deploy, you need to add the environment variable in the next step.

---

## 6. Step 4 — Connect Frontend to Backend

This is the critical step — the frontend needs to know where the backend API lives.

### 6a. Add the environment variable in Vercel

Still on the Vercel project setup screen (before clicking Deploy):

1. Scroll down to **Environment Variables**
2. Add the following:

| Name | Value |
|---|---|
| `NEXT_PUBLIC_API_URL` | `https://YOUR-RAILWAY-URL.railway.app` |

   Replace `YOUR-RAILWAY-URL` with the actual Railway domain from Step 4e.
   **No trailing slash.**

   Example:
   ```
   NEXT_PUBLIC_API_URL = https://shipfuel-ai-production.up.railway.app
   ```

3. Make sure the environments selected are: **Production**, **Preview**, **Development** (all three checked)

### 6b. Deploy

Click **Deploy**. Vercel will:
1. Clone your repo
2. Run `npm install`
3. Run `next build`
4. Serve the static output globally via CDN

This takes about 1–2 minutes. When done, you'll see a success screen with your live URL like:
```
https://shipfuel-ai.vercel.app
```

### 6c. Update the env variable after the fact (if needed)

If you ever need to change the Railway URL:
1. Go to your Vercel project → **Settings** → **Environment Variables**
2. Edit `NEXT_PUBLIC_API_URL`
3. Go to **Deployments** → click the three dots on the latest deployment → **Redeploy**

---

## 7. Verify the Live App

Open your Vercel URL in a browser. Check that:

- [ ] The page loads with the dark navy theme and the anchor logo
- [ ] The header shows the dataset stats (92,567 records, 80,567 real, 12,000 synthetic)
- [ ] The 9 parameter sliders are visible and interactive
- [ ] Clicking **Predict Fuel Consumption** returns results from all 3 models
- [ ] The Feature Importance bar chart renders
- [ ] The Model Accuracy Radar chart renders
- [ ] The Model Performance metrics table shows R² > 0.96 for all models

If predictions don't load, open the browser **DevTools Console** (F12) — any CORS or connection errors will appear there.

---

## 8. Local Development

Run both services locally to test changes before pushing.

### 8a. Backend

```bash
cd production/backend

# Create a virtual environment (first time only)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies (first time only)
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload --port 8000
```

The backend will be live at:
- API: `http://localhost:8000`
- Interactive docs (Swagger UI): `http://localhost:8000/docs`
- Auto-reload on every file save (`--reload` flag)

### 8b. Frontend

```bash
cd production/frrontend

# Install dependencies (first time only)
npm install

# Make sure .env.local points to local backend
# File: production/frrontend/.env.local
# Should contain:
#   NEXT_PUBLIC_API_URL=http://localhost:8000

# Run the dev server
npm run dev
```

The frontend will be live at `http://localhost:3000` with hot-reload.

### 8c. Test a prediction manually (optional)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rpm": 80,
    "power": 4500,
    "mean_draft": 14,
    "wave_height": 1.5,
    "wind_speed": 8,
    "current_speed": 0.3,
    "speed_through_water": 12,
    "swell_height": 1.2,
    "trim": 0.5
  }'
```

Expected response:
```json
{
  "linear_regression": 960.97,
  "random_forest": 1130.02,
  "xgboost": 1196.53,
  "ensemble_avg": 1095.84,
  "unit": "L/hr",
  "inputs": { ... }
}
```

---

## 9. Retraining Models

If you update the dataset or want to retrain from scratch:

```bash
cd production/backend

# Activate venv first
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux

# Run the training script
python prepare_and_train.py
```

This will:
1. Load `dataset versions/v3/synthetic_dataset.csv` (relative to the repo root)
2. Compute `Trim = Draft Aft - Draft Fwd`
3. Select the 9 model features + target + `is_synthetic` label
4. Save the production CSV to `data/ship_data_production.csv`
5. Train Linear Regression, Random Forest (200 trees), and XGBoost
6. Save all `.pkl` files to `models/`
7. Print a full metrics comparison table

After retraining, commit the new `.pkl` files and push to GitHub. Railway and Vercel will auto-redeploy.

```bash
git add backend/models/ backend/data/
git commit -m "Retrain models with updated dataset"
git push
```

---

## 10. Troubleshooting

### Frontend shows "Failed to connect to API"

**Cause:** The `NEXT_PUBLIC_API_URL` environment variable is wrong or missing.

**Fix:**
1. Go to Vercel → Settings → Environment Variables
2. Confirm `NEXT_PUBLIC_API_URL` is set to your Railway URL (no trailing slash)
3. Redeploy the frontend

### Railway deployment fails at build

**Cause:** Usually a missing dependency or Python version mismatch.

**Fix:**
1. Check the Railway build logs for the specific error
2. If it's a package version issue, loosen the version pins in `requirements.txt`
   (e.g. change `scikit-learn==1.6.0` to `scikit-learn>=1.5.0`)
3. Push the change — Railway will auto-rebuild

### CORS error in browser console

**Cause:** The backend is rejecting requests from the Vercel domain.

**Fix:** The backend already has `allow_origins=["*"]` in `main.py`, so this
should not happen. If it does, check that the Railway URL has no path suffix
(it should end in `.railway.app`, not `.railway.app/predict`).

### Railway URL is not generating

**Cause:** Railway requires a credit card on file to generate public domains on the free plan in some regions.

**Fix:** Add a card (you won't be charged on the free tier) or use Railway's
Hobby plan ($5/month).

### Vercel build fails

**Cause:** TypeScript or ESLint errors.

**Fix:**
```bash
cd production/frrontend
npm run build
```
Fix any errors shown, then push again.

### Models give wildly wrong predictions

**Cause:** Input values are outside the training data range.

**Expected ranges from training data:**
| Parameter | Min | Max | Typical |
|---|---|---|---|
| RPM | 0 | ~120 | 80–100 |
| Power | 0 | ~12,000 kW | 3,000–6,000 |
| Mean Draft | 5 | 22 m | 12–16 |
| Wave Height | 0 | ~8 m | 0.5–3 |
| Wind Speed | 0 | ~30 m/s | 3–15 |
| Current Speed | 0 | ~3 m/s | 0–1 |
| Speed Through Water | 0 | 22 kn | 8–14 |
| Swell Height | 0 | ~8 m | 0.5–2.5 |
| Trim | -3 | 3 m | -1 to 1 |

---

## Quick Reference

| What | Where |
|---|---|
| Backend URL (local) | `http://localhost:8000` |
| Frontend URL (local) | `http://localhost:3000` |
| API docs (local) | `http://localhost:8000/docs` |
| API docs (live) | `https://YOUR-RAILWAY-URL.railway.app/docs` |
| Frontend (live) | `https://YOUR-APP.vercel.app` |
| Retrain script | `production/backend/prepare_and_train.py` |
| Env variable (frontend) | `NEXT_PUBLIC_API_URL` |
| Railway config | `production/backend/railway.toml` |

---

*ShipFuel AI — IISER Ship Engine Fuel Consumption Prediction Project*
