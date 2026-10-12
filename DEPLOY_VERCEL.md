# Deploying FantasyScore AI to Vercel

This directory is pre-configured for one-click deployment on **Vercel** using Python Serverless Functions.

---

## 📁 Key Vercel Configuration Files

1. **[`api/index.py`](api/index.py)**: The serverless entry point exporting the WSGI `app` / `handler`.
2. **[`vercel.json`](vercel.json)**: Directs all incoming traffic routes (`/(.*)`) to the serverless entry point using `@vercel/python`.
3. **[`requirements.txt`](requirements.txt)**: Optimized runtime dependencies (`flask`, `lightgbm`, `scikit-learn`, `scipy`, `pandas`, `numpy`, `joblib`), omitting heavy offline visualization libraries to stay well within Vercel's 250 MB serverless uncompressed size limit.
4. **[`.vercelignore`](.vercelignore)**: Prevents large scratch files, local caches, and notebooks from being uploaded to the deployment bundle.

---

## Method 1: Deploy via GitHub (Recommended)

1. Open your terminal in this directory:
   ```bash
   cd "d:/Documents/Sem 2-1/ML/Fantasy_Score_Prediction/Fantasy_Score_Prediction"
   ```
2. Initialize Git (if not already done) and push to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Configure FantasyScore AI for Vercel deployment"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
3. Go to [https://vercel.com](https://vercel.com) and log in.
4. Click **"Add New..."** &rarr; **"Project"**.
5. Import your GitHub repository.
6. Under **Root Directory**, leave as `./` (or select the project folder if your repo has outer folders).
7. Click **"Deploy"**.

---

## Method 2: Deploy via Vercel CLI

You can deploy directly from your local terminal using `npx`:

1. Open terminal inside the project directory:
   ```powershell
   cd "d:\Documents\Sem 2-1\ML\Fantasy_Score_Prediction\Fantasy_Score_Prediction"
   ```
2. Run the Vercel CLI:
   ```powershell
   npx vercel
   ```
3. Answer the prompts:
   - **Set up and deploy?** &rarr; `Y`
   - **Which scope?** &rarr; Select your Vercel account
   - **Link to existing project?** &rarr; `N`
   - **What's your project's name?** &rarr; `fantasyscore-ai` (or your choice)
   - **In which directory is your code located?** &rarr; `./`
4. For production deployment, run:
   ```powershell
   npx vercel --prod
   ```
5. Vercel will output your live deployment URL (e.g., `https://fantasyscore-ai.vercel.app`).
