# GitHub Deployment Guide

Follow these steps to push MeshNet AI to GitHub.

---

## Step 1 — Create the GitHub repository

1. Go to https://github.com/new
2. Set **Repository name**: `meshnet-ai`
3. Set visibility: **Public** (or Private)
4. **Do NOT** initialize with a README (you already have one)
5. Click **Create repository**

---

## Step 2 — Initialize Git locally

Open a terminal inside the project folder and run:

```bash
git init
git add .
git commit -m "feat: initial commit — MeshNet AI Tunisia disaster recovery"
```

---

## Step 3 — Connect to GitHub and push

Replace `YOUR_USERNAME` with your GitHub username:

```bash
git remote add origin https://github.com/YOUR_USERNAME/meshnet-ai.git
git branch -M main
git push -u origin main
```

---

## Step 4 — Install the package in editable mode (for contributors)

```bash
pip install -e .
```

This makes `import src.network`, `import src.ai`, etc. work from anywhere in the project.

---

## Step 5 — Update the README badge URL

In `README.md`, replace both occurrences of `YOUR_USERNAME` with your actual GitHub username so the CI badge and clone command are correct.

---

## Step 6 — (Optional) Track model weights with Git LFS

Model `.zip` files can be large (50–200 MB). To store them in the repo without bloating history:

```bash
git lfs install
git lfs track "models/**/*.zip"
git add .gitattributes
git commit -m "chore: configure Git LFS for model weights"
git push
```

If you prefer not to commit weights, collaborators can regenerate them with:

```bash
python train_rl_agents.py
```

---

## Workflow summary

```
clone → pip install -e . → train → compare → dashboard
```

```bash
git clone https://github.com/YOUR_USERNAME/meshnet-ai.git
cd meshnet-ai
pip install -r requirements.txt && pip install -e .
python train_rl_agents.py      # ~15-40 min
python comparison_test.py      # generates charts + metrics JSON
python dashboard.py            # open http://127.0.0.1:8050
```
