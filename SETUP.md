# Project Setup Guide

## Dedicated Virtual Environment

This project uses a dedicated virtual environment named `ai_image_search_env` to keep dependencies isolated from other projects.

## Quick Start

### 1. Activate the Environment

```bash
source ai_image_search_env/bin/activate
```

You should see `(ai_image_search_env)` prefix in your terminal.

### 2. Launch Jupyter

```bash
jupyter notebook ai_image_search.ipynb
```

### 3. Select the Kernel

When the notebook opens:
- Click "Kernel" → "Change Kernel"
- Select **"AI Image Search (Python 3.9)"**

That's it! The notebook will now use the isolated environment.

---

## What Was Created

✅ **Virtual Environment**: `ai_image_search_env/`
- Located in project root
- Contains all dependencies
- Python 3.9

✅ **Jupyter Kernel**: `ai_image_search`
- Display name: "AI Image Search (Python 3.9)"
- Uses the project's virtual environment
- Available in all Jupyter sessions

✅ **Dependencies Installed**:
- cohere >= 5.0.0
- openai >= 1.0.0
- jupyter >= 1.0.0
- pandas >= 2.0.0
- pillow >= 10.0.0
- ipython >= 8.0.0
- ipywidgets >= 8.0.0
- numpy >= 1.24.0
- requests >= 2.31.0
- python-dotenv >= 1.0.0

---

## Benefits of This Setup

### ✅ Project Isolation
- Dependencies don't conflict with other projects
- Clean environment for this project only

### ✅ Easy Identification
- Kernel name clearly shows what project it's for
- No confusion with system Python or other projects

### ✅ Reproducible
- Anyone can recreate the environment
- Same dependencies every time

### ✅ Safe
- Won't affect global Python installation
- Can delete and recreate anytime

---

## Common Commands

### Activate Environment
```bash
source ai_image_search_env/bin/activate
```

### Deactivate Environment
```bash
deactivate
```

### Check Active Environment
```bash
which python
# Should show: /Users/.../ai_image_search_env/bin/python
```

### Install New Package
```bash
source ai_image_search_env/bin/activate
pip install package_name
```

### Update requirements.txt
```bash
source ai_image_search_env/bin/activate
pip freeze > requirements.txt
```

---

## Running the Notebook

### Option 1: With Jupyter (Recommended)

```bash
# Activate environment
source ai_image_search_env/bin/activate

# Launch Jupyter
jupyter notebook ai_image_search.ipynb

# In Jupyter: Select "AI Image Search (Python 3.9)" kernel
```

### Option 2: With JupyterLab

```bash
# Activate environment
source ai_image_search_env/bin/activate

# Launch JupyterLab
jupyter lab ai_image_search.ipynb

# Select kernel as above
```

### Option 3: In VS Code

1. Open the notebook in VS Code
2. Click "Select Kernel" (top right)
3. Choose "AI Image Search (Python 3.9)"

---

## Troubleshooting

### Kernel Not Showing Up

If you don't see "AI Image Search (Python 3.9)" in kernel list:

```bash
source ai_image_search_env/bin/activate
python -m ipykernel install --user --name=ai_image_search --display-name="AI Image Search (Python 3.9)"
```

### Wrong Python Version

Check which Python is being used:
```bash
source ai_image_search_env/bin/activate
python --version
which python
```

### Dependencies Not Found

Reinstall dependencies:
```bash
source ai_image_search_env/bin/activate
pip install -r requirements.txt
```

### Environment Corrupted

Delete and recreate:
```bash
rm -rf ai_image_search_env
python3 -m venv ai_image_search_env
source ai_image_search_env/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=ai_image_search --display-name="AI Image Search (Python 3.9)"
```

---

## Cleaning Up (Optional)

### Remove Kernel Only

```bash
jupyter kernelspec uninstall ai_image_search
```

### Remove Environment

```bash
# Remove the directory
rm -rf ai_image_search_env

# Remove the kernel
jupyter kernelspec uninstall ai_image_search
```

---

## For Other Team Members

To set up the same environment:

```bash
# 1. Clone the repository
cd /path/to/ai_image_search

# 2. Create virtual environment
python3 -m venv ai_image_search_env

# 3. Activate it
source ai_image_search_env/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install Jupyter kernel
python -m ipykernel install --user --name=ai_image_search --display-name="AI Image Search (Python 3.9)"

# 6. Add API keys to .env file
cp .env.example .env
# Edit .env with your keys

# 7. Launch Jupyter
jupyter notebook ai_image_search.ipynb
```

---

## Notes

- The `.gitignore` file excludes `ai_image_search_env/` from git
- Your `.env` file with API keys is also excluded (safe)
- Images in `images/` folder ARE tracked in git
- The kernel is installed in your user directory, not the project

---

**Environment is ready!** Just activate and launch Jupyter.
