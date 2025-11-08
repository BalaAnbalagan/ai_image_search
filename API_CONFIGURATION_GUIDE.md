# API Configuration Guide

This guide explains how to configure and switch between different API providers (Cohere, OpenAI) for the AI Image Search assignment.

## Quick Start

### Option 1: Use Cohere (Recommended for Assignment)

1. **Get Cohere API Key**:
   - Sign up at https://cohere.com
   - Go to https://dashboard.cohere.com/api-keys
   - Copy your API key

2. **Configure the script**:
   Open `ai_image_search.py` and set:
   ```python
   API_PROVIDER = "cohere"
   COHERE_API_KEY = "your-cohere-api-key-here"
   ```

3. **Install dependencies**:
   ```bash
   pip install cohere numpy requests
   ```

### Option 2: Use OpenAI

1. **Get OpenAI API Key**:
   - Sign up at https://platform.openai.com
   - Go to https://platform.openai.com/api-keys
   - Create and copy your API key

2. **Configure the script**:
   Open `ai_image_search.py` and set:
   ```python
   API_PROVIDER = "openai"
   OPENAI_API_KEY = "your-openai-api-key-here"
   ```

3. **Install dependencies**:
   ```bash
   pip install openai numpy requests
   ```

## API Provider Comparison

| Feature | Cohere | OpenAI |
|---------|--------|--------|
| **Image Embeddings** | ✅ Native support (embed-v4.0) | ⚠️ Limited (URL-based text embedding) |
| **Text Embeddings** | ✅ Optimized for search | ✅ High quality (text-embedding-3-large) |
| **Cross-Modal Search** | ✅ Designed for it | ⚠️ Workaround only |
| **Free Trial** | ✅ Yes | ✅ Yes (with credits) |
| **Assignment Spec** | ✅ Matches requirements | ❌ Doesn't match spec |

## Important Notes

### For Cohere (Recommended)
- **Pros**:
  - Supports true image embeddings
  - Matches assignment requirements exactly
  - Designed for cross-modal search
  - Free trial available

- **Cons**:
  - Requires separate API key from OpenAI

### For OpenAI
- **Pros**:
  - You may already have an API key
  - Excellent text embeddings

- **Cons**:
  - **Does NOT support direct image embeddings** via the embeddings API
  - Script uses a workaround (embedding image URLs as text)
  - **Violates assignment requirements** (asks for Cohere embed-v4.0)
  - Results won't be as meaningful for image-to-image comparison

## Security Best Practices

### Never Commit API Keys to Git

The `.gitignore` file already excludes common patterns, but be careful:

```bash
# Check what will be committed
git status

# Make sure no API keys are staged
git diff --cached
```

### Use Environment Variables (Advanced)

Create a `.env` file:
```bash
# .env
COHERE_API_KEY=your-cohere-key-here
OPENAI_API_KEY=your-openai-key-here
```

Then modify the script to load from environment:
```python
import os
from dotenv import load_dotenv

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "YOUR_COHERE_API_KEY_HERE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
```

Don't forget to install python-dotenv:
```bash
pip install python-dotenv
```

## Switching Between Providers

It's as simple as changing one line in `ai_image_search.py`:

```python
# Line 138 in ai_image_search.py
API_PROVIDER = "cohere"  # or "openai"
```

Then run:
```bash
python ai_image_search.py
```

## Troubleshooting

### Error: "Cohere SDK not installed"
```bash
pip install cohere>=5.0.0
```

### Error: "OpenAI SDK not installed"
```bash
pip install openai>=1.0.0
```

### Error: "Please set your [PROVIDER] API key"
- Open `ai_image_search.py`
- Replace the placeholder with your actual API key
- Make sure you're using the correct provider's key

### OpenAI Results Don't Make Sense
- This is expected! OpenAI's embeddings API doesn't support images directly
- The script embeds image URLs as text (a workaround)
- For proper image embeddings, use Cohere

## Recommendation for Assignment Submission

**Use Cohere** to meet the exact assignment requirements:
1. Assignment specifically asks for "Cohere Embeddings – embed-v4.0"
2. Cohere provides true multimodal embeddings
3. Results will be more meaningful
4. Free trial is sufficient for this assignment

If you submit with OpenAI, mention in your Canvas submission that you used an alternative due to [reason], but note that it doesn't fully meet the spec.
