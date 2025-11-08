# Comparison Mode Guide

## Overview

The AI Image Search script now supports a **Comparison Mode** that allows you to run both Cohere and OpenAI APIs simultaneously and compare their results side-by-side.

## How to Use Comparison Mode

When you have both `COHERE_API_KEY` and `OPENAI_API_KEY` configured in your `.env` file, the script will present you with three options:

```
1. COHERE only - Cohere embed-v4.0 (Recommended for assignment)
2. OPENAI only - OpenAI text-embedding-3-large
3. COMPARE BOTH - Run both and generate comparison report
```

Select option **3** to run comparison mode.

## What Comparison Mode Does

1. **Downloads images once** - Efficient single download for both APIs
2. **Runs Cohere API** - Generates embeddings using Cohere embed-v4.0
3. **Runs OpenAI API** - Generates embeddings using OpenAI text-embedding-3-large
4. **Computes similarities for both** - Calculates all cosine similarities
5. **Generates comparison report** - Creates side-by-side comparison with differences

## Output Files

Comparison mode generates:

- `comparison_report.txt` - Detailed comparison showing:
  - Image-to-image similarity from both providers
  - Text-to-image similarities from both providers
  - Absolute differences between scores
  - Percentage differences
  - Key observations about the differences

## Sample Output

```
================================================================================
COMPARISON RESULTS: Cohere vs OpenAI
================================================================================

1. Image-to-Image Similarity (College of Science vs Social Sciences):
--------------------------------------------------------------------------------
  Cohere:  0.227330
  OpenAI:  0.890859
  Difference: 0.663529 (291.88%)

2. Text-to-Image Similarities:
--------------------------------------------------------------------------------

   Query: "person with tape and cap"
   ----------------------------------------------------------------------------
     vs Science:
       Cohere:  0.162016
       OpenAI:  0.163689
       Difference: 0.001672 (1.03%)
```

## Key Observations

The comparison report includes analysis explaining:

- **Cohere embed-v4.0** supports native multimodal image embeddings
- **OpenAI text-embedding-3-large** only supports text, so image URLs are embedded as text
- This fundamental difference explains why similarity scores differ significantly
- For true image understanding, Cohere's multimodal model is recommended

## Why Use Comparison Mode?

1. **Educational** - Understand how different embedding models work
2. **Analytical** - See concrete differences between text-only and multimodal embeddings
3. **Validation** - Verify that Cohere's multimodal approach provides better image understanding
4. **Research** - Compare embedding dimensions (Cohere: 1536, OpenAI: 3072)

## Running the Comparison

```bash
# Make sure both API keys are set in .env
python3 ai_image_search.py

# When prompted, select option 3
Enter your choice (1, 2, or 3): 3
```

## Single Provider Mode

You can still run individual providers by selecting:
- Option 1 for Cohere only (generates `output_cohere.txt`)
- Option 2 for OpenAI only (generates `output_openai.txt`)

This is useful when you only need results from one provider or want to save API costs.

## Git Ignore

All output files are excluded from git:
- `output_*.txt`
- `comparison_report.txt`

This ensures your experimental results don't clutter the repository.
