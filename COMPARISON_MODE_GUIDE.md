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

1. **Downloads images once** - Efficient caching prevents duplicate downloads
2. **Runs Cohere API** - Generates embeddings using Cohere embed-v4.0
3. **Runs OpenAI API** - Generates embeddings using OpenAI text-embedding-3-large
4. **Computes similarities for both** - Calculates all cosine similarities
5. **Displays table comparison** - Easy-to-read table format with side-by-side results
6. **Calculates statistics** - Average, max, and min differences across all comparisons
7. **Generates comparison report** - Saves detailed report to file

## Output Files

Comparison mode generates **two report formats**:

### 1. Text Report (`comparison_report.txt`)
- Plain text format with tables
- Image-to-image similarity from both providers
- Text-to-image similarities from both providers
- Absolute differences between scores
- Percentage differences
- Summary statistics
- Key observations

### 2. HTML Report (`comparison_report.html`) ⭐ **Submission Ready**
- **Professional format with embedded images**
- Visual display of both test images
- Styled tables and statistics
- Methodology section
- Summary statistics in grid cards
- Key observations highlighted
- **Perfect for assignment submission**
- **Print to PDF directly from browser**

## Sample Output

```
================================================================================
COMPARISON RESULTS: Cohere vs OpenAI
================================================================================

1. Image-to-Image Similarity
Comparison                  Cohere    OpenAI    Difference  Diff %
-------------------------------------------------------------------
Science vs Social Sciences  0.227330  0.890503  0.663172    291.72%

2. Text-to-Image Similarities
Query                     Image            Cohere    OpenAI    Difference  Diff %
----------------------------------------------------------------------------------
person with tape and cap  Science          0.162016  0.163549  0.001533    0.95%
person with tape and cap  Social Sciences  0.044612  0.146793  0.102181    229.05%
cart with single tire     Science          0.018512  0.076938  0.058427    315.62%
cart with single tire     Social Sciences  0.238193  0.066356  0.171837    72.14%

================================================================================
Summary Statistics
================================================================================
  Embedding Dimensions:
    • Cohere embed-v4.0: 1536 dimensions
    • OpenAI text-embedding-3-large: 3072 dimensions

  Similarity Difference Statistics:
    • Average difference: 0.199430
    • Max difference: 0.663172
    • Min difference: 0.001533
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

After completion, you'll see:
```
✓ Comparison report saved to: comparison_report.txt
✓ HTML comparison report saved to: comparison_report.html
  → Open in browser and print to PDF for submission
```

## Converting HTML to PDF for Submission

1. Open `comparison_report.html` in your web browser
2. Press `Cmd+P` (Mac) or `Ctrl+P` (Windows/Linux)
3. Select "Save as PDF" as the destination
4. Click "Save"
5. Submit the PDF to Canvas

**The HTML report includes:**
- Both test images displayed visually
- All comparison results in styled tables
- Professional formatting suitable for academic submission
- Timestamp showing when the analysis was run

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
