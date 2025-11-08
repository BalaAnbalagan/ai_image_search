# How to Generate Submission Report

This guide shows you how to create a professional comparison report with images for your assignment submission.

## Quick Start

1. **Run the comparison mode:**
   ```bash
   python3 ai_image_search.py
   ```

2. **Select option 3** when prompted:
   ```
   Enter your choice (1, 2, or 3): 3
   ```

3. **Wait for completion** - You'll see:
   ```
   ✓ Comparison report saved to: comparison_report.txt
   ✓ HTML comparison report saved to: comparison_report.html
     → Open in browser and print to PDF for submission
   ```

## Converting to PDF

### Method 1: Using Web Browser (Recommended)

1. **Open the HTML file:**
   - Double-click `comparison_report.html`
   - Or drag it into your browser

2. **Print to PDF:**
   - **Mac:** Press `Cmd + P`, select "Save as PDF"
   - **Windows:** Press `Ctrl + P`, select "Save as PDF" or "Microsoft Print to PDF"
   - **Linux:** Press `Ctrl + P`, select "Print to File (PDF)"

3. **Save the PDF:**
   - Choose a name like `CMPE273_Comparison_Report.pdf`
   - Save in your assignment folder

### Method 2: Using Command Line (macOS/Linux)

```bash
# Install wkhtmltopdf if not already installed
# macOS: brew install wkhtmltopdf
# Ubuntu: sudo apt-get install wkhtmltopdf

# Convert to PDF
wkhtmltopdf comparison_report.html comparison_report.pdf
```

## What's Included in the Report

The HTML report includes:

✅ **Visual Display of Test Images**
   - SJSU College of Science image
   - SJSU College of Social Sciences image
   - Both images displayed in high quality

✅ **Test Queries**
   - "person with tape and cap"
   - "cart with single tire"

✅ **Methodology Section**
   - Explanation of Cohere embed-v4.0 (multimodal)
   - Explanation of OpenAI text-embedding-3-large (text-only)
   - Similarity metric description

✅ **Results Tables**
   - Image-to-image similarity comparison
   - Text-to-image similarity comparisons
   - Side-by-side Cohere vs OpenAI scores

✅ **Summary Statistics**
   - Embedding dimensions
   - Average, max, and min differences
   - Visual stat cards

✅ **Key Observations**
   - Analysis of differences
   - Explanation of why scores differ
   - Recommendations

✅ **Professional Formatting**
   - Clean, modern design
   - Color-coded sections
   - Easy to read tables
   - Suitable for academic submission

## Submission Checklist

Before submitting, make sure your report has:

- [ ] Both images displayed correctly
- [ ] All similarity scores present
- [ ] Tables are properly formatted
- [ ] Summary statistics visible
- [ ] Key observations included
- [ ] Professional appearance
- [ ] Your name/date (edit HTML if needed)

## Troubleshooting

**Images not showing in PDF?**
- Make sure you have internet connection when printing
- Images are loaded from SJSU URLs
- Try Method 2 (wkhtmltopdf) if browser method fails

**Tables look weird in PDF?**
- Use browser print preview to check formatting
- Adjust print margins to "Default" or "Minimum"
- Try a different browser (Chrome recommended)

**Need to customize?**
- Edit `comparison_report.html` in any text editor
- Update the name, date, or add notes
- Re-print to PDF

## Example Workflow

```bash
# 1. Generate the report
python3 ai_image_search.py
# → Select option 3

# 2. Open HTML in browser
open comparison_report.html  # macOS
xdg-open comparison_report.html  # Linux
start comparison_report.html  # Windows

# 3. Print to PDF using browser
# Cmd+P or Ctrl+P → Save as PDF

# 4. Submit to Canvas
# Upload the PDF file
```

## Additional Notes

- The HTML report is **not tracked in git** (excluded by .gitignore)
- Generate a fresh report before each submission
- The timestamp shows when the analysis was performed
- Images are embedded from SJSU URLs (require internet to view)

## Need Help?

- See [COMPARISON_MODE_GUIDE.md](COMPARISON_MODE_GUIDE.md) for detailed comparison mode info
- See [README.md](README.md) for general project documentation
- Check [API_CONFIGURATION_GUIDE.md](API_CONFIGURATION_GUIDE.md) for API setup

---

**Ready to submit!** Your comparison report is professional, comprehensive, and includes all required elements.
