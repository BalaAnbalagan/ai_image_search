"""
PDF Generation Script for Assignment Report
Run this script to convert assignment_report.html to PDF

Installation:
pip install weasyprint

OR open assignment_report.html in browser and print to PDF
"""

try:
    from weasyprint import HTML

    print("Generating PDF from HTML...")
    HTML('assignment_report.html').write_pdf('assignment_report.pdf')
    print("✓ PDF generated successfully: assignment_report.pdf")

except ImportError:
    print("=" * 80)
    print("WeasyPrint not installed.")
    print("=" * 80)
    print("\nOption 1: Install WeasyPrint")
    print("  pip install weasyprint")
    print("  python generate_pdf.py")
    print("\nOption 2: Use Browser")
    print("  1. Open assignment_report.html in your web browser")
    print("  2. Press Ctrl+P (or Cmd+P on Mac)")
    print("  3. Select 'Save as PDF' as destination")
    print("  4. Save as 'assignment_report.pdf'")
    print("=" * 80)
