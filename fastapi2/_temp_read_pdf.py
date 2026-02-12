
import sys
import os

# Set explicit path
pdf_path = r"c:\Users\KOUSTAV BERA\OneDrive\Desktop\chiranjeevi\fastapi2\reports\PR-20260211-115116.pdf"

print(f"Attempting to read: {pdf_path}")

# Try pypdf
try:
    import pypdf
    print("Using pypdf...")
    reader = pypdf.PdfReader(pdf_path)
    print(f"--- START PDF CONTENT ---")
    for i, page in enumerate(reader.pages):
        print(f"--- PAGE {i+1} ---")
        print(page.extract_text())
    print(f"--- END PDF CONTENT ---")
    sys.exit(0)
except ImportError:
    pass
except Exception as e:
    print(f"pypdf error: {e}")

# Try PyPDF2
try:
    import PyPDF2
    print("Using PyPDF2...")
    reader = PyPDF2.PdfReader(pdf_path)
    print(f"--- START PDF CONTENT ---")
    for i, page in enumerate(reader.pages):
        print(f"--- PAGE {i+1} ---")
        print(page.extract_text())
    print(f"--- END PDF CONTENT ---")
    sys.exit(0)
except ImportError:
    pass
except Exception as e:
    print(f"PyPDF2 error: {e}")

print("Could not import pypdf or PyPDF2. Please ensure one is installed (pip install pypdf).")
