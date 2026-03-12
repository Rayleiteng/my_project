
import fitz  # PyMuPDF
import os
import time
import subprocess
from pathlib import Path

def run_baseline_extraction(pdf_path, output_dir):
    """
    [Task 1.1] Baseline: PyMuPDF extraction (Fast, but lossy)
    """
    print(f"--- [Baseline] Processing {os.path.basename(pdf_path)} ---")
    doc = fitz.open(pdf_path)
    text_content = []
    
    for i, page in enumerate(doc):
        text_content.append(f"\n--- PAGE {i+1} ---\n")
        text_content.append(page.get_text())

    # 保存 Baseline 结果
    base_name = os.path.basename(pdf_path).replace(".pdf", "_plain.txt")
    output_path = os.path.join(output_dir, base_name)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(text_content))
    print(f"✅ Baseline text saved to: {output_path}")

def run_nougat_extraction(pdf_path, output_dir):
    """
    [Task 1.2] Core: Nougat extraction (Slow, but preserves LaTeX)
    Wraps the 'nougat' CLI command.
    """
    print(f"\n--- [Core] Running Nougat on {os.path.basename(pdf_path)} ---")
    print("This may take a while (downloading weights or processing)...")
    
    start_time = time.time()
    
    command = [
        "nougat",
        pdf_path,
        "-o", output_dir,
        "--markdown"
    ]
    
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        
        base_name = os.path.basename(pdf_path).replace(".pdf", ".mmd")
        expected_output = os.path.join(output_dir, base_name)
        
        elapsed = time.time() - start_time
        print(f" Nougat extraction complete! Time: {elapsed:.2f}s")
        print(f" LaTeX/Markdown saved to: {expected_output}")
        
    except subprocess.CalledProcessError as e:
        print(" Error running Nougat:")
        print(e.stderr)
    except FileNotFoundError:
        print(" Error: 'nougat' command not found. Did you run 'pip install nougat-ocr'?")

if __name__ == "__main__":
    pdf_filename = "NotesMS.pdf"
    
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data" / "raw_pdfs" / pdf_filename
    
    baseline_output = project_root / "data" / "processed_text"
    latex_output = project_root / "data" / "processed_latex"
    
    os.makedirs(baseline_output, exist_ok=True)
    os.makedirs(latex_output, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: File not found at {input_path}")
    else:
        run_baseline_extraction(str(input_path), str(baseline_output))
        
        run_nougat_extraction(str(input_path), str(latex_output))