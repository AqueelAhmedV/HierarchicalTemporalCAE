import os
import pypdf
import markdown
from pathlib import Path

def convert_pdf_to_markdown_or_txt(pdf_path):
    try:
        # Read PDF
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()

        # Try to convert to markdown
        try:
            md = markdown.markdown(text)
            output_dir = Path('./mds')
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / pdf_path.with_suffix('.md').name
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md)
            print(f"Converted {pdf_path} to markdown: {output_path}")
        except:
            # If markdown conversion fails, save as txt
            output_dir = Path('./txts')
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / pdf_path.with_suffix('.txt').name
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Converted {pdf_path} to text: {output_path}")
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")

def main():
    pdf_dir = Path('./pdfs')
    for pdf_file in pdf_dir.glob('*.pdf'):
        convert_pdf_to_markdown_or_txt(pdf_file)

if __name__ == "__main__":
    main()
