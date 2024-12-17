import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'projectpackages'))
import argparse
import fitz
import subprocess
import shutil
from process_document import process_pdf
import torch
from projectpackages.CRAFT import CRAFTModel
from projectpackages.surya.model.detection.model import load_model, load_processor
from projectpackages.surya.settings import settings
from projectpackages.surya.model.ordering.processor import load_processor as load_ordering_processor
from projectpackages.surya.model.ordering.model import load_model as load_ordering_model
from projectpackages.surya.model.table_rec.model import load_model as load_table_model
from projectpackages.surya.model.table_rec.processor import load_processor as load_table_processor
import traceback
import logging
from bs4 import BeautifulSoup
from internetarchive import download, get_files, get_session

def ensure_folders_exist():
    folders = ['partitions', 'regionimages', 'shelves', 'weights']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def delete_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def delete_shelves_contents():
    shelves_dir = os.path.join(os.getcwd(), 'shelves')
    for item in os.listdir(shelves_dir):
        item_path = os.path.join(shelves_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def split_pdf(input_pdf, chunk_size):
    doc = fitz.open(input_pdf)
    num_pages = doc.page_count
    num_chunks = (num_pages + chunk_size - 1) // chunk_size

    num_digits = len(str(num_chunks))

    pdf_name = os.path.splitext(os.path.basename(input_pdf))[0]
    shelves_dir = os.path.join(os.getcwd(), 'shelves')
    os.makedirs(shelves_dir, exist_ok=True)
    chunk_input_dir = os.path.join(shelves_dir, f'chunk_input_{pdf_name}')
    os.makedirs(chunk_input_dir, exist_ok=True)

    for i in range(num_chunks):
        start_page = i * chunk_size
        end_page = min((i + 1) * chunk_size, num_pages)
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page-1)
        
        chunk_num = str(i + 1).zfill(num_digits)
        chunk_output = os.path.join(chunk_input_dir, f"chunk_{chunk_num}.pdf")
        
        chunk_doc.save(chunk_output)
        chunk_doc.close()

    doc.close()
    return num_chunks, chunk_size

def process_chunks(chunk_output_dir, output_txt, create_single_file=True):
    chunk_files = sorted([f for f in os.listdir(chunk_output_dir) if f.endswith(".html")])
    combined_text = ""

    for chunk_file in chunk_files:
        with open(os.path.join(chunk_output_dir, chunk_file), "r", encoding="utf-8") as f:
            chunk_text = f.read()
            combined_text += chunk_text + "\n"

    if create_single_file:
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(combined_text)
    else:
        return combined_text

def get_dataset_filename(original_pdf_filename, page_num):
    base_filename = os.path.splitext(os.path.basename(original_pdf_filename))[0]
    return f"{base_filename}_{page_num}.html"

def process_dataset_output(combined_html, original_pdf_filename, output_dir):
    soup = BeautifulSoup(combined_html, 'html.parser')
    pages = soup.find_all('div', class_='page')
    
    for i, page in enumerate(pages, start=1):
        # Remove the page-number div
        page_number_div = page.find('div', class_='page-number')
        if page_number_div:
            page_number_div.decompose()
        
        # Remove all base64-encoded images (pictures and figures)
        for img in page.find_all('img'):
            if img.get('src', '').startswith('data:image'):
                img.decompose()
        
        # Remove empty figure elements
        for figure in page.find_all('figure'):
            if not figure.contents:
                figure.decompose()
        
        # Create the HTML output
        page_html = f"{str(page)}"
        
        output_filename = get_dataset_filename(original_pdf_filename, i)
        with open(os.path.join(output_dir, output_filename), "w", encoding="utf-8") as f:
            f.write(page_html)

    os.remove(original_pdf_filename)

def is_valid_pdf(filename):
    """Check if a PDF filename is valid (not ending in _bw or _text)"""
    base = os.path.splitext(filename)[0]
    return (not base.endswith('_bw') and 
            not base.endswith('_text') and 
            filename.lower().endswith('.pdf'))

def check_processed(base_name, output_dir):
    """Check if a document has been processed by looking for first pair of files"""
    html_file = os.path.join(output_dir, f"{base_name}_1.html")
    boxes_file = os.path.join(output_dir, f"{base_name}_1.boxes")
    return os.path.exists(html_file) and os.path.exists(boxes_file)

def process_archive_documents(output_dir, chunk_size, models):
    """Process documents from archive.org based on identifiers.txt"""
    identifiers_file = os.path.join(project_root, 'identifiers.txt')
    if not os.path.exists(identifiers_file):
        logging.error("identifiers.txt not found in root directory")
        return

    # Read identifiers
    with open(identifiers_file, 'r') as f:
        identifiers = [line.strip() for line in f if line.strip()]

    session = get_session()

    for identifier in identifiers:
        logging.info(f"Processing identifier: {identifier}")
        try:
            # Get all files for this identifier
            files = list(get_files(identifier))
            
            # Filter for valid PDFs
            pdf_files = [f for f in files if is_valid_pdf(f.name)]
            
            for pdf_file in pdf_files:
                base_name = os.path.splitext(pdf_file.name)[0]
                
                # Skip if already processed
                if check_processed(base_name, output_dir):
                    logging.info(f"Skipping {pdf_file.name} - already processed")
                    continue

                # Download the PDF
                logging.info(f"Downloading {pdf_file.name}")
                download_path = os.path.join(output_dir, pdf_file.name)
                pdf_file.download(download_path)

                # Process the PDF
                process_single_document(download_path, output_dir, chunk_size, models)

        except Exception as e:
            logging.error(f"Error processing identifier {identifier}: {str(e)}")
            continue

def process_single_document(input_pdf, output_dir, chunk_size, models):
    """Process a single PDF document"""
    try:
        output_txt = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_pdf))[0]}.html")

        if not os.path.isfile(input_pdf):
            raise FileNotFoundError(f"Input PDF file not found: {input_pdf}")

        num_chunks, chunk_size = split_pdf(input_pdf, chunk_size)

        pdf_name = os.path.splitext(os.path.basename(input_pdf))[0]
        chunk_input_dir = os.path.join(os.getcwd(), 'shelves', f'chunk_input_{pdf_name}')
        chunk_output_dir = os.path.join(os.getcwd(), 'shelves', f'chunk_output_{pdf_name}')
        os.makedirs(chunk_output_dir, exist_ok=True)

        chunk_files = sorted([f for f in os.listdir(chunk_input_dir) if f.endswith(".pdf")])
        for chunk_num, chunk_file in enumerate(chunk_files, start=1):
            chunk_input = os.path.join(chunk_input_dir, chunk_file)
            chunk_output = os.path.join(chunk_output_dir, f"{os.path.splitext(chunk_file)[0]}.html")
            logging.info(f"Processing chunk: {chunk_file}")
            process_pdf(chunk_input, chunk_output, models['craft_word_model'], 
                      models['model'], models['processor'], 
                      models['det_model'], models['det_processor'],
                      models['table_model'], models['table_processor'], 
                      models['order_model'], models['order_processor'],
                      chunk_num, chunk_size, True, output_dir, input_pdf)

        combined_html = process_chunks(chunk_output_dir, output_txt, create_single_file=False)
        process_dataset_output(combined_html, input_pdf, output_dir)

        delete_files_in_directory(chunk_input_dir)
        delete_files_in_directory(chunk_output_dir)
        
        logging.info(f"Successfully processed {os.path.basename(input_pdf)}")

    except Exception as e:
        logging.error(f"Error processing {os.path.basename(input_pdf)}:")
        logging.error(f"Error type: {type(e).__name__}")
        logging.error(f"Error message: {str(e)}")
        logging.error("Traceback:", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to the input directory containing PDF files")
    parser.add_argument("--output_dir", help="Path to the output directory for text files (optional)")
    parser.add_argument("--chunk_size", type=int, default=10, help="Number of pages per chunk (default: 10)")
    parser.add_argument("--replace", action="store_true", help="Replace input PDFs with output text files")
    parser.add_argument("--dataset", action="store_true", help="Create individual HTML files for each page")
    args = parser.parse_args()

    ensure_folders_exist()
    delete_shelves_contents()

    if not args.replace and not args.dataset and not args.output_dir:
        parser.error("Either --output_dir, --replace, or --dataset must be specified.")
    
    if args.replace or args.dataset:
        output_dir = args.input_dir
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

    # Initialize models
    models = {
        'craft_word_model': CRAFTModel('weights/', torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                                     use_refiner=False, fp16=True),
        'model': load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT),
        'processor': load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT),
        'det_model': load_model(),
        'det_processor': load_processor(),
        'order_model': load_ordering_model(),
        'order_processor': load_ordering_processor(),
        'table_model': load_table_model(),
        'table_processor': load_table_processor()
    }

    if args.dataset:
        process_archive_documents(output_dir, args.chunk_size, models)
    else:
        # Original functionality for processing local PDFs
        pdf_files = [f for f in os.listdir(args.input_dir) if f.endswith(".pdf")]
        for pdf_file in pdf_files:
            input_pdf = os.path.join(args.input_dir, pdf_file)
            try:
                process_single_document(input_pdf, output_dir, args.chunk_size, models)
                if args.replace:
                    os.remove(input_pdf)
            except Exception as e:
                print(f"Error processing {pdf_file}:", str(e))
                continue

    delete_shelves_contents()

if __name__ == "__main__":
    main()