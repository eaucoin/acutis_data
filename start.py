import sys
import time
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

MAX_PAGES = 16

def update_dashboard(input_dir):
    """Update dashboard.txt with current processing statistics."""
    try:
        # Initialize counters
        identifiers_with_docs = 0
        total_processed_docs = 0
        total_pages = 0
        
        # Go through each identifier directory
        for identifier in os.listdir(input_dir):
            identifier_path = os.path.join(input_dir, identifier)
            if not os.path.isdir(identifier_path):
                continue
                
            # Check each document directory in this identifier
            docs_with_pages = 0
            for doc in os.listdir(identifier_path):
                doc_path = os.path.join(identifier_path, doc)
                if not os.path.isdir(doc_path):
                    continue
                    
                # Count html files (which equals number of pages)
                html_files = len([f for f in os.listdir(doc_path) if f.endswith('.html')])
                if html_files > 0:
                    docs_with_pages += 1
                    total_pages += html_files
            
            if docs_with_pages > 0:
                identifiers_with_docs += 1
                total_processed_docs += docs_with_pages
        
        # Write statistics to dashboard.txt
        dashboard_path = os.path.join(input_dir, 'dashboard.txt')
        with open(dashboard_path, 'w') as f:
            f.write(f"Processing Statistics\n")
            f.write(f"====================\n")
            f.write(f"Identifiers with processed documents: {identifiers_with_docs}\n")
            f.write(f"Total documents processed: {total_processed_docs}\n")
            f.write(f"Total pages processed: {total_pages}\n")
            f.write(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
    except Exception as e:
        print(f"Error updating dashboard: {str(e)}")

def ensure_folders_exist():
   folders = ['partitions', 'regionimages', 'shelves', 'weights']
   for folder in folders:
       os.makedirs(folder, exist_ok=True)

def delete_files_in_directory(directory):
   """Safely delete all files in a directory if it exists."""
   try:
       if os.path.exists(directory):
           for filename in os.listdir(directory):
               file_path = os.path.join(directory, filename)
               if os.path.isfile(file_path):
                   os.remove(file_path)
               elif os.path.isdir(file_path):
                   shutil.rmtree(file_path)
   except Exception as e:
       print(f"Warning: Error cleaning directory {directory}: {str(e)}")

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
   num_pages = min(doc.page_count, MAX_PAGES)  # Limit to MAX_PAGES
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

def process_dataset_output(combined_html, output_dir):
   """Process HTML and write page files to PDF-specific directory"""
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
       
       # Create the HTML output with just page number as filename
       page_html = f"{str(page)}"
       output_filename = f"{i}.html"
       output_path = os.path.join(output_dir, output_filename)
       
       with open(output_path, "w", encoding="utf-8") as f:
           f.write(page_html)

def is_valid_pdf(filename):
   """Check if a PDF filename is valid (not ending in _bw or _text)"""
   base = os.path.splitext(filename)[0]
   return (not base.endswith('_bw') and 
           not base.endswith('_text') and 
           filename.lower().endswith('.pdf'))

def read_identifier_status(identifiers_file):
   """Read identifiers and their statuses from file"""
   identifiers_dict = {}
   if os.path.exists(identifiers_file):
       with open(identifiers_file, 'r') as f:
           for line in f:
               parts = line.strip().split(',')
               if len(parts) == 2:
                   identifiers_dict[parts[0]] = parts[1]
               elif len(parts) == 1:
                   identifiers_dict[parts[0]] = ""
   return identifiers_dict

def update_identifier_status(identifiers_file, identifier, status):
   """Update status for a single identifier in the file"""
   identifiers_dict = read_identifier_status(identifiers_file)
   identifiers_dict[identifier] = status
   
   with open(identifiers_file, 'w') as f:
       for ident, stat in identifiers_dict.items():
           if stat:
               f.write(f"{ident},{stat}\n")
           else:
               f.write(f"{ident}\n")

def check_processed(base_name, output_dir):
   """Check if a document has been processed by looking for first pair of files"""
   html_file = os.path.join(output_dir, "1.html")
   boxes_file = os.path.join(output_dir, "1.boxes")
   return os.path.exists(html_file) and os.path.exists(boxes_file)

def find_pdfs_in_directory(directory):
   """Find all PDF files in directory and its subdirectories"""
   found_pdfs = []
   for root, dirs, files in os.walk(directory):
       for file in files:
           if file.lower().endswith('.pdf'):
               found_pdfs.append(os.path.join(root, file))
   return found_pdfs

def process_archive_documents(output_dir, chunk_size, models):
    """Process documents from archive.org one identifier at a time"""
    identifiers_file = os.path.join(project_root, 'identifiers.txt')
    print("Starting process_archive_documents")
    if not os.path.exists(identifiers_file):
        print("identifiers.txt not found in root directory")
        return

    identifiers_dict = read_identifier_status(identifiers_file)
    print(f"Found {len(identifiers_dict)} identifiers to process")

    session = get_session()
    print("Created Internet Archive session")

    for identifier, status in identifiers_dict.items():
        print(f"\nStarting to process identifier: {identifier}")
        if status in ["Done", "No Eligible Documents"]:
            print(f"Skipping {identifier} - Status: {status}")
            continue

        try:
            print(f"Creating directory for {identifier}")
            identifier_dir = os.path.join(output_dir, identifier)
            os.makedirs(identifier_dir, exist_ok=True)

            print(f"Getting files for {identifier}")
            files = list(get_files(identifier))
            pdf_files = [f for f in files if is_valid_pdf(f.name)]
            print(f"Found {len(pdf_files)} eligible PDFs for {identifier}")
            print(f"PDF files: {[f.name for f in pdf_files]}")
            
            if not pdf_files:
                update_identifier_status(identifiers_file, identifier, "No Eligible Documents")
                continue

            # Check if all PDFs are already processed
            all_processed = True
            for pdf_file in pdf_files:
                pdf_base_name = os.path.splitext(pdf_file.name)[0]
                pdf_output_dir = os.path.join(identifier_dir, pdf_base_name)
                os.makedirs(pdf_output_dir, exist_ok=True)
                
                if not os.path.exists(pdf_output_dir) or not check_processed(pdf_base_name, pdf_output_dir):
                    all_processed = False
                    break

            if all_processed:
                update_identifier_status(identifiers_file, identifier, "Done")
                continue
                
            # Process PDFs one at a time
            for pdf_file in pdf_files:
                pdf_name = os.path.splitext(pdf_file.name)[0]
                pdf_output_dir = os.path.join(identifier_dir, pdf_name)
                os.makedirs(pdf_output_dir, exist_ok=True)
                
                if check_processed(pdf_name, pdf_output_dir):
                    print(f"Skipping {pdf_file.name} - already processed")
                    continue

                print(f"Downloading {pdf_file.name}")
                ia_subdir = os.path.join(identifier_dir, identifier)
                pdf_path = os.path.join(ia_subdir, pdf_file.name)

                # Clean any existing PDF with same name if it exists
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)

                print(f"Waiting for PDF at path: {pdf_path}")
                if os.path.exists(ia_subdir):
                    print(f"Subdir contents before download: {os.listdir(ia_subdir)}")
                
                # Download and wait for this specific PDF
                download(identifier, files=[pdf_file.name], destdir=identifier_dir)
                
                # Wait for this specific PDF with timeout
                pdf_found = False
                timeout = 300
                start_time = time.time()
                while not pdf_found and time.time() - start_time < timeout:
                    time.sleep(1)
                    if os.path.exists(ia_subdir):
                        subdir_contents = os.listdir(ia_subdir)
                        if pdf_file.name in subdir_contents:
                            # Additional check: wait until file size stabilizes
                            initial_size = os.path.getsize(pdf_path)
                            time.sleep(2)  # Wait 2 seconds
                            if os.path.getsize(pdf_path) == initial_size:
                                pdf_found = True
                                break
                    if time.time() - start_time > 5:  # After 5 seconds
                        print(f"Still waiting... Current subdir contents: {os.listdir(ia_subdir) if os.path.exists(ia_subdir) else 'subdir not found'}")

                if not pdf_found:
                    print(f"Failed to download {pdf_file.name}")
                    if os.path.exists(ia_subdir):
                        print(f"Final subdir contents: {os.listdir(ia_subdir)}")
                    continue

                print(f"PDF found and download complete, processing {pdf_file.name}")
                process_single_document(pdf_path, pdf_output_dir, chunk_size, models)
                os.remove(pdf_path)  # Remove PDF after processing
                print(f"Completed processing {pdf_file.name}")

            # Check if all PDFs were processed
            all_processed = True
            for pdf_file in pdf_files:
                pdf_name = os.path.splitext(pdf_file.name)[0]
                pdf_output_dir = os.path.join(identifier_dir, pdf_name)
                if not check_processed(pdf_name, pdf_output_dir):
                    all_processed = False
                    break

            if all_processed:
                update_identifier_status(identifiers_file, identifier, "Done")

        except Exception as e:
            print(f"Error processing identifier {identifier}: {str(e)}")
            traceback.print_exc()
            continue

def process_single_document(input_pdf, output_dir, chunk_size, models):
   """Process a single PDF document with outputs going to a PDF-specific directory"""
   try:
       if not os.path.isfile(input_pdf):
           raise FileNotFoundError(f"Input PDF file not found: {input_pdf}")

       # Get number of chunks needed
       doc = fitz.open(input_pdf)
       total_pages = min(doc.page_count, MAX_PAGES)  # Apply page limit
       doc.close()
       num_chunks = (total_pages + chunk_size - 1) // chunk_size

       # Setup working directories in project directory
       pdf_name = os.path.splitext(os.path.basename(input_pdf))[0]
       working_dir = os.getcwd()
       chunk_input_dir = os.path.join(working_dir, 'shelves', f'chunk_input_{pdf_name}')
       chunk_output_dir = os.path.join(working_dir, 'shelves', f'chunk_output_{pdf_name}')
       
       # Create working directories
       os.makedirs(chunk_input_dir, exist_ok=True)
       os.makedirs(chunk_output_dir, exist_ok=True)

       # Process each chunk
       for chunk_num in range(1, num_chunks + 1):
           print(f"Processing chunk {chunk_num} of {num_chunks}")
           chunk_output = os.path.join(chunk_output_dir, f"chunk_{chunk_num}.html")
           
           # Process chunk with output going to PDF-specific directory
           process_pdf(input_pdf, chunk_output, 
                   models['craft_word_model'], 
                   models['model'], models['processor'], 
                   models['det_model'], models['det_processor'],
                   models['table_model'], models['table_processor'], 
                   models['order_model'], models['order_processor'],
                   chunk_num, chunk_size, MAX_PAGES, True, output_dir, input_pdf)

       # Clean up temporary directories if they exist
       if os.path.exists(chunk_input_dir):
           shutil.rmtree(chunk_input_dir)
       if os.path.exists(chunk_output_dir):
           shutil.rmtree(chunk_output_dir)

       print("All chunks processed successfully")

       # Update dashboard after successful processing
       input_dir = os.path.dirname(os.path.dirname(output_dir))  # Go up two levels to get input_dir
       update_dashboard(input_dir)

   except Exception as e:
       print(f"Error processing {os.path.basename(input_pdf)}: {str(e)}")
       traceback.print_exc()
       # Cleanup on error
       if os.path.exists(chunk_input_dir):
           shutil.rmtree(chunk_input_dir)
       if os.path.exists(chunk_output_dir):
           shutil.rmtree(chunk_output_dir)
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