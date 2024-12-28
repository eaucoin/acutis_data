import os
import fitz
from tqdm import tqdm
import cv2
import numpy as np
from projectpackages.CRAFT import CRAFTModel
from layout import get_layout
from projectpackages.surya.layout import batch_layout_detection
import subprocess
import math
import shutil
import random

def ensure_folders_exist():
    folders = ['partitions', 'regionimages']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def polygon_to_bbox(polygon):
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
    return x1, y1, x2, y2

def calculate_bbox_height(bbox):
    _, y1, _, y2 = bbox
    return abs(y2 - y1)

def calculate_bbox_width(bbox):
    x1, _, x2, _ = bbox
    return abs(x2 - x1)

def draw_bounding_boxes(image, bboxes, average_height, average_width):
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        bbox_height = abs(y2 - y1)
        bbox_width = abs(x2 - x1)
        x_initial = int(x1 + (bbox_height / 4))
        x_final = int(x2 - (bbox_height / 4))
        
        vertical_center = (y1 + y2) / 2
        vertical_1 = int(vertical_center - (bbox_height))
        vertical_4 = int(vertical_center + (bbox_height))

        if x_initial < x_final:
            for x in range(int(x_initial), int(x_final), max(1, math.ceil(bbox_height/4))):
                cv2.putText(image, "O", (x * 2, int(vertical_center)), cv2.FONT_HERSHEY_SIMPLEX, bbox_height / 50, (0, 0, 0), math.ceil(bbox_height / 20))
    return image

def get_dataset_filename(original_pdf_filename, page_num):
   """Generate filename for dataset files using just the page number."""
   return f"{page_num}" # Will be appended with .html or .boxes by the calling function

def process_page_craft(args):
    """Process a single page with CRAFT. Separate function for multiprocessing."""
    chunk_page_num, page_num, page_folder, scaled_img, craft_word_model = args
    try:
        polygons = craft_word_model.get_polygons(scaled_img)
        craft_bboxes = [polygon_to_bbox(polygon) for polygon in polygons]
        
        heights = [calculate_bbox_height(bbox) for bbox in craft_bboxes]
        widths = [calculate_bbox_width(bbox) for bbox in craft_bboxes]
        average_height = sum(heights) / len(heights) if len(heights) > 0 else 0
        average_width = sum(widths) / len(widths) if len(widths) > 0 else 0
        
        denoised_image = np.ones((scaled_img.shape[0], scaled_img.shape[1] * 2, 3), dtype=np.uint8) * 255
        denoised_image = draw_bounding_boxes(denoised_image, craft_bboxes, average_height, average_width)
        
        denoised_path = os.path.join(page_folder, 'denoised.png')
        cv2.imwrite(denoised_path, denoised_image)
        
        return chunk_page_num, craft_bboxes
    except Exception as e:
        print(f"Error running CRAFT on page {page_num + 1}: {str(e)}")
        return chunk_page_num, []

def process_pdf(input_pdf, output_txt, craft_word_model, model, processor, det_model, det_processor, 
             table_model, table_processor, order_model, order_processor, chunk_num, chunk_size, 
             max_pages, is_dataset_mode=False, input_dir=None, original_pdf_filename=None):
    """
    Process a chunk of a PDF file with parallel CRAFT processing based on chunk_size.
    """
    # Ensure necessary folders exist in working directory
    working_dir = os.getcwd()
    ensure_folders_exist()
    # Create and clean the working directories for this chunk
    partitions_dir = os.path.join(working_dir, "partitions")
    regionimages_dir = os.path.join(working_dir, "regionimages")
    # Clean working directories
    for subfolder in os.listdir(partitions_dir):
        subfolder_path = os.path.join(partitions_dir, subfolder)
        if os.path.isdir(subfolder_path):
            shutil.rmtree(subfolder_path)
    for file in os.listdir(regionimages_dir):
        file_path = os.path.join(regionimages_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    import torch.multiprocessing as mp
    from tqdm import tqdm
    # Calculate page range for this chunk
    doc = fitz.open(input_pdf)
    start_page = (chunk_num - 1) * chunk_size
    end_page = min(start_page + chunk_size, min(doc.page_count, max_pages))
    pages_in_chunk = range(start_page, end_page)

    # Prepare all pages first
    page_data = []
    for chunk_page_num, page_num in enumerate(pages_in_chunk):
        page_folder = os.path.join(partitions_dir, f'{chunk_page_num:04d}')
        create_folder(page_folder)
        page = doc.load_page(page_num)
        zoom_factor = 4
        zoom_mat = fitz.Matrix(zoom_factor, zoom_factor)
        zoomed_pix = page.get_pixmap(matrix=zoom_mat)
        zoomed_img = cv2.cvtColor(
            np.frombuffer(zoomed_pix.samples, np.uint8).reshape(
                zoomed_pix.height, zoomed_pix.width, zoomed_pix.n
            ), 
            cv2.COLOR_BGR2RGB
        )
        scaled_height = 2048
        scaled_width = int(zoomed_img.shape[1] * (scaled_height / zoomed_img.shape[0]))
        scaled_img = cv2.resize(zoomed_img, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
        raw_path = os.path.join(page_folder, 'raw.png')
        cv2.imwrite(raw_path, scaled_img)

        page_data.append((chunk_page_num, page_num, page_folder, scaled_img, craft_word_model))
    doc.close()
    # Process pages in parallel using chunk_size
    all_craft_bboxes = [None] * len(page_data)

    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)

    with mp.Pool(processes=min(chunk_size, len(page_data))) as pool:
        for chunk_page_num, craft_bboxes in tqdm(
            pool.imap(process_page_craft, page_data),
            total=len(page_data),
            desc='Processing Pages'
        ):
            all_craft_bboxes[chunk_page_num] = craft_bboxes
    # Continue with layout parsing
    print("Running layout parsing")
    get_layout(partitions_dir, model, processor, det_model, det_processor, table_model, 
             table_processor, order_model, order_processor, all_craft_bboxes,
             input_dir,
             start_page,
             is_dataset_mode,
             chunk_size)
    print("Running OCR")
    subprocess.run(['node', 'ocr.js', output_txt, str(chunk_num), str(chunk_size), input_dir], 
                 universal_newlines=True)
    print(f"Chunk {chunk_num} processing completed.")
