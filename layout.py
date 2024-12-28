from PIL import Image
import cv2
import numpy as np
import os
from tqdm import tqdm
from projectpackages.surya.detection import batch_text_detection
from projectpackages.surya.layout import batch_layout_detection
from projectpackages.surya.ordering import batch_ordering
from projectpackages.surya.tables import batch_table_recognition
from projectpackages.surya.schema import TextDetectionResult, PolygonBox, ColumnLine

def extract_bounding_boxes(layout_predictions, is_sparse=False):
   bounding_boxes = []
   for layout_result in layout_predictions:
       for layout_box in layout_result.bboxes:
           label = layout_box.label
           bounding_boxes.append(layout_box.bbox + [label])
   return bounding_boxes

def filter_text_labels(bounding_boxes):
   return [box for box in bounding_boxes if box[4] != 'Text']

def shrink_bbox_horizontally(bbox):
   x1, y1, x2, y2 = bbox
   shift_value = abs(x1 - x2) / 3
   new_x1 = x1 + shift_value
   new_x2 = x2 - shift_value
   return [new_x1, y1, new_x2, y2]

def get_shortened_label(label):
   label_map = {
       'Caption': 'c', 'Footnote': 'fo', 'Formula': 'fr', 'List-item': 'l',
       'Page-footer': 'pf', 'Page-header': 'ph', 'Picture': 'p', 'Figure': 'f',
       'Section-header': 's', 'Table': 'a', 'Text': 'e', 'Title': 'i'
   }
   return label_map.get(label, 'u')

def assign_boxIDs(bounding_boxes):
   return [bbox + [i] for i, bbox in enumerate(bounding_boxes)]

def adjust_bounding_boxes_final(boxes, image_bbox):
   x_max, y_max = image_bbox[2], image_bbox[3]
   
   adjusted_boxes_vertical = [box.copy() for box in boxes]
   adjusted_boxes_horizontal = [box.copy() for box in boxes]

   while True:
       changes_made = False
       for i, box1 in enumerate(adjusted_boxes_vertical):
           x_1l, y_1b, x_1r, y_1t, label, box_id, position = box1
           if label in ['Picture', 'Figure']:
               continue
           intersection_found = False
           for j, box2 in enumerate(adjusted_boxes_vertical):
               if i != j and rectOverlap(box1[:4], box2[:4]):
                   intersection_found = True
                   break
           if not intersection_found:
               y_1t_new = y_1t + 1 if y_1t < y_max else y_1t
               y_1b_new = y_1b - 1 if y_1b > 0 else y_1b

               if y_1t_new != y_1t or y_1b_new != y_1b:
                   adjusted_boxes_vertical[i] = [x_1l, y_1b_new, x_1r, y_1t_new, label, box_id, position]
                   changes_made = True
       if not changes_made:
           break

   while True:
       changes_made = False
       for i, box1 in enumerate(adjusted_boxes_horizontal):
           x_1l, y_1b, x_1r, y_1t, label, box_id, position = box1
           if label in ['Picture', 'Figure']:
               continue
           intersection_found = False
           for j, box2 in enumerate(adjusted_boxes_horizontal):
               if i != j and rectOverlap(box1[:4], box2[:4]):
                   intersection_found = True
                   break
           if not intersection_found:
               x_1l_new = x_1l - 1 if x_1l > 0 else x_1l
               x_1r_new = x_1r + 1 if x_1r < x_max else x_1r
               if x_1l_new != x_1l or x_1r_new != x_1r:
                   adjusted_boxes_horizontal[i] = [x_1l_new, y_1b, x_1r_new, y_1t, label, box_id, position]
                   changes_made = True
       if not changes_made:
           break

   final_adjusted_boxes = []
   for box_vertical, box_horizontal in zip(adjusted_boxes_vertical, adjusted_boxes_horizontal):
       x_1l_v, y_1b_v, x_1r_v, y_1t_v, label_v, box_id_v, position_v = box_vertical
       x_1l_h, y_1b_h, x_1r_h, y_1t_h, label_h, box_id_h, position_h = box_horizontal
       if label_v in ['Picture', 'Figure']:
           final_box = box_vertical
       else:
           final_box = [x_1l_h, y_1b_v, x_1r_h, y_1t_v, label_v, box_id_v, position_v]
       final_adjusted_boxes.append(final_box)

   return final_adjusted_boxes

def valueInRange(value, min_val, max_val):
   return (min_val <= value <= max_val)

def rectOverlap(box1, box2):
   x1_l, y1_b, x1_r, y1_t = box1[:4]
   x2_l, y2_b, x2_r, y2_t = box2[:4]

   xOverlap = valueInRange(x1_l, x2_l, x2_r) or valueInRange(x2_l, x1_l, x1_r)
   yOverlap = valueInRange(y1_b, y2_b, y2_t) or valueInRange(y2_b, y1_b, y1_t)

   return xOverlap and yOverlap

def full_encapsulation(primary_box, secondary_box):
   x11, y11, x12, y12 = primary_box[:4]
   x21, y21, x22, y22 = secondary_box[:4]
   return (x11 <= x21 < x22 <= x12) and (y11 <= y21 < y22 <= y12)

def consolidate_regions(sparse_line_regions, layout_regions):
   to_delete_from_layout = set()
   to_delete_from_sparse = set()

   for i, primary_box in enumerate(sparse_line_regions):
       for j, secondary_box in enumerate(layout_regions):
           if (primary_box[4] in ['Title', 'Section-header', 'Page-header', 'List-item'] and rectOverlap(primary_box[:4], secondary_box[:4]) and 2 * abs(int(primary_box[1]) - int(primary_box[3])) > abs(int(secondary_box[1]) - int(secondary_box[3]))) or (primary_box[4] in ['Table', 'Caption', 'Footnote', 'Page-footer'] and full_encapsulation(primary_box[:4], secondary_box[:4])):
               to_delete_from_layout.add(j)

   consolidated_regions = [box for i, box in enumerate(sparse_line_regions) if i not in to_delete_from_sparse]
   consolidated_regions += [box for i, box in enumerate(layout_regions) if i not in to_delete_from_layout]

   return consolidated_regions

def write_normalized_boxes(boxes, image_width, image_height, output_path):
   """Write normalized bounding boxes to a .boxes file"""
   normalized_boxes = []
   for box in boxes:
       if len(box) >= 7:  # Regular box with x1,y1,x2,y2,label,box_id,position
           x1, y1, x2, y2, label, box_id, position = box[:7]
           norm_box = [
               x1/image_width,
               y1/image_height,
               x2/image_width,
               y2/image_height,
               label,
               box_id,
               position
           ]
           normalized_boxes.append(norm_box)
       elif len(box) >= 8:  # Table cell with row_id,col_id
           x1, y1, x2, y2, label, filename, row_id, col_id = box[:8]
           norm_box = [
               x1/image_width,
               y1/image_height,
               x2/image_width,
               y2/image_height,
               label,
               row_id,
               col_id
           ]
           normalized_boxes.append(norm_box)
   
   with open(output_path, 'w', encoding='utf-8') as f:
       for box in normalized_boxes:
           f.write(f"{box}\n")

def get_layout(partitions_directory, model, processor, det_model, det_processor, 
              table_model, table_processor, order_model, order_processor, craft_bboxes,
              input_dir=None, start_page=None, is_dataset_mode=False, chunk_size=None):
   """Process document layout for a specific chunk of pages."""
   
   partitions_directory = os.path.join(os.getcwd(), 'partitions')
   page_folders = sorted([folder for folder in os.listdir(partitions_directory) if folder.isdigit()], 
                        key=lambda x: int(x))

   denoised_images = []
   normal_images = []
   raw_images = []

   # Load images for this chunk
   for page_folder in page_folders:
       raw_image_path = os.path.join(partitions_directory, page_folder, 'raw.png')
       denoised_image_path = os.path.join(partitions_directory, page_folder, 'denoised.png')
       
       denoised_image = Image.open(denoised_image_path)
       raw_image = cv2.imread(raw_image_path)
       normal_image = Image.open(raw_image_path)

       normal_width, normal_height = normal_image.size
       denoised_image = denoised_image.resize((normal_width, normal_height), Image.LANCZOS)
       
       denoised_images.append(denoised_image)
       normal_images.append(normal_image)
       raw_images.append(raw_image)

   # Create craft_line_predictions
   craft_line_predictions = []
   for normal_image, page_craft_bboxes in zip(normal_images, craft_bboxes):
       bboxes = []
       for craft_bbox in page_craft_bboxes:
           x1, y1, x2, y2 = craft_bbox
           polygon_box = PolygonBox(
               polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
               confidence=0.6
           )
           bboxes.append(polygon_box)
       
       craft_line_prediction = TextDetectionResult(
           bboxes=bboxes,
           vertical_lines=[],
           heatmap=None,
           affinity_map=None,
           image_bbox=[0, 0, normal_image.width, normal_image.height]
       )
       craft_line_predictions.append(craft_line_prediction)

   dense_line_predictions = batch_text_detection(denoised_images, det_model, det_processor)
   sparse_line_predictions = batch_text_detection(normal_images, det_model, det_processor)
   layout_predictions = batch_layout_detection(denoised_images, model, processor, dense_line_predictions)
   sparse_layout_predictions = batch_layout_detection(normal_images, model, processor, sparse_line_predictions)

   original_layout_regions_scaled_list = []
   normal_image_cv_list = []
   vertically_swelled_layout_regions_scaled_list = []
   normal_image_cv_original_list = []
   vertically_swelled_layout_regions_coords_list = []

   for page_number, normal_image, denoised_image, layout_prediction, sparse_layout_prediction, sparse_line_prediction, craft_line_prediction, page_craft_bboxes in tqdm(zip(range(len(page_folders)), normal_images, denoised_images, layout_predictions, sparse_layout_predictions, sparse_line_predictions, craft_line_predictions, craft_bboxes), total=len(page_folders), desc="Processing Pages"):
       layout_regions = extract_bounding_boxes([layout_prediction], is_sparse=False)
       sparse_line_regions = extract_bounding_boxes([sparse_layout_prediction], is_sparse=True)
       
       image_bbox = [0, 0, normal_image.width, normal_image.height]

       filtered_adjusted_sparse_line_regions = filter_text_labels(sparse_line_regions)
       
       consolidated_regions = consolidate_regions(filtered_adjusted_sparse_line_regions, layout_regions)
       consolidated_regions_with_ids = assign_boxIDs(consolidated_regions)
       
       if len(consolidated_regions_with_ids) >= 255:
           x_min = min(bbox[0] for bbox in consolidated_regions_with_ids)
           y_min = min(bbox[1] for bbox in consolidated_regions_with_ids)
           x_max = max(bbox[2] for bbox in consolidated_regions_with_ids)
           y_max = max(bbox[3] for bbox in consolidated_regions_with_ids)
           consolidated_regions_with_ids = [[x_min, y_min, x_max, y_max, 'Text', 0]]

       original_layout_regions_scaled = consolidated_regions_with_ids
       
       normal_image_cv = cv2.cvtColor(np.array(normal_image), cv2.COLOR_RGB2BGR)
   
       vertically_swelled_layout_regions_scaled = [bbox.copy() for bbox in original_layout_regions_scaled]
   
       vertically_swelled_layout_regions_coords = [bbox[:4] for bbox in vertically_swelled_layout_regions_scaled]
   
       original_layout_regions_scaled_list.append(original_layout_regions_scaled)
       normal_image_cv_list.append(normal_image_cv)
       normal_image_cv_original_list.append(normal_image_cv.copy())
       vertically_swelled_layout_regions_scaled_list.append(vertically_swelled_layout_regions_scaled)
       vertically_swelled_layout_regions_coords_list.append(vertically_swelled_layout_regions_coords)
   
   order_predictions = batch_ordering(normal_images, vertically_swelled_layout_regions_coords_list, order_model, order_processor)

   original_layout_regions_scaled_with_position_list = []

   for page_number, original_layout_regions_scaled, vertically_swelled_layout_regions_scaled_original, order_prediction, raw_image, normal_image_cv, sparse_line_prediction in tqdm(zip(range(len(page_folders)), original_layout_regions_scaled_list, vertically_swelled_layout_regions_scaled_list, order_predictions, raw_images, normal_image_cv_list, sparse_line_predictions), total=len(page_folders), desc="Extracting Region Images"):
       ordered_original_layout_regions_scaled = []
       
       coord_to_box_map = {tuple(box[:4]): box for box in vertically_swelled_layout_regions_scaled_original}

       for order_box in order_prediction.bboxes:
           bbox_tuple = tuple(order_box.bbox)
           position = order_box.position

           if bbox_tuple in coord_to_box_map:
               original_box = coord_to_box_map[bbox_tuple]
               x1, y1, x2, y2, label, box_id = original_box
               ordered_original_layout_regions_scaled.append([x1, y1, x2, y2, label, box_id, position])

       ordered_original_layout_regions_scaled.sort(key=lambda x: x[6])

       normal_cv_height, normal_cv_width, _ = normal_image_cv.shape
       image_bbox = [0, 0, normal_cv_width, normal_cv_height]

       # Generate .boxes file for dataset mode before adjustment
       if is_dataset_mode and input_dir:
           try:
               # Calculate actual page number based on start_page of chunk
               actual_page_num = (start_page or 0) + page_number + 1
               boxes_filename = f"{actual_page_num}.boxes"
               boxes_path = os.path.join(input_dir, boxes_filename)
               write_normalized_boxes(ordered_original_layout_regions_scaled, normal_cv_width, normal_cv_height, boxes_path)
           except Exception as e:
               print(f"Error writing boxes file: {str(e)}")

       adjusted_ordered_original_layout_regions_scaled = adjust_bounding_boxes_final(ordered_original_layout_regions_scaled, image_bbox)

       final_regions = []
       table_images = []
       table_cells_list = []
       table_positions = []

       for region in adjusted_ordered_original_layout_regions_scaled:
           if region[4] == 'Table':
               x1, y1, x2, y2, _, _, position = region
               table_image = normal_image_cv[int(y1):int(y2), int(x1):int(x2)]
               table_images.append(Image.fromarray(cv2.cvtColor(table_image, cv2.COLOR_BGR2RGB)))
               
               cells_in_table = [cell for cell in sparse_line_prediction.bboxes if full_encapsulation(region[:4], cell.bbox)]
               table_cells = [{"bbox": [cell.bbox[0]-x1, cell.bbox[1]-y1, cell.bbox[2]-x1, cell.bbox[3]-y1]} for cell in cells_in_table]
               table_cells_list.append(table_cells)
               table_positions.append((position, (x1, y1, x2, y2)))
           else:
               final_regions.append(region)

       if table_images:
           try:
               table_results = batch_table_recognition(table_images, table_cells_list, table_model, table_processor)

               for table_result, (position, (table_x1, table_y1, _, _)) in zip(table_results, table_positions):
                   for cell in table_result.cells:
                       x1, y1, x2, y2 = cell.bbox
                       x1 += table_x1
                       y1 += table_y1
                       x2 += table_x1
                       y2 += table_y1
                       filename = f"{page_number}_{position}_a_{cell.row_id}_{cell.col_id}.png"
                       final_regions.append([x1, y1, x2, y2, 'a', filename, cell.row_id, cell.col_id])
           except Exception as e:
               print(f"Error in table recognition for page {page_number}: {str(e)}")
               final_regions.extend([region for region in adjusted_ordered_original_layout_regions_scaled if region[4] == 'Table'])

       original_layout_regions_scaled_with_position_list.append(final_regions)

       raw_height, raw_width, _ = raw_image.shape
       x_scale = raw_width / normal_cv_width
       y_scale = raw_height / normal_cv_height

       # Save regionimages in working directory
       regionimages_directory = os.path.join(os.getcwd(), 'regionimages')
       os.makedirs(regionimages_directory, exist_ok=True)

       num_images = len(final_regions)
       num_digits = len(str(num_images))

       for region in final_regions:
           x1, y1, x2, y2, label, *rest = region
           scaled_x1 = int(x1 * x_scale)
           scaled_y1 = int(y1 * y_scale)
           scaled_x2 = int(x2 * x_scale)
           scaled_y2 = int(y2 * y_scale)
           
           if scaled_x2 <= scaled_x1 or scaled_y2 <= scaled_y1:
               print(f"Warning: Skipping invalid region in page {page_number}: {region}")
               continue

           region_image = raw_image[scaled_y1:scaled_y2, scaled_x1:scaled_x2]
           region_image_rgb = cv2.cvtColor(region_image, cv2.COLOR_BGR2RGB)

           pil_image = Image.fromarray(region_image_rgb)
           
           width, height = pil_image.size
           longest_side = max(width, height)
           
           if longest_side > 0:
               rescale_factor = 1000 / longest_side
               new_width = max(1, int(width * rescale_factor))
               new_height = max(1, int(height * rescale_factor))
               
               try:
                   pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
               except Exception as e:
                   print(f"Error resizing image for region in page {page_number}: {str(e)}. Skipping this region.")
                   continue
           else:
               print(f"Warning: Skipping region with invalid dimensions in page {page_number}: {region}")
               continue
           
           if label == 'a':  # Table cell
               file_name = rest[0]  # Use the filename generated in table recognition
           else:
               position = rest[1] if len(rest) > 1 else 0
               padded_index = str(position).zfill(num_digits)
               shortened_label = get_shortened_label(label)
               file_name = f"{page_number}_{padded_index}_{shortened_label}.png"
               
           file_path = os.path.join(regionimages_directory, file_name)
           pil_image.save(file_path)

   return original_layout_regions_scaled_with_position_list