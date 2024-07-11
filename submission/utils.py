import numpy as np
import cv2
from typing import List
import os
import json
from tqdm import tqdm

def extract_boundaries(mask_path: str) -> List[List[int]]:
    # Load the binary mask
    binary_image = np.load(mask_path)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract boundary coordinates
    boundaries = []
    for contour in contours:
        boundary = contour.squeeze().flatten().tolist()
        boundaries.append(boundary)
    
    return boundaries



def create_submission_json(test_mask_dir:str = 'results/output_masks'):
    submission = []
    for mask_file in tqdm(os.listdir(test_mask_dir), desc = 'creating submission json'):
        mask_path = os.path.join(test_mask_dir, mask_file)
        boundaries = extract_boundaries(mask_path)
        res = {
            'file_name': mask_file.split('.')[0]+'.tif',
            'annotations': []
        }
        for boundary in boundaries:
            res['annotations'].append({
                'class' : 'field',
                'segmentation': boundary
            })
        submission.append(res)
    
    with open('submission/submission.json', 'w') as f:
        json.dump( {'images' :submission}, f)
    return

if __name__ == '__main__':
    create_submission_json()
    print('Submission JSON created successfully!')