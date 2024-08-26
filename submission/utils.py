import numpy as np
import cv2
from typing import List
import os
import json
from tqdm import tqdm
import re


import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.validation import explain_validity
from typing import List

from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import explain_validity
from shapely.errors import TopologicalError
from shapely.ops import unary_union

def validate_polygon(boundary: List[int]) -> List[int]:
    try:
        # Create a Shapely Polygon from the boundary coordinates
        polygon = Polygon([(boundary[i], boundary[i+1]) for i in range(0, len(boundary), 2)])

        # Check if the polygon is valid
        if not polygon.is_valid:
            print(f"Invalid polygon detected: {explain_validity(polygon)}")
            # Attempt to fix the polygon using buffer(0)
            polygon = polygon.buffer(0)

            # Further simplification if needed
            if not polygon.is_valid:
                polygon = polygon.buffer(0.01).buffer(-0.01)
            # Further simplification if needed
            if not polygon.is_valid:
                polygon = polygon.simplify(tolerance=0.05, preserve_topology=True)

            # Try unary_union to clean the geometry
            if not polygon.is_valid:
                polygon = unary_union(polygon)

        # Handle the case where the result is a MultiPolygon
        if isinstance(polygon, MultiPolygon):
            # Choose the largest polygon by area (or implement other logic as needed)
            polygon = max(polygon.geoms, key=lambda p: p.area)

        # Convert the fixed polygon back to a list of coordinates
        if polygon.is_valid:
            boundary = list(polygon.exterior.coords[:-1])  # Remove the closing coordinate to match input format
            boundary = [coord for point in boundary for coord in point]  # Flatten the list
        else:
            boundary = []

    except TopologicalError as e:
        print(f"TopologyException: {e}. Could not fix the polygon.")
        boundary = []

    return boundary


def extract_boundaries(mask_path: str, threshold_value: float = 0.7) -> List[List[int]]:
    # Load the binary mask
    binary_image = np.load(mask_path).astype(np.float32)

    # Apply threshold to convert to a binary image
    _, binary_image = cv2.threshold(binary_image, threshold_value, 1, cv2.THRESH_BINARY)
    binary_image = binary_image.astype(np.uint8)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract boundary coordinates
    boundaries = []
    for contour in contours:
        boundary = contour.squeeze().flatten().tolist()
        if len(boundary) >= 8:
            boundary = validate_polygon(boundary)
            if boundary:  # Only add valid boundaries
                boundaries.append(boundary)

    return boundaries




def create_submission_json(test_mask_dir:str ):
    submission = []
    li = os.listdir(test_mask_dir)
    li = sorted(li, key=lambda x: int(re.findall(r'\d+', x)[0]))

    for mask_file in tqdm(li, desc = 'creating submission json'):
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



from submission.pq_score_calculate import get_score_for_all_images
from processing.utils import logger
import pickle
if __name__ == '__main__':
    create_submission_json(test_mask_dir='results/inference/stitched/UnetPlusPlus__E-resnet50__W-imagenet__C-20_ensemble')
    print('Submission JSON created successfully!')

    # ground_truth = json.load(open('data/train_annotations.json'))
    # prediction = json.load(open('submission/train_submission.json'))
    # scores = get_score_for_all_images(ground_truth['images'], prediction['images'])

    # pickle.dump(scores, open('submission/scores.pkl', 'wb'))
