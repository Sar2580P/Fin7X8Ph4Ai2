from shapely.geometry import Polygon
import numpy as np
from tqdm import tqdm
from typing import List
from processing.utils import logger


def getIOU(polygon1: Polygon, polygon2: Polygon) -> float:
    intersection = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    if union == 0:
        return 0
    return intersection / union


# https://arxiv.org/abs/1801.00868
def compute_pq(gt_polygons: list, pred_polygons: list, iou_threshold=0.5):
    matched_instances = {}
    gt_matched = np.zeros(len(gt_polygons))
    pred_matched = np.zeros(len(pred_polygons))

    gt_matched = np.zeros(len(gt_polygons))
    pred_matched = np.zeros(len(pred_polygons))
    for gt_idx, gt_polygon in tqdm(enumerate(gt_polygons)):
        best_iou = iou_threshold
        best_pred_idx = None
        for pred_idx, pred_polygon in enumerate(pred_polygons):
            # if gt_matched[gt_idx] == 1 or pred_matched[pred_idx] == 1:
            #     continue
            
            iou = getIOU(gt_polygon, pred_polygon)
            if iou == 0:
                continue
            
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx
        if best_pred_idx is not None:
            matched_instances[(gt_idx, best_pred_idx)] = best_iou
            gt_matched[gt_idx] = 1
            pred_matched[best_pred_idx] = 1

    
    sq_sum = sum(matched_instances.values())
    num_matches = len(matched_instances)
    sq = sq_sum / num_matches if num_matches else 0
    rq = num_matches / ((len(gt_polygons) + len(pred_polygons))/2.0) if (gt_polygons or pred_polygons) else 0
    pq = sq * rq

    return pq, sq, rq


def get_polygons_from_annot(annot:List):
    polygons = []
    for instance in annot:
        coords = instance['segmentation']
        polygon = Polygon([(coords[i], coords[i+1]) for i in range(0, len(coords), 2)])
        polygons.append(polygon)
    return polygons


def get_score_for_all_images(gt_annots :List, pred_annots:List)->dict:
    pq_scores = {}

    for gt_annot , pred_annot in zip(gt_annots, pred_annots):
        gt_polygons = get_polygons_from_annot(gt_annot['annotations'])
        pred_polygons = get_polygons_from_annot(pred_annot['annotations'])
        assert gt_annot['file_name'] == pred_annot['file_name'] , 'file names are not same'
        try:
            pq, sq, rq = compute_pq(gt_polygons, pred_polygons)
            pq_scores[gt_annot['file_name']] = {'pq': pq, 'sq': sq, 'rq': rq}
            logger.info(f"{gt_annot['file_name']} : pq: {pq}, sq: {sq}, rq: {rq}")
        except Exception as e:
            logger.error(f'Error in calculating pq score for {gt_annot["file_name"]}')
            logger.debug(e)
            
    return pq_scores
        
    
    
    
def test_compute_pq():
    polygon1 = Polygon([(1, 2), (2, 4), (3, 1)])
    polygon2 = Polygon([(0, 0), (1, 3), (2, 2), (3, 0)])
    polygon3 = Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3)])
    polygon4 = Polygon([(2, 2), (3, 4), (4, 4), (5, 2), (3, 1)])
    polygon5 = Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4)])
    polygon6 = Polygon([(1, 1), (2, 3), (3, 3), (2, 1)])
    polygon7 = Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2)])
    
    true_polygons = [polygon1, polygon3, polygon5, polygon7]
    pred_polygons = [polygon1, polygon2, polygon3, polygon7]
    
    pq, sq, rq = compute_pq(true_polygons, pred_polygons)
    assert round(pq,1) == 0.8
    assert sq == 1
    assert round(rq,1) == 0.8
    
def test_get_iou():
    polygon1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    polygon2 = Polygon([(0, 0), (0, 1), (2, 1), (2, 0)])
    
    assert getIOU(polygon1, polygon2) == 0.5
    
def test_same_score_with_dif_order():
    true_polygons = [
        Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]),
        Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]),
        Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]),
    ]
    pred_polygons = [
        Polygon([(7, -3), (8, -2), (9, -3), (10, -4), (7, -5), (7, -3)]),
        Polygon([(9, 8), (10, 10), (12, 11), (13, 9), (12, 8), (9, 8)]),
        Polygon([(4, 4), (5, 6), (7, 6), (8, 4), (6, 3), (4, 4)]),
    ]
    true_order_1 = [1, 0, 2]
    pred_order_1 = [0, 1, 2]
    true_order_2 = [0, 1, 2]
    pred_order_2 = [1, 0, 2]
    pq1, sq1, rq1 = compute_pq([true_polygons[i] for i in true_order_1], [pred_polygons[i] for i in pred_order_1])
    pq2, sq2, rq2 = compute_pq([true_polygons[i] for i in true_order_2], [pred_polygons[i] for i in pred_order_2])
    
    assert pq1 == pq2
    assert sq1 == sq2
    assert rq1 == rq2
    
