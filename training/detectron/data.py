from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import json 
import tqdm as tqdm
from typing import List, Dict
from processing.utils import logger

def create_lightdata_for_instance_seg(data_dir:str , json_file:str = 'data/updated_train_annotation.json')->List[Dict]:
    with open(json_file) as f:
        data = json.load(f)
        
    dataset_dicts = []
    for id , img in tqdm(enumerate(data['images']) , desc = 'Detectron2 light_data for instance segmentation'):
        record = {}
        record['file_name'] = f"{data_dir}/{img['file_name']}"
        record['image_id'] = id
        record['height'] = img['height']
        record['width'] = img['width']
        record['annotations'] = []
        for annotation in img['annotations']:
            annotation_dict = {}
            annotation_dict['bbox'] = annotation['bounding_box']
            annotation_dict['bbox_mode'] = BoxMode.XYWH_ABS
            annotation_dict['category_id'] = annotation['category_id']
            annotation_dict['segmentation'] = annotation['segmentation']
            record['annotations'].append(annotation_dict)
        dataset_dicts.append(record)
        
        
def register_lightdata(data_config:Dict):
    
    task = data_config['task']
    for d , data_dir , annot_path in zip(["train", "val"] ,[data_config['train_data_dir'] , data_config['val_data_dir']], 
                                         [data_config['train_annot_path'], data_config['val_annot_path']]):
        
        DatasetCatalog.register(f"{task}_{d}", lambda d=d: create_lightdata_for_instance_seg(data_dir , annot_path))
        MetadataCatalog.get(task).thing_classes = data_config['categories']
        MetadataCatalog.get(task).thing_colors = data_config['category_colors']  # RGB colors for each class
        MetadataCatalog.get(task).evaluator_type = data_config['evaluator_type']
    
    logger.info(MetadataCatalog)
    
    
    


    
# # Mapper
# def mapper(dataset_dict):
#     dataset_dict = copy.deepcopy(dataset_dict)  # as it will be modified by code below

#     image = utils.read_image(dataset_dict["file_name"], format="BGR")     # TODO

#     augs = T.AugmentationList([
#                     T.RandomBrightness(0.9, 1.1),
#                     T.RandomFlip(prob=0.5),
#                     T.RandomCrop("absolute", (640, 640))
#                     ]) 
#     auginput = T.AugInput(image)
#     transform = augs(auginput)
#     image = torch.from_numpy(auginput.image.transpose(2, 0, 1))
#     annos = [
#         utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
#         for annotation in dataset_dict.pop("annotations")
#     ]
#     return {
#        # create the format that the model expects
#        "image": image,
#        "instances": utils.annotations_to_instances(annos, image.shape[1:])
#     }
    
    
# Data loader

# Augmentation

# Sampler

