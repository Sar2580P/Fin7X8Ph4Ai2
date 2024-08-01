from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
from processing.utils import read_yaml_file
from training.detectron.data import register_lightdata
from training.detectron.custom_modules.trainer import CustomTrainer
from training.detectron.custom_modules.data import TIFF_Mapper

# Load the configuration file
static_config = read_yaml_file('training/modelling/detectron/config.yaml')

def initialize_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(static_config['model_config_file']))
    cfg.DATASETS.TRAIN = (f"{static_config['task']}_train",)
    cfg.DATASETS.TEST = ()
    cfg.OUTPUT_DIR = static_config['OUTPUT_DIR']
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(static_config['model_config_file'])  
    cfg.SOLVER.IMS_PER_BATCH = static_config['IMS_PER_BATCH'] 
    cfg.SOLVER.BASE_LR = static_config['BASE_LR']  
    cfg.SOLVER.MAX_ITER = static_config['MAX_ITER']    
    cfg.SOLVER.STEPS = static_config['STEPS']       
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = static_config['BATCH_SIZE_PER_IMAGE']  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = static_config['NUM_CLASSES'] 
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg


register_lightdata(static_config['data'])

cfg = initialize_config()


train_mapper = TIFF_Mapper.from_config(cfg, is_train=True)
val_mapper = TIFF_Mapper.from_config(cfg, is_train=False)
trainer = CustomTrainer(cfg , train_mapper=train_mapper, test_mapper=val_mapper) 
trainer.register_all_hooks()

trainer.resume_or_load(resume=False)
trainer.train()
# trainer.test()

