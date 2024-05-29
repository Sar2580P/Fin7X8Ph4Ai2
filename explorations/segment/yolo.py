from ultralytics.models.yolo.segment import SegmentationTrainer
import yaml
import os, sys
sys.path.insert(0, os.getcwd())
from utils import Utils

with open('segment/override.yaml', 'r') as file:
    override_config = yaml.safe_load(file)
    
trainer = SegmentationTrainer(overrides=override_config)
trainer.setup_model()

config = {
    'BATCH_SIZE': 32,
    'C': 3,
    'H': 640,
    'W': 640,
    'model_name': 'yolo_v8n-seg',
    'dir': 'segment'

}
Utils.plot_model(config , trainer.model)


trainer.train()