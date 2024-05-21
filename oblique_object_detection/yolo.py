from ultralytics.models.yolo.obb import OBBTrainer
import yaml
import os, sys
sys.path.insert(0, os.getcwd())
from utils import Utils

with open('oblique_object_detection/override.yaml', 'r') as file:
    override_config = yaml.safe_load(file)
  
trainer = OBBTrainer(overrides=override_config)
trainer.setup_model()


config = {
    'BATCH_SIZE': 32,
    'C': 3,
    'H': 640,
    'W': 640,
    'model_name': 'yolo_v8n-obb',
    'dir': 'oblique_object_detection'

}
Utils.plot_model(config , trainer.model)


trainer.train()
