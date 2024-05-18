from ultralytics.models.yolo.segment import SegmentationTrainer
import yaml


with open('segment/override.yaml', 'r') as file:
    override_config = yaml.safe_load(file)
    
trainer = SegmentationTrainer(overrides=override_config)
trainer.train()