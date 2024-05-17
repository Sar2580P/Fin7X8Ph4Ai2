# default_config : ultralytics/cfg/default.yaml
import yaml






from ultralytics.models.yolo.detect import DetectionTrainer


override_config = yaml.safe_load('obj_detection/override_config.yaml')
trainer = DetectionTrainer(overrides=override_config)
trainer.train()
