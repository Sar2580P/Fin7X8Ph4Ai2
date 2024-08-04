from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
from processing.utils import read_yaml_file
from training.detectron.data import register_lightdata
from training.detectron.custom_modules.trainer import CustomTrainer , ExtendedPredictor
from training.detectron.custom_modules.data import TIFF_Mapper
from detectron2.data import  MetadataCatalog

# Load the configuration file
static_config = read_yaml_file('training/detectron/config.yaml')
data_model_config = static_config['data']

def initialize_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(data_model_config['model_config_file']))
    cfg.DATASETS.TRAIN = (f"{data_model_config['task']}_train",)
    cfg.DATASETS.TEST = (f"{data_model_config['task']}_val",)
    cfg.OUTPUT_DIR = data_model_config['OUTPUT_DIR']
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(data_model_config['model_config_file'])
    cfg.DATALOADER.NUM_WORKERS = static_config['NUM_WORKERS']
    cfg.SOLVER.IMS_PER_BATCH = static_config['IMS_PER_BATCH']
    cfg.SOLVER.BASE_LR = static_config['BASE_LR']
    cfg.SOLVER.MAX_ITER = static_config['MAX_ITER']
    cfg.SOLVER.STEPS = static_config['STEPS']
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = static_config['BATCH_SIZE_PER_IMAGE']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = static_config['NUM_CLASSES']

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


register_lightdata(data_model_config)

cfg = initialize_config()


DO_TRAINING = True

if DO_TRAINING:
    train_mapper = TIFF_Mapper(**TIFF_Mapper.from_config(cfg, is_train=True))
    val_mapper = TIFF_Mapper(**TIFF_Mapper.from_config(cfg, is_train=False))
    trainer = CustomTrainer(cfg , train_mapper=train_mapper, test_mapper=val_mapper)
    trainer.register_all_hooks()
    trainer.resume_or_load(resume=False)
    trainer.train()
    trainer.test()

else:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = ExtendedPredictor(cfg)

    from detectron2.utils.visualizer import ColorMode
    import rasterio , numpy as np
    import matplotlib.pyplot as plt
    from detectron2.utils.visualizer import Visualizer

    if not os.path.exists(f'{cfg.OUTPUT_DIR}/predictions'):
        print('Creating predictions directory')
        os.makedirs(f'{cfg.OUTPUT_DIR}/predictions')

    print(os.listdir(f'{cfg.OUTPUT_DIR}/predictions'))
    BASE_DIR = static_config['data']['val_data_dir']
    test_files = [f for f in os.listdir(BASE_DIR) if f.startswith('test')]
    for file_name in test_files:
        file_path = os.path.join(BASE_DIR, file_name)
        image = rasterio.open(file_path).read()
        image = np.transpose(image, (1, 2, 0))
        outputs = predictor(image)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(image,
                    metadata= MetadataCatalog.get(data_model_config['task']),
                    scale=0.5,
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        np.save(f'{cfg.OUTPUT_DIR}/predictions/{file_name[:-4]}.npy', out.get_image())


