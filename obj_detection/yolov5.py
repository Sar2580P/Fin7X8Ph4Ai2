# default_config : ultralytics/cfg/default.yaml

import os, sys
sys.path.insert(0, os.getcwd())
print(sys.path)
from utils import Utils
import yaml
from ultralytics.models.yolo.detect import DetectionTrainer
import torch.nn as nn

class FC(nn.Module):
    def __init__(self, drop ,in_size, out_size):
      super(FC ,self).__init__()
      self.drop , self.in_size , self.out_size = drop , in_size , out_size
      self.model = self.get_model()
    def get_model(self):
      return nn.Sequential(
          nn.Linear(self.in_size, self.out_size) ,
          nn.PReLU(),
          nn.Dropout(self.drop),
      )
    def forward(self, x):
      return self.model(x)

def f(trainer):
    # print('\n'*10 ,type(trainer))
    # for attr, value in vars(trainer).items():
    #     print(f"{attr}: {value}")
    print("Jai shree ram" , '\n'*10)

def g(trainer):
    print('Radhe Radhe')

available_models = 1

override_config = yaml.safe_load('/home/bala/Desktop/sri_krishna/computer_vision/obj_detection/override_config.yaml')
print(type(override_config) , override_config)
trainer = DetectionTrainer(overrides=override_config)
# trainer.model =  FC(0.5,32,128)          to pass custom architecture
# trainer.add_callback("on_pretrain_routine_start" , f)
# trainer.add_callback("on_train_start" ,  g)
# trainer.setup_model()
trainer.train()
# print(trainer.model)


config = {
    'BATCH_SIZE': 32,
    'C': 3,
    'H': 640,
    'W': 640,
    'model_name': 'yolo_v5s',
    'dir': 'obj_detection'

}
print(type(trainer.model))

Utils.plot_model(config , trainer.model)
