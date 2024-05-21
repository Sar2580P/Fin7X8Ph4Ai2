import fiftyone.zoo 

dataset = fiftyone.zoo.load_zoo_dataset(
              "open-images-v6",
              splits="validation",
              label_types=["segmentations"],
              classes=["Dog", "Horse", "Woman"],
              max_samples=300,
              dataset_dir = 'datasets/seg',
          )
