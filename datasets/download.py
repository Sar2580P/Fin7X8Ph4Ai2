import fiftyone.zoo

dataset = fiftyone.zoo.load_zoo_dataset(
              "open-images-v6",
              splits="validation",
              label_types=["segmentations"],
              classes=[
                    "Dog", "Horse", "Woman", "Accordion", "Adhesive tape", "Aircraft", "Alarm clock",
                    "Alpaca", "Ambulance", "Ant", "Antelope", "Apple"],
              max_samples=3000,
              dataset_dir = 'datasets/segmentation/open-images-v6',
          )
