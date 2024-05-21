import shutil
import os
import random

DATA_FOLDER = '/home/bala/Desktop/sri_krishna/computer_vision/data'
LABEL_SET = 'full_labels'

IMAGES_PATH = {'train': fr'{DATA_FOLDER}/images/train', 'val': fr'{DATA_FOLDER}/images/val'}

LABELS_PATH = {'train': fr'{DATA_FOLDER}/object_detection/labels/train/{LABEL_SET}',
            'val': fr'{DATA_FOLDER}/object_detection/labels/val/{LABEL_SET}',
            'class_numbering_scheme': fr'{DATA_FOLDER}/object_detection/labels/class_numbering_scheme/{LABEL_SET}.json'}




OVERWRITE_FILES = True


if __name__ == '__main__':

    for mode in ['val', 'train']:
        print(f'Copying {mode} images')

        source_folders = {'images': IMAGES_PATH[mode], 'labels': LABELS_PATH[mode]}

        root_destination_folder = '/home/bala/Desktop/sri_krishna/computer_vision/datasets/open_images'
        destination_folders = {'images': f'{root_destination_folder}/images/{mode}', 'labels': f'{root_destination_folder}/labels/{mode}'}

        for folder in destination_folders.values():

            # Check if the folder exists
            if os.path.exists(folder) and OVERWRITE_FILES:
                # If it does, remove it along with all its contents
                shutil.rmtree(folder)
            else:
                # Create the folder
                os.makedirs(folder, exist_ok = True)

        image_names = os.listdir(source_folders['images'])
        image_names = [file[:-4] for file in image_names]

        no_files = len(image_names)//2

        image_names = random.sample(image_names, no_files)

        index = 0
        for image_name in image_names:
            shutil.copy(f'{source_folders["images"]}/{image_name}.jpg', destination_folders["images"])
            shutil.copy(f'{source_folders["labels"]}/{image_name}.txt', destination_folders["labels"])
            index += 1
            if index%1000 == 0 or index == no_files-1:
                print(f'    {index} images copied')

        print(f'    {mode} images copied', '\n\n')

    print('\n\n\n\n')

