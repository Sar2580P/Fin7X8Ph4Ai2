import pandas as pd

import os
import json

import tree_algo as ta


#---------------------------------------------#---------------------------------------------#---------------------------------------------
#---------------------------------------------#---------------------------------------------#---------------------------------------------
#---------------------------------------------#---------------------------------------------#---------------------------------------------

DATA_FOLDER = '/home/bala/Desktop/sri_krishna/computer_vision/data/'

IMAGES_PATH = {'train': f'{DATA_FOLDER}/images/train', 'val': f'{DATA_FOLDER}/images/val'}

BOXES_PATH = {'train': f'{DATA_FOLDER}/object_detection/boxes/challenge-2019-train-detection-bbox.csv',
            'val': f'{DATA_FOLDER}/object_detection/boxes/challenge-2019-validation-detection-bbox.csv'}

LABELS_PATH = {'train': f'{DATA_FOLDER}/object_detection/labels/challenge-2019-train-detection-human-imagelabels.csv',
            'val': f'{DATA_FOLDER}/object_detection/labels/challenge-2019-validation-detection-human-imagelabels.csv'}

METADATA = {'classes': f'{DATA_FOLDER}/object_detection/metadata/challenge-2019-classes-description-500.csv',
            'class_hierarchy': f'{DATA_FOLDER}/object_detection/metadata/challenge-2019-label500-hierarchy.json'}

classes = pd.read_csv(METADATA['classes'])

data = None
with open(METADATA['class_hierarchy'], 'r') as file:
    data = json.load(file)

LABELCODE_TO_LABELNAME = {row["LabelCode"]: row["LabelName"] for _, row in classes.iterrows()}

TREE = ta.json_to_tree(data)



#---------------------------------------------#---------------------------------------------#---------------------------------------------
#---------------------------------------------#---------------------------------------------#---------------------------------------------
#---------------------------------------------#---------------------------------------------#---------------------------------------------
def load_data(mode):

    list_of_images_paths = os.listdir(IMAGES_PATH[mode])
    list_of_images_paths = [file_name[:-4] for file_name in list_of_images_paths]

    labeling_data = pd.read_csv(BOXES_PATH[mode])

    labeling_data = labeling_data[labeling_data['ImageID'].isin(list_of_images_paths)]
    labeling_data['Width'] = labeling_data['XMax'] - labeling_data['XMin']
    labeling_data['Height'] = labeling_data['YMax'] - labeling_data['YMin']

    sample_label_codes = list(labeling_data['LabelName'].values)

    labeling_data = labeling_data[['ImageID', 'LabelName', 'XMin', 'YMin', 'Width', 'Height']]

    return list_of_images_paths, labeling_data, sample_label_codes



def preprare_data(data_folder, mode, tree, depth, labelcode_to_labelname, list_of_images_paths, labeling_data, sample_label_codes):
    destination_root_folder = fr'{data_folder}/object_detection/labels/'

    reverse_mapping = ta.reverse_mapper(tree, depth)
    labelcode_numbering = ta.class_no_mapper(reverse_mapping)
    class_mapping = ta.class_mapper(reverse_mapping=reverse_mapping, class_no_mapping=labelcode_numbering)

    labelname_numbering = {}
    for key, value in labelcode_numbering.items():
        labelname_numbering[value] = labelcode_to_labelname[key]

    sample_class_no = []
    for code in sample_label_codes:
        sample_class_no.append(class_mapping[code])

    labeling_data['LabelNumber'] = sample_class_no
    labeling_data = labeling_data[['ImageID', 'LabelNumber', 'XMin', 'YMin', 'Width', 'Height']]


    labelname_numbering_path = fr'{destination_root_folder}/class_numbering_scheme/depth_{depth}/class_numbering.json'
    if not os.path.exists(labelname_numbering_path):
        with open(labelname_numbering_path, 'w') as file:
            json.dump(labelname_numbering, file)

    index = 0
    total_images = len(list_of_images_paths)
    for path in list_of_images_paths:
        file = labeling_data[labeling_data['ImageID'].isin([path])]
        txt_file = file[['LabelNumber', 'XMin', 'YMin', 'Width', 'Height']]
        txt_file.to_csv(fr'{destination_root_folder}/{mode}/depth_{depth}/{path}.txt', sep=' ', index = False, header=False)
        index += 1

        if index % 10000 == 0 or index == total_images-1:
            print(f'            {index} images done')



#---------------------------------------------#---------------------------------------------#---------------------------------------------
#---------------------------------------------#---------------------------------------------#---------------------------------------------
#---------------------------------------------#---------------------------------------------#---------------------------------------------

# val data for depth 1 already done
for mode in ['train']:
    print(f'Processing {mode} data')
    list_of_images_paths, labeling_data, sample_label_codes = load_data(mode)
    print('    Data loaded')

    for depth in range(1, 2): #Other than depth 1,you can't do it for other depths, either take only 177, or all 500
        print(f'        Processing depth ---- {depth}')
        preprare_data(DATA_FOLDER, mode, TREE, depth, LABELCODE_TO_LABELNAME, list_of_images_paths, labeling_data, sample_label_codes)
        print(f'        Processing of depth ---- {depth} completed')

    print(f'    Processing of {mode} data completed', '\n\n')



