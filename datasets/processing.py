import json
from typing import List
import pickle
import pandas as pd
    
def reduce_classes():
    with open('datasets/challenge-2019-label500-hierarchy.json' , 'r') as f:
        data = json.load(f)
    reverse_dict = {}
    li :List[dict] = data['Subcategory']
    total_classes =  0 
    for x in li:
        total_classes += 1
        base_key = x['LabelName']
        if 'Subcategory' in x.keys():
            for y in x['Subcategory']:
                reverse_dict[y['LabelName']] = base_key
                
    with open('datasets/reverse_dict.pkl', 'wb') as f:
        pickle.dump(reverse_dict, f)
        
    return reverse_dict

print(reduce_classes())