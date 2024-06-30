from processing.img_preproc import *
import os , json

def apply_preprocessing():
    source_dir = 'data/original_images'
    output_dir = 'data/3channel_images'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for file_name in os.listdir(source_dir):
        if file_name.lower().endswith('.tif'):
            img_path = os.path.join(source_dir , file_name)         
            # Process the image
            try:
                save_path = os.path.join(output_dir, file_name)
                take_3_channels(channel_indices=[4,5,6], img_path = img_path, save_path= save_path)
                # print(f"Processed {img_path} and saved to {output_path}")
            except ValueError as e:
                print(f"Error processing {img_path}: {e}")
                
 
        

        
        

if __name__ == '__main__':
    # apply_preprocessing()
    # create_masks()
    pass