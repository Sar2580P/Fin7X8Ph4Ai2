import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Function to create and save subplots
def create_and_save_plots(files, directory, output_filename):
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for ax, file in zip(axes, files):
        img = np.load(os.path.join(directory, file))
        ax.imshow(img, cmap='gray')
        ax.set_title(file)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()


def plot_masks(train_dir:str , predicted_dir:str, ct:int = 5):
    # Get the list of files in each directory
    train_files = [f for f in os.listdir(predicted_dir) if f.startswith('train_') and f.endswith('.npy')]
    test_files = [f for f in os.listdir(predicted_dir) if f.startswith('test_') and f.endswith('.npy')]
    # train_true_mask_files = [f for f in os.listdir(train_dir) if f.startswith('train_mask_') and f.endswith('.npy')]

    # Randomly select 5 files from each
    selected_train_files = random.sample(train_files, ct)
    selected_test_files = random.sample(test_files, ct)
    selected_mask_files = selected_train_files


    # Create and save the plots
    create_and_save_plots(selected_train_files, predicted_dir, 'pics/predicted_train_masks.png')
    create_and_save_plots(selected_test_files, predicted_dir, 'pics/predicted_test_masks.png')
    create_and_save_plots(selected_mask_files, train_dir, 'pics/true_train_masks.png')

    print('Plots saved successfully.')
