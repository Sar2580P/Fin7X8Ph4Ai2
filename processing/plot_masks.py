import os
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_masks(df, true_masks_dir, sample_count=5, masks_dir='results/output_masks/', save_path='pics/predicted_train_masks.png'):
    # Sample filenames from the DataFrame
    sampled_filenames = df['mask'].sample(n=sample_count, random_state=42).tolist()

    # Get unique model names from the directory
    model_names = os.listdir(masks_dir)

    # Set up the plot
    num_models = len(model_names) + 1  # Add one for the true masks row
    fig, axes = plt.subplots(num_models, sample_count, figsize=(sample_count * 5, num_models * 5))

    # Plot true masks in the first row
    for j, filename in enumerate(sampled_filenames):
        # Construct the full path to the true mask image
        true_mask_path = os.path.join(true_masks_dir, filename)

        # Load the true mask image
        if true_mask_path.endswith('.npy'):
            # Load the true mask image from a .npy file
            true_mask_image = np.load(true_mask_path)
            # Normalize if needed (assuming mask is binary)
            true_mask_image = (true_mask_image * 255).astype(np.uint8) if true_mask_image.max() <= 1 else true_mask_image
            axes[0, j].imshow(true_mask_image, cmap='gray')
        else:
            try:
                true_mask_image = Image.open(true_mask_path)
                axes[0, j].imshow(true_mask_image, cmap='gray')
            except Exception as e:
                print(f"Error loading true mask image: {true_mask_path}, {e}")
                axes[0, j].imshow(np.zeros((256, 256)), cmap='gray')  # Placeholder for missing mask

        axes[0, j].axis('off')  # Hide axes

    # Set the title for the true masks row
    axes[0, 0].set_title('True Masks', fontsize=14, fontweight='bold', loc='left')

    # Plot predicted masks for each model
    for i, model_name in enumerate(model_names):
        # Directory for the current model
        model_dir = os.path.join(masks_dir, model_name)

        # Set the title for the model row
        axes[i + 1, 0].set_title(model_name, fontsize=14, fontweight='bold', loc='left')

        for j, filename in enumerate(sampled_filenames):
            # Construct the full path to the predicted mask image
            mask_path = os.path.join(model_dir, filename)

            # Check the file extension
            if mask_path.endswith('.npy'):
                # Load the mask image from a .npy file
                mask_image = np.load(mask_path)
                # Normalize if needed (assuming mask is binary)
                mask_image = (mask_image * 255).astype(np.uint8) if mask_image.max() <= 1 else mask_image
                axes[i + 1, j].imshow(mask_image, cmap='gray')
            else:
                # Load the mask image if it’s a standard image format
                try:
                    mask_image = Image.open(mask_path)
                    axes[i + 1, j].imshow(mask_image, cmap='gray')
                except Exception as e:
                    print(f"Error loading mask image: {mask_path}, {e}")
                    axes[i + 1, j].imshow(np.zeros((256, 256)), cmap='gray')  # Placeholder for missing mask

            axes[i + 1, j].axis('off')  # Hide axes

    # Adjust layout to make room for titles
    plt.tight_layout()

    # Save the plot to the specified file
    plt.savefig(save_path)


# Example usage:
# df = pd.DataFrame({'filenames': ['mask1.png', 'mask2.png', 'mask3.png', 'mask4.png', 'mask5.png', ...]})
# plot_masks(df)


plot_masks(df = pd.read_csv('data/test_patch_df.csv') , true_masks_dir='data/patched_masks')