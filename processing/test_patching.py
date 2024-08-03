import unittest
import tempfile
import shutil
import os
import pandas as pd
from patching import Patch

class TestPatch(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

        # Create subdirectories for patched images and masks
        self.patched_images_dir = os.path.join(self.test_dir, 'patched_images')
        self.patched_masks_dir = os.path.join(self.test_dir, 'patched_masks')
        os.makedirs(self.patched_images_dir)
        os.makedirs(self.patched_masks_dir)

        # Create dummy patched image and mask files
        for i in range(6):  # Ensure at least two samples per class
            with open(os.path.join(self.patched_images_dir, f'train_{i}-0.tif'), 'w') as f:
                f.write('dummy image content')
            with open(os.path.join(self.patched_masks_dir, f'train_{i}-0.npy'), 'w') as f:
                f.write('dummy mask content')
        
        for i in range(2):
            with open(os.path.join(self.patched_images_dir, f'test_{i}-0.tif'), 'w') as f:
                f.write('dummy image content')

    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_create_patch_df(self):
        patch = Patch(
            source_image_dir=self.patched_images_dir,
            source_mask_dir=self.patched_masks_dir,
            patch_size=256,
            save_patch_img_dir=self.patched_images_dir,
            save_patch_mask_dir=self.patched_masks_dir
        )

        # Call the method
        patch.create_patch_df(dir=self.test_dir)

        # Check if the CSV files are created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'predict_patch_df.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'train_patch_df.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'val_patch_df.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'test_patch_df.csv')))

        # Verify the contents of the CSV files
        train_df = pd.read_csv(os.path.join(self.test_dir, 'train_patch_df.csv'))
        val_df = pd.read_csv(os.path.join(self.test_dir, 'val_patch_df.csv'))
        test_df = pd.read_csv(os.path.join(self.test_dir, 'test_patch_df.csv'))
        predict_df = pd.read_csv(os.path.join(self.test_dir, 'predict_patch_df.csv'))

        self.assertEqual(len(train_df), 4)
        self.assertEqual(len(val_df), 1)
        self.assertEqual(len(test_df), 1)
        self.assertEqual(len(predict_df), 2)

if __name__ == '__main__':
    unittest.main()