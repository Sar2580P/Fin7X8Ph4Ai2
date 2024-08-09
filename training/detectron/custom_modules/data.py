from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper

import numpy as np
import torch
import copy
from PIL import Image
import tifffile as tiff
from typing import List, Union, Optional
from processing.utils import logger

class TIFF_Mapper(DatasetMapper):

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        return super().from_config(cfg, is_train)

    def read_image(self, file_name: str, format: str) -> np.ndarray:
        """
        Args:
            file_name (str): Path to the image file.
            format (str): Desired format ('RGB', 'YUV', etc.)

        Returns:
            np.ndarray: The image as a NumPy array.
        """
        # Read the TIFF image
        image = tiff.imread(file_name)
        # logger.info(f"Image shape: {image.shape}")

        # If the format is 'RGB', convert the image
        if format == 'RGB':
            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)  # Convert grayscale to RGB
            elif image.shape[-1] > 3:
                image = image[..., :3]  # Keep only the first three channels
        elif format == 'YUV':
            image = self.convert_RGB_to_YUV(image)

        return image

    def convert_RGB_to_YUV(self, image: np.ndarray) -> np.ndarray:
        """
        Convert an RGB NumPy array to YUV (BT.601).

        Args:
            image (np.ndarray): The input RGB image as a NumPy array.

        Returns:
            np.ndarray: The image as a YUV NumPy array.
        """
        conversion_matrix = np.array([
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001]
        ])

        image = image / 255.0
        image_yuv = np.dot(image, conversion_matrix.T)
        image_yuv = (image_yuv * 255).clip(0, 255).astype(np.uint8)

        return image_yuv

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # Use the custom read_image function
        image = self.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = self.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)) , dtype=torch.float32)
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)
        return dataset_dict
