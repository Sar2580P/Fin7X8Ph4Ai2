import logging
import weakref
import torch
import os
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer, DefaultPredictor, create_ddp_model, AMPTrainer, SimpleTrainer
from detectron2.modeling import build_model
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data import MetadataCatalog
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

def build_evaluator(cfg:CfgNode, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class ExtendedPredictor(DefaultPredictor):
    """
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
        cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model:torch.nn.Module = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format


    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():
            # Handle different input formats
            if self.input_format == "RGB":
                # Convert BGR to RGB if needed
                original_image = original_image[:, :, ::-1]

            # Get the height and width of the image
            height, width = original_image.shape[:2]

            # Apply the augmentation and resizing
            image = self.aug.get_transform(original_image).apply_image(original_image)

            # Convert the image to tensor and maintain the channels
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image = image.to(self.cfg.MODEL.DEVICE)

            inputs = {"image": image, "height": height, "width": width}

            # Make the prediction
            predictions = self.model([inputs])[0]
            return predictions


class CustomTrainer(DefaultTrainer):

    def __init__(self, cfg:CfgNode, train_mapper:DatasetMapper=None,
                 test_mapper:DatasetMapper=None):
        """
        Args:
            cfg (CfgNode): configuration file
            train_mapper (callable, optional): a function/method to map training dataset items.
            test_mapper (callable, optional): a function/method to map test dataset items.
        """
        super().__init__(cfg)
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg, train_mapper)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.hooks = [] #self.build_hooks()           # Initialize hooks list

        # print(self.hooks , ')'*50)

    def add_hook(self, hook):
        """
        Args:
            hook (HookBase): a hook to be added to the list.
        """
        self.hooks.append(hook)

    def register_all_hooks(self):
        """
        Register all hooks. This method should be called after adding custom hooks.
        """
        self.register_hooks(self.hooks)

    @classmethod
    def build_train_loader(cls, cfg, mapper:DatasetMapper=None):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader` with an optional mapper argument.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name, mapper:DatasetMapper=None):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader` with an optional mapper argument.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

