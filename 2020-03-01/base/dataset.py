from .coco import CocoDataset
from .registry import DATASETS

import os

@DATASETS.register_module
class Dataset(CocoDataset):

    CLASSES = (os.environ['MMDET_CLASSES'])