from .coco import CocoDataset
from .builder import DATASETS

import os

@DATASETS.register_module
class Dataset(CocoDataset):

    CLASSES = (os.environ['MMDET_CLASSES'])