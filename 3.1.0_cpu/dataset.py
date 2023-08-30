from .coco import CocoDataset
from mmdet.registry import DATASETS

import os

ENV_VAR = 'MMDET_CLASSES'


def determine_classes():
    """
    Determines the class labels: reads the MMDET_CLASSES environment variable
    and, if it points to a file, reads the comma-separated labels from there
    (format: "a,b,c"), otherwise interprets them as string (format: "'a','b','c'" or "a,b,c")

    :return: the list of string labels
    :rtype: list
    """

    result = None
    if ENV_VAR in os.environ:
        classes = os.environ[ENV_VAR]
        if os.path.exists(classes) and os.path.isfile(classes):
            with open(classes, "r") as labels_file:
                class_names = labels_file.read().strip()
                result = class_names.split(",")
        else:
            result = [x.strip() for x in os.environ[ENV_VAR].split(",")]
            for i in range(len(result)):
                if result[i].startswith("'") and result[i].endswith("'"):
                    result[i] = result[i][1:len(result[i])-1]
                elif result[i].startswith('"') and result[i].endswith('"'):
                    result[i] = result[i][1:len(result[i]) - 1]
    else:
        raise Exception("Environment variable %s not set!" % ENV_VAR)
    print("Labels determined from %s:" % ENV_VAR, result)
    return result


@DATASETS.register_module
class Dataset(CocoDataset):

    CLASSES = determine_classes()
