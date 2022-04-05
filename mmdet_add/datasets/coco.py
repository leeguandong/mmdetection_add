from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module()
class CocoDatasetAdd(CocoDataset):
    """
    文本检测的任务只需要一类 text，目标检测任务有多少类即写多少类，softamx-based方式，不考虑背景类
    """
    CLASSES = ("0", "1")
