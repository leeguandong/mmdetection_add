from mmdet.datasets.builder import DATASETS
from mmdet.datasets.voc import VOCDataset


@DATASETS.register_module()
class VOCDatasetAdd(VOCDataset):
    CLASSES = ("0", "1")
