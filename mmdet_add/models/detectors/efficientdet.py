'''
@Time    : 2022/4/3 21:23
@Author  : leeguandon@gmail.com
'''
from mmdet.models import SingleStageDetector, DETECTORS


@DETECTORS.register_module()
class EfficientDet(SingleStageDetector):
    param_dict = {
        'b0': (64, 3, 3, [40, 112, 320]),
        'b1': (88, 4, 3, [40, 112, 320]),
        'b2': (112, 5, 3, [48, 120, 352]),
        'b3': (160, 6, 4, [48, 136, 384]),
        'b4': (224, 7, 4, [56, 160, 448]),
        'b5': (228, 7, 4, [64, 176, 512]),
        'b6': (384, 8, 5, [72, 200, 576]),
        'b7': (384, 8, 5, [80, 224, 640]),
    }

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 arch,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        bifpn_width, bifpn_depth, head_depth, bifpn_channels = self.param_dict[arch]

        backbone.update({'arch': arch})
        neck.update({'in_channels': bifpn_channels,
                     'out_channels': bifpn_width,
                     'stack': bifpn_depth})
        bbox_head.update({'stacked_convs': head_depth,
                          'in_channels': bifpn_width,
                          'feat_channels': bifpn_width})

        super(EfficientDet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                           test_cfg, pretrained, init_cfg)
