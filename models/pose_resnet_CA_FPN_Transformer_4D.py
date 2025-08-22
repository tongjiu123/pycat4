import torch
import torch.nn as nn
from .pose_resnet import *
from .pose_resnet_CA_FPN_Transformer import *
from mmcv.cnn.bricks.conv_module import ConvModule

class PoseResNet_withCA_FPN_Transformer_4D(PoseResNet_withCA_FPN_Transformer):

    def __init__(self,  **kwargs):
        super().__init__( **kwargs)
        pretrained = 'PyMAF/data/pretrained_model/swin_large_patch4_window12_384_22k.pth'
        
        self.FPN = FPN([192,384,768,1536], 1024, 4)
        self.fuser_channels = 1024
        self.fuser = ConvModule(
                        self.fuser_channels ,
                        self.fuser_channels * 2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv2d'))
    def forward(self, x):
        x = self.swin_transformer(x)
        x = self.FPN(x)[-1]
        x = self.fuser(x)
        if self.global_mode:
            g_feat = self.avgpool(x)
            g_feat = g_feat.view(g_feat.size(0), -1)
            s_feat_list = x
        else:
            g_feat = None
            if self.extra.NUM_DECONV_LAYERS == 3:
                deconv_blocks = [self.deconv_layers[0:3], self.deconv_layers[3:6], self.deconv_layers[6:9]]

            s_feat_list = []
            s_feat = x
            for i in range(self.extra.NUM_DECONV_LAYERS):
                s_feat = deconv_blocks[i](s_feat)
                s_feat_list.append(s_feat)

        return s_feat_list, g_feat


resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}

def get_resnet_encoder(cfg, is_train=True, global_mode=False, **kwargs):
    num_layers = cfg.POSE_RES_MODEL.EXTRA.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]
    kwargs['layers'] = layers
    kwargs['cfg'] = cfg
    kwargs['global_mode'] = global_mode
    kwargs['block'] = block_class
    model = PoseResNet_withCA_FPN_Transformer_4D( **kwargs)

    # if is_train and cfg.POSE_RES_MODEL.INIT_WEIGHTS:
    #     if num_layers == 50:
    #         if cfg.POSE_RES_MODEL.PRETR_SET in ['imagenet']:
    #             model.init_weights(cfg.POSE_RES_MODEL.PRETRAINED_IM)
    #             logger.info('loaded ResNet imagenet pretrained model')
    #         elif cfg.POSE_RES_MODEL.PRETR_SET in ['coco']:
    #             model.init_weights(cfg.POSE_RES_MODEL.PRETRAINED_COCO)
    #             logger.info('loaded ResNet coco pretrained model')
    #     else:
    #         raise NotImplementedError

    return model