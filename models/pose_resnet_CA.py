import torch
import torch.nn as nn
from .pose_resnet import *

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CABlock(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CABlock, self).__init__()
        # height avg pooling
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # width avg pooling
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out



class PoseResNet_withCA(PoseResNet):

    def __init__(self,  **kwargs):
        super().__init__( **kwargs)

        self.CA = CABlock(2048,2048)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #attention
        x = self.CA(x)
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

def get_resnet_CA_FPN_Transformer_encoder(cfg, is_train=True, global_mode=False, **kwargs):
    num_layers = cfg.POSE_RES_MODEL.EXTRA.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]
    kwargs['layers'] = layers
    kwargs['cfg'] = cfg
    kwargs['global_mode'] = global_mode
    kwargs['block'] = block_class
    model = PoseResNet_withCA( **kwargs)

    if is_train and cfg.POSE_RES_MODEL.INIT_WEIGHTS:
        if num_layers == 50:
            if cfg.POSE_RES_MODEL.PRETR_SET in ['imagenet']:
                model.init_weights(cfg.POSE_RES_MODEL.PRETRAINED_IM)
                logger.info('loaded ResNet imagenet pretrained model')
            elif cfg.POSE_RES_MODEL.PRETR_SET in ['coco']:
                model.init_weights(cfg.POSE_RES_MODEL.PRETRAINED_COCO)
                logger.info('loaded ResNet coco pretrained model')
        else:
            raise NotImplementedError

    return model