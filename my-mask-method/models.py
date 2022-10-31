import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import transforms

from resnet import resnet50, Bottleneck


class MyModel(nn.Module):
    def __init__(self, num_classes, last_conv_stride=1, last_conv_dilation=1):
        super(MyModel, self).__init__()

        self.base = resnet50(pretrained=True, last_conv_stride=last_conv_stride, last_conv_dilation=last_conv_dilation)

        self.rvd_conv = nn.Sequential(nn.Conv2d(2048, 256, 1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 1, 1))

        self.global_conv = nn.Sequential(nn.Conv2d(2048, 512, 1),
                                            nn.BatchNorm2d(512),
                                            nn.ReLU(inplace=True))

        self.global_class = nn.Linear(512, num_classes)

        self.bottleneck = Bottleneck(2048, 512)

        self.fdb_conv = nn.Sequential(nn.Conv2d(2048, 1024, 1),
                                            nn.BatchNorm2d(1024),
                                            nn.ReLU(inplace=True))
        
        self.fdb_class = nn.Linear(1024, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))


    def forward(self, x, occlusion_mask=None):
        # shape [N, C, H, W]
        feat = self.base(x)

        # global branch
        global_feat = self.avgpool(feat)
        global_feat = self.global_conv(global_feat)
        global_feat = global_feat.view(global_feat.size(0), -1)
        global_logits = self.global_class(global_feat)

        # compute occlusion mask (region visibility discriminator)
        rvd_logits = self.rvd_conv(feat)

        # feature dropping branch

        # if occlusion mask is None we apply the mask calculated by 
        # the model, otherwise we use the provided mask (ground truth)
        if occlusion_mask is None:
            occlusion_mask = rvd_logits.detach()
            occlusion_mask = (torch.sigmoid(occlusion_mask) > 0.5).float()
            # occlusion_mask = torch.ones_like(rvd_logits)
        else:
            if occlusion_mask.size() != feat.size():
                occlusion_mask = transforms.functional.resize(occlusion_mask, feat.size()[-2:])
        
        fdb_feat = self.bottleneck(feat)

        fdb_feat = fdb_feat * occlusion_mask

        fdb_feat = self.maxpool(fdb_feat)
        fdb_feat = self.fdb_conv(fdb_feat)
        fdb_feat = fdb_feat.view(fdb_feat.size(0), -1)
        fdb_logits = self.fdb_class(fdb_feat)

        return global_feat, global_logits, fdb_feat, fdb_logits, rvd_logits, occlusion_mask


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = MyModel(100)
    model = model.to(device)

    model.train()

    print(model)

    x = torch.randn(64, 3, 384, 128)
    x = x.to(device)

    occlusion_mask = torch.randn(64, 1, 384, 128)
    occlusion_mask = occlusion_mask.to(device)

    global_feat, global_logits, fdb_feat, fdb_logits, rvd_logits, occlusion_mask = model(x, occlusion_mask)

    print(global_feat.size())
    print(global_logits.size())
    print(fdb_feat.size())
    print(fdb_logits.size())
    print(rvd_logits.size())
    print(occlusion_mask.size())
    

if __name__ == '__main__':
    main()

