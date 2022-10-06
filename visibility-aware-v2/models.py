import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from resnet import resnet50


class MyModel(nn.Module):
    def __init__(self, num_classes, last_conv_stride=1, last_conv_dilation=1, num_stripes=6, local_conv_out_channels=256):
        super(MyModel, self).__init__()

        self.base = resnet50(pretrained=True, last_conv_stride=last_conv_stride, last_conv_dilation=last_conv_dilation)
        
        self.num_stripes = num_stripes

        self.local_conv_list = nn.ModuleList()

        for _ in range(num_stripes):
            self.local_conv_list.append(nn.Sequential(nn.Conv2d(2048, local_conv_out_channels, 1),
                                                        nn.BatchNorm2d(local_conv_out_channels),
                                                        nn.ReLU(inplace=True)))

        self.fc_list = nn.ModuleList()

        for _ in range(num_stripes):
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_list.append(fc)

        self.rvd_conv_list = nn.ModuleList()

        for _ in range(num_stripes):
            self.rvd_conv_list.append(nn.Sequential(nn.Conv2d(2048, local_conv_out_channels, 1),
                                                        nn.BatchNorm2d(local_conv_out_channels),
                                                        nn.ReLU(inplace=True)))

        self.rvd_fc_list = nn.ModuleList()

        for _ in range(num_stripes):
            fc = nn.Linear(local_conv_out_channels, 2)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.rvd_fc_list.append(fc)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.global_fc = nn.Linear(2048, num_classes)
        init.normal_(self.global_fc.weight, std=0.001)
        init.constant_(self.global_fc.bias, 0)


    def forward(self, x):
        # shape [N, C, H, W]
        feat = self.base(x)

        assert feat.size(2) % self.num_stripes == 0

        stripe_h = int(feat.size(2) / self.num_stripes)

        local_feat_list = []
        local_logits_list = []

        rvd_logits_list = []

        for i in range(self.num_stripes):
            # shape [N, C, 1, 1]
            local_feat = F.avg_pool2d(feat[:, :, i * stripe_h: (i + 1) * stripe_h, :], (stripe_h, feat.size(-1)))

            # shape [N, c, 1, 1]
            local_conv_feat = self.local_conv_list[i](local_feat)
            # shape [N, c]
            local_conv_feat = local_conv_feat.view(local_conv_feat.size(0), -1)
            local_feat_list.append(local_conv_feat)
            local_logits_list.append(self.fc_list[i](local_conv_feat))

            # rvd
            # shape [N, c, 1, 1]
            rvd_conv_feat = self.rvd_conv_list[i](local_feat)
            # shape [N, c]
            rvd_conv_feat = rvd_conv_feat.view(rvd_conv_feat.size(0), -1)
            # shape [N, 2]
            rvd_logits_list.append(self.rvd_fc_list[i](rvd_conv_feat))
        

        global_feat = self.avgpool(feat)
        global_feat = global_feat.view(global_feat.size(0), -1)
        global_logits = self.global_fc(global_feat)

        return global_feat, global_logits, local_feat_list, local_logits_list, rvd_logits_list


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = MyModel(100)
    model = model.to(device)

    model.train()

    print(model)

    x = torch.randn(64, 3, 384, 128)
    x = x.to(device)

    global_feat, global_logits, local_feat_list, local_logits_list, rvd_logits_list = model(x)

    print(global_feat.size())
    print(global_logits.size())
    print(len(local_feat_list))
    print(local_feat_list[0].size())
    print(len(local_logits_list))
    print(local_logits_list[0].size())
    print(len(rvd_logits_list))
    print(rvd_logits_list[0].size())



if __name__ == '__main__':
    main()

