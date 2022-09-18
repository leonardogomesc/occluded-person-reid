from turtle import forward
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from resnet import resnet50

class Attention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        out = self.attn(x, x, x)[0]
        out = self.norm(x + out)

        return out

class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, q, k, v):
        out = self.attn(q, k, v)
        out = self.norm(q + out)

        return out


class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model),
                                nn.ReLU(),
                                nn.Linear(d_model, d_model))
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        out = self.ffn(x)
        out = self.norm(x + out)

        return out


class MyModel(nn.Module):
    def __init__(self, num_classes, nhead=8, local_conv_out_channels=256, last_conv_stride=1, last_conv_dilation=1):
        super(MyModel, self).__init__()

        self.base = resnet50(pretrained=True, last_conv_stride=last_conv_stride, last_conv_dilation=last_conv_dilation)

        self.conv = nn.Sequential(nn.Conv2d(2048, local_conv_out_channels, 1),
                                            nn.BatchNorm2d(local_conv_out_channels),
                                            nn.ReLU())
        
        self.self_att_1 = Attention(local_conv_out_channels, nhead)
        self.ffn_1 = FFN(local_conv_out_channels)

        self.bn_1 = nn.BatchNorm2d(local_conv_out_channels)
        self.classifier_1 = nn.Linear(local_conv_out_channels, num_classes)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        return


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

