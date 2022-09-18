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
        out = self.attn(q, k, v)[0]
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
    def __init__(self, num_classes, nk, nhead=8, local_conv_out_channels=256, last_conv_stride=1, last_conv_dilation=1):
        super(MyModel, self).__init__()

        self.base = resnet50(pretrained=True, last_conv_stride=last_conv_stride, last_conv_dilation=last_conv_dilation)

        self.conv = nn.Sequential(nn.Conv2d(2048, local_conv_out_channels, 1),
                                            nn.BatchNorm2d(local_conv_out_channels),
                                            nn.ReLU())
        
        self.enc_self_att = Attention(local_conv_out_channels, nhead)
        self.enc_ffn = FFN(local_conv_out_channels)

        self.enc_classifier = nn.Sequential(nn.BatchNorm1d(local_conv_out_channels),
                                            nn.Linear(local_conv_out_channels, num_classes))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.pk = nn.parameter.Parameter(torch.normal(mean=0, std=0.001, size=(nk, local_conv_out_channels)))
        self.dec_self_att = Attention(local_conv_out_channels, nhead)
        self.dec_cross_att = CrossAttention(local_conv_out_channels, nhead)
        self.dec_ffn = FFN(local_conv_out_channels)

        self.dec_classifier = nn.ModuleList()

        for _ in range(nk):
            self.dec_classifier.append(nn.Sequential(nn.BatchNorm1d(local_conv_out_channels),
                                                    nn.Linear(local_conv_out_channels, num_classes)))
        
    def forward(self, x):
        feat = self.base(x)
        feat = self.conv(feat)

        n, c, h, w = feat.size()

        enc_feat = feat.view(n, c, -1).permute(2, 0, 1)
        enc_feat = self.enc_self_att(enc_feat)
        enc_feat = self.enc_ffn(enc_feat)

        enc_feat_gap = self.avgpool(enc_feat.permute(1, 2, 0).view(n, c, h, w)).view(n, -1)

        enc_feat_logits = self.enc_classifier(enc_feat_gap)

        dec_feat = self.pk.unsqueeze(1).repeat(1, n, 1)
        dec_feat = self.dec_self_att(dec_feat)
        dec_feat = self.dec_cross_att(dec_feat, enc_feat, enc_feat)
        dec_feat = self.dec_ffn(dec_feat)

        dec_logits_list = []

        for i in range(dec_feat.size(0)):
            dec_logits_list.append(self.dec_classifier[i](dec_feat[i]))

        return enc_feat_gap, enc_feat_logits, dec_feat, dec_logits_list



def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = MyModel(100, 6)
    model = model.to(device)

    model.train()

    print(model)

    x = torch.randn(64, 3, 256, 128)
    x = x.to(device)

    enc_feat_gap, enc_feat_logits, dec_feat, dec_logits_list = model(x)

    print(enc_feat_gap.size())
    print(enc_feat_logits.size())
    print(dec_feat.size())
    print(torch.stack(dec_logits_list, 0).size())


if __name__ == '__main__':
    main()

