from turtle import forward
from core.yolox.models.network_blocks import BaseConv, Focus, get_activation
from core.swin_transformer.backbone import BasicLayer, PatchEmbed3D, PatchMerging, PatchMergingTime, trunc_normal_, rearrange
from core.swin_transformer.corr_extract import corr_BasicLayer
import torch
from torch import nn
import time
from math import log2
import numpy as np


class Temporal_Active_Focus(Focus):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="gelu"):
        super().__init__(in_channels, out_channels, ksize, stride, act)
        time_channels = int(in_channels/2)
        reduce_times = int(log2(time_channels))
        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        #self.dropouts = nn.ModuleList()
        for i in range(reduce_times-1):
            self.convs.append(nn.utils.weight_norm(nn.Conv2d(in_channels, in_channels, 1, groups = int(time_channels/(2 ** (i + 1))))))
            #self.dropouts.append(nn.Dropout2d(0.1))
            #self.relus.append(nn.ReLU)
        self.convs.append(nn.utils.weight_norm(nn.Conv2d(in_channels, in_channels, 1)))
        #self.dropouts.append(nn.Dropout2d(0.1))
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        for i in range(len(self.convs)):
            self.convs[i].weight.data.normal_(0, 0.01)
    
    def patch(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return x
    
    def forward(self, x):
        x = x[...,0]
        for i in range(len(self.convs)):
            #x = self.dropouts[i](self.relu(self.convs[i](x)))
            x = self.relu(self.convs[i](x))
        mask = self.patch(x)
        return self.conv(mask)

class Temporal_Active_Focus_connect(Focus):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="gelu"):
        time_channels = int(in_channels/2)
        self.embed_dim = 4
        reduce_times = int(log2(time_channels))
        super().__init__(self.embed_dim * reduce_times, out_channels, ksize, stride, act)
        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        #self.dropouts = nn.ModuleList()
        for i in range(reduce_times):
            if i == 0:
                input_dim = 2
            else:
                input_dim = self.embed_dim
            self.convs.append(nn.utils.weight_norm(nn.Conv2d(int(input_dim * time_channels), int(self.embed_dim * time_channels / 2), 1, groups = int(time_channels / 2))))
            time_channels = time_channels / 2
            #self.dropouts.append(nn.Dropout2d(0.1))
            #self.relus.append(nn.ReLU)
        self.trans_up = nn.Conv2d(self.embed_dim * reduce_times, self.embed_dim * reduce_times * 4, 1)
        self.act = get_activation(act)
        self.drop = nn.Dropout2d(0.1)
        self.trans_down = nn.Conv2d(self.embed_dim * reduce_times * 4, self.embed_dim * reduce_times, 1)
        #self.dropouts.append(nn.Dropout2d(0.1))
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        for i in range(len(self.convs)):
            self.convs[i].weight.data.normal_(0, 0.01)
    
    def patch(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return x
    
    def forward(self, x):
        x = x[...,0]
        xout = []
        for i in range(len(self.convs)):
            x = self.relu(self.convs[i](x))#self.dropouts[i](self.relu(self.convs[i](x)))
            xout.append(x[:,:self.embed_dim])
        x = torch.cat(xout, dim=1)
        xout = self.trans_up(x)
        xout = self.act(xout)
        xout = self.drop(xout)
        xout = self.trans_down(xout)
        xout = self.drop(xout)
        x = x + xout
        mask = self.patch(x)
        return self.conv(mask)

class Temporal_Active_Focus_corr(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="gelu"):
        super().__init__()
        time_channels = int(in_channels/2)
        reduce_times = int(log2(time_channels))
        embed_dim = 16
        self.embed_dim = embed_dim
        window_size = [2, 4, 4]
        self.deltas = [0, 5, 10, 25]
        #self.patch_embed = PatchEmbed3D(patch_size=patch_size, in_chans=2, embed_dim=embed_dim,norm_layer=None)
        self.patch_embed = nn.Conv2d(2, embed_dim, 2, 2)

        self.out_channels = out_channels

        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        self.dropouts = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(reduce_times):
            self.layer_norms.append(nn.LayerNorm(nn.LayerNorm((time_channels * embed_dim, 128, 160))))
            self.convs.append(nn.Conv2d(time_channels * embed_dim, time_channels * embed_dim, 1, groups = int(time_channels/(2 ** (i + 1)))))
            self.dropouts.append(nn.Dropout2d(0.1))
        
        self.patch_embed_ref = nn.Conv2d(2, embed_dim, 2, 2)

        self.convs_ref = nn.ModuleList()
        self.dropouts_ref = nn.ModuleList()
        self.layer_norms_ref = nn.ModuleList()
        for i in range(reduce_times-1):
            self.layer_norms_ref.append(nn.LayerNorm((time_channels * embed_dim, 128, 160)))
            self.convs_ref.append(nn.Conv2d(time_channels * embed_dim, time_channels * embed_dim, 1, groups = int(time_channels/(2 ** (i + 1)))))
            self.dropouts_ref.append(nn.Dropout2d(0.1))

        self.corr_extracts = nn.ModuleList()
        for i in range(reduce_times):
            self.corr_extracts.append(corr_BasicLayer(embed_dim * (2 ** i), len(self.deltas), window_size=window_size))

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            for i in range(len(self.convs)):
                self.convs[i].weight.data.normal_(0, 0.01)
                self.convs_ref[i].weight.data.normal_(0, 0.01)
        self.apply(_init_weights)

    def forward(self, x):
        x = x.view(x.shape[0], int(x.shape[1]/2), 2, x.shape[2], x.shape[3])    #B, D, C, H, W
        deltas = torch.tensor(self.deltas).to(x.device)[None,:,None,None,None,None]
        x = x[:, None]  #B, 1, D, C, H, W
        x_ref = 1 - torch.log1p(torch.expm1((1 - x) * 8.7) + deltas) / 8.7  #B, R, D, C, H, W
        B, R, D, C, H, W = x_ref.shape
        x = x.view(B * D, C, H, W)
        x = self.patch_embed(x)
        x_ref = x_ref.view(B * R * D, C, H, W)
        x_ref = self.patch_embed_ref(x_ref)

        x = x.view(B, 1, D, self.embed_dim, int(H // 2), int(W // 2))
        x_ref = x_ref.view(B, R, D, self.embed_dim, int(H // 2), int(W // 2))

        for i in range(len(self.convs)):
            x = self.corr_extracts[i](x, x_ref)
            B, R, D, C, H, W = x_ref.shape
            x = x.view(B, D * C, H, W)
            x_ref = x_ref.view(B * R, D * C, H, W)
            x = self.dropouts[i](self.relu(self.convs[i](self.layer_norms[i](x))))
            x = x.view(B, 1, int(D / 2), 2 * C, H, W)
            if i < len(self.convs) - 1:
                x_ref = self.dropouts_ref[i](self.relu(self.convs_ref[i](self.layer_norms_ref[i](x_ref))))
                x_ref = x_ref.view(B, R, int(D / 2), 2 * C, H, W)
        
        x = x.view(B, self.out_channels, H, W)
        return x

class myLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return (x - torch.mean(x, dim = [-4, -3, -2, -1], keepdim=True)) / torch.sqrt(torch.var(x, dim = [-4, -3, -2, -1], keepdim=True) + 1e-16)

class Temporal_Active_Focus_swin(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="gelu"):
        super().__init__()
        time_channels = int(in_channels/2)
        reduce_times = int(log2(time_channels-1))

        embed_dim = 16
        self.patch_norm = False
        #norm_layer = myLayerNorm
        norm_layer = nn.LayerNorm
        drop_rate=0.1
        drop_path_rate=0.2
        #drop_path_rate=0
        #if time_channels > 1:
        patch_size = [2, 2, 2]
        window_size = [2, 4, 5]
        # else:
        #     patch_size = [1, 2, 2]
        #     window_size = [1, 4, 5]
        
        depths = [2 for i in range(reduce_times)]
        reduces = [PatchMergingTime for i in range(reduce_times)]

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=2, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        #self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            # if i_layer == 0:
            #     in_dim = embed_dim
            #     out_dim = embed_dim * 2
            # else:
            #     in_dim = embed_dim * 2
            #     out_dim = embed_dim * 2
            in_dim = embed_dim * (2 ** i_layer)
            out_dim = embed_dim * (2 ** (i_layer+1))
            # if i_layer == len(depths) - 1:
            #     out_dim = out_channels
            layer = BasicLayer(
                D = int(time_channels / (2 ** i_layer)),
                H = 128,
                W = 160,
                in_dim=in_dim,
                out_dim=out_dim,
                depth=depths[i_layer],
                num_heads=1 * (2 ** i_layer),
                window_size=window_size,
                mlp_ratio=2.0,
                qkv_bias=False,
                qk_scale=None,
                drop=drop_rate,
                attn_drop=0.,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=reduces[i_layer],
                use_checkpoint=False)
            self.layers.append(layer)

        self.norm = norm_layer(out_dim)
        
        # self.conv2 = BaseConv(embed_dim * (2 ** len(depths)), out_channels, ksize=3, stride=1, act = act)
        # self.bn = nn.BatchNorm2d(out_channels)
        # self.act = get_activation(act)

        # add a norm layer for each output

        self.init_weights()

        self.conv2 = BaseConv(embed_dim * (2 ** (reduce_times)), out_channels, ksize=3, stride=1, act = act)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, x):
        """Forward function."""
        x = x.view(x.shape[0], int(x.shape[1]/2), 2, x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4).contiguous()

        x = self.patch_embed(x)

        #x = self.pos_drop(x)

        for i, layer in enumerate(self.layers):
            # start = time.time()
            x = layer(x.contiguous())
            # torch.cuda.synchronize()
            # infer_time = time.time() - start
            # print(infer_time)

        if not (self.norm is None):
            x = rearrange(x, 'n c d h w -> n d h w c')
            x = self.norm(x)
            x = rearrange(x, 'n d h w c -> n c d h w').contiguous()
        x = x.contiguous().squeeze(2)
        
        x = self.conv2(x)

        return x

class Temporal_Active_Focus_3D(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="gelu"):
        super().__init__()
        time_channels = int(in_channels/2)
        reduce_times = int(log2(time_channels))

        embed_dim = 32

        self.embed_dim = embed_dim
        self.time_channels = time_channels

        # self.convs = nn.ModuleList()
        # self.relu = nn.ReLU()
        # #self.dropouts = nn.ModuleList()
        # for i in range(reduce_times):
        #     if i == 0:
        #         stride = [2, 2, 2]
        #         in_ch = 2
        #         out_ch = embed_dim
        #         patch_size = [2, 2, 2]
        #         padding = 0
        #     else:
        #         stride = [2, 1, 1]
        #         in_ch = embed_dim * (2 ** (i-1))
        #         out_ch = embed_dim * (2 ** i)
        #         patch_size = [2, 3, 3]
        #         padding = [0, 1, 1]
        #     self.convs.append(nn.utils.weight_norm(nn.Conv3d(in_ch, out_ch, patch_size, stride, padding)))
        #     #self.dropouts.append(nn.Dropout3d(0.1))
        # self.init_weights()

        #self.act = get_activation(act)

        self.convs = nn.ModuleList()
        #self.bns = nn.ModuleList()
        #self.dropouts = nn.ModuleList()

        self.convs.append(BaseConv(in_channels, int(time_channels / 2 * embed_dim), 3, 2, int(time_channels/2), True, act))

        for i in range(1, reduce_times):
            self.convs.append(BaseConv(int(time_channels / (2 ** i) * embed_dim), int(time_channels / (2 ** (i + 1)) * embed_dim), 3, 1, int(time_channels/(2 ** (i + 1))), True, act))

        # self.convs = nn.ModuleList()
        # self.relu = nn.ReLU()
        # self.dropouts = nn.ModuleList()
        # self.convs.append(nn.utils.weight_norm(nn.Conv2d(in_channels, in_channels * 4, 2, 2, groups = int(time_channels/2))))
        # self.dropouts.append(nn.Dropout2d(0.1))
        # for i in range(1,reduce_times-1):
        #     self.convs.append(nn.utils.weight_norm(nn.Conv2d(in_channels * 4, in_channels * 4, 3, 1, 1, groups = int(time_channels/(2 ** (i + 1))))))
        #     self.dropouts.append(nn.Dropout2d(0.1))
        #     #self.relus.append(nn.ReLU)
        # self.convs.append(nn.utils.weight_norm(nn.Conv2d(in_channels * 4, 32, 3, 1, 1)))
        # self.dropouts.append(nn.Dropout2d(0.1))
        # self.trans_dim = nn.Conv2d(in_channels * 4, 32, 1)
        #self.init_weights()

        self.conv2 = BaseConv(reduce_times * embed_dim, out_channels, ksize=1, stride=1, act = act, dropout=0.25)

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        for i in range(len(self.convs)):
            self.convs[i].weight.data.normal_(0, 0.01)

    def forward(self, x):
        #x = x.view(x.shape[0], int(x.shape[1]/2), 2, x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4).contiguous()
        # x = x[...,0]
        # for i in range(len(self.convs)):
        #     xout = self.dropouts[i](self.relu(self.convs[i](x)))
        #     if x.shape == xout.shape:
        #         x = xout + x
        #     else:
        #         x = xout + self.trans_dim(x)
        #x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])

        x = x[...,0]
        # B, C, H, W = x.shape
        # time_channels = self.time_channels
        xout = []
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            # #time_channels = time_channels // 2
            # #x = x.view(B * time_channels, self.embed_dim, H // 2, W // 2)
            # x = self.bns[i](x)
            # #x = x.view(B, time_channels * self.embed_dim, H // 2, W // 2)
            # x = self.act(x)
            xout.append(x[:,:self.embed_dim])
        x = self.conv2(torch.cat(xout, dim = 1))
        return x