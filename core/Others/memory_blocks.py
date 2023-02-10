from turtle import forward
import torch
import torch.nn as nn
from core.yolox.models.network_blocks import BaseConv, get_activation
from torch.autograd import Variable
import math

class memoryModel(nn.Module):
    def __init__(self, memory_blocks):
        super().__init__()

        if not (memory_blocks is None):
            self.lstms = memory_blocks
            self.has_memory = True
        else:
            self.has_memory = False
    
    def clean_memory(self):
        if self.has_memory:
            for i in range(len(self.lstms)):
                self.lstms[i].clean_memory()
    
    def forward(self,x_in):
        x_out = []
        for i in range(len(self.lstms)):
            x_out.append(self.lstms[i](x_in[i]))
        return x_out

def makeMemoryBlocks(lstm, kernel_sizes, in_channels, out_channels, strides, act):
    lstms = nn.ModuleList()
    for i in range(len(kernel_sizes)):
        lstms.append(lstm(in_channels[i], out_channels[i], kernel_sizes[i], strides[i], act))
    return lstms

class memoryBlocks(nn.Module):
    def __init__(self):
        super().__init__()
        self.clean_memory()
    
    def clean_memory(self):
        self.has_memory = False

# class ConvLSTMCell(memoryBlocks):
#     def __init__(self, input_channel, output_channel, kernel_size):
#         super().__init__(input_channel, output_channel, kernel_size)

#         self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
#                                 out_channels=4 * self.hidden_dim,
#                                 kernel_size=self.kernel_size,
#                                 padding=self.padding)

#     def forward(self, x):
#         '''
#         :param input_tensor:[batch,dim,inp_height,inp_width]
#         :param cur_state: [h,c] h:[batch,dim,H,W]
#         :return:
#         '''
#         if not (self.has_memory):
#             self.h = Variable(torch.zeros([size for size in x.shape[:-1]] + [self.hidden_dim]).to(x.device))
#             self.c = Variable(torch.zeros([size for size in x.shape[:-1]] + [self.hidden_dim]).to(x.device))
        
#         h_cur, c_cur = self.h, self.c
 
#         combined = torch.cat([x, h_cur], dim=1)  
 
#         combined_conv = self.conv(combined)
#         cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
#         i = torch.sigmoid(cc_i)
#         f = torch.sigmoid(cc_f)
#         o = torch.sigmoid(cc_o)
#         g = torch.tanh(cc_g)
 
#         c_next = f * c_cur + i * g
#         h_next = o * torch.tanh(c_next)

#         self.h = h_next
#         self.c = c_next
#         self.has_memory = True
 
#         return h_next

class ConvLSTMCell(memoryBlocks):

    def __init__(self, input_dim, hidden_dim, kernel_size, stride, act):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel for both cnn and rnn.
        cnn_dropout, rnn_dropout: float
            cnn_dropout: dropout rate for convolutional input.
            rnn_dropout: dropout rate for convolutional state.
        bias: bool
            Whether or not to add the bias.
        peephole: bool
            add connection between cell state to gates
        layer_norm: bool
            layer normalization 
        """

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = int(self.kernel_size/2)
        self.stride = stride
        self.bias = True
        
        self.input_conv = nn.Conv2d(in_channels=self.input_dim, out_channels=4*self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  stride = self.stride,
                                  padding=self.padding,
                                  bias=self.bias)
        self.rnn_conv = nn.Conv2d(self.hidden_dim, out_channels=4*self.hidden_dim, 
                                  kernel_size = self.kernel_size,
                                  padding=math.floor(self.kernel_size/2),
                                  bias=self.bias)
        
    
    def forward(self, x):

        x_conv = self.input_conv(x)

        if not (self.has_memory):
            self.h = Variable(torch.zeros((x_conv.shape[0], self.hidden_dim, x_conv.shape[2], x_conv.shape[3])).to(x.device))
            self.c = Variable(torch.zeros((x_conv.shape[0], self.hidden_dim, x_conv.shape[2], x_conv.shape[3])).to(x.device))
        
        h_cur, c_cur = self.h, self.c

        # separate i, f, c o
        x_i, x_f, x_c, x_o = torch.split(x_conv, self.hidden_dim, dim=1)
        
        h_conv = self.rnn_conv(h_cur)
        # separate i, f, c o
        h_i, h_f, h_c, h_o = torch.split(h_conv, self.hidden_dim, dim=1)
        
        f = torch.sigmoid((x_f + h_f))
        i = torch.sigmoid((x_i + h_i))
        
        g = torch.tanh((x_c + h_c))
        c_next = f * c_cur + i * g

        o = torch.sigmoid((x_o + h_o))

        h_next = o * torch.tanh(c_next)

        self.h = h_next
        self.c = c_next
        self.has_memory = True

        return h_next

class BaseConvNoAct(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=1.0,
        act="gelu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = BaseConvNoAct
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(hidden_channels, hidden_channels, 3, stride=1, act=act)
        self.conv3 = Conv(hidden_channels, out_channels, 1, stride=1)
        self.use_add = shortcut and in_channels == out_channels
        self.act = get_activation(act)

    def forward(self, x):
        y = self.conv3(self.conv2(self.conv1(x)))
        if self.use_add:
            y = y + x
        return self.act(y)

class recConvCell(memoryBlocks):

    def __init__(self, input_dim, hidden_dim, kernel_size, stride, act):

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.bconv1 = Bottleneck(input_dim * 2, hidden_dim, act = act)
        self.bconv2 = Bottleneck(input_dim * 2, hidden_dim, act = act)

        self.wz = BaseConv(hidden_dim, hidden_dim, 3, 1, hidden_dim, act = act)
        self.wr = BaseConv(hidden_dim, hidden_dim, 3, 1, hidden_dim, act = act)
        self.wH = BaseConv(hidden_dim, hidden_dim, 3, 1, hidden_dim, act = act)
        
    
    def forward(self, m):

        if not (self.has_memory):
            self.h = Variable(torch.zeros((m.shape[0], self.hidden_dim, m.shape[2], m.shape[3])).to(m.device))
        
        h1 = self.h

        # separate i, f, c o
        mh1 = self.bconv1(torch.cat([h1,m],1))
        r = torch.sigmoid(self.wr(mh1))
        z = torch.sigmoid(self.wz(mh1))
        
        mr = self.bconv2(torch.cat([h1 * r,m],1))
        H = torch.sigmoid(self.wH(mr))

        h = z * H + (1 - z) * h1

        self.h = h
        self.has_memory = True

        return h

# class LSTEM(memoryBlocks):
#     def __init__(self, x_shape, C, input_channel, input_shape, kernel_size):
#         super().__init__(x_shape, C, input_channel, input_shape, kernel_size)
#         rate = int(input_shape[0] / x_shape[0])
#         rate1 = int(input_shape[1] / x_shape[1])
#         assert rate == rate1

#         self.pooling = nn.AvgPool2d(rate, rate)
#         self.batchnorm = nn.BatchNorm2d(1)
 
#         self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
#                               out_channels=2 * self.hidden_dim,
#                               kernel_size=self.kernel_size,
#                               padding=self.padding)
        
#         self.conv2 = nn.Conv2d(input_channel * 2 + kernel_size ** 2, 2 * self.hidden_dim, 1, padding=0)

#         self.softmax = nn.Softmax(dim = -1)

#         masks = []
#         for i in range(-1, 2):
#             for j in range(-1, 2):
#                 mask = torch.zeros((1, 1, input_shape[0], input_shape[1]))
#                 if i < 0: 
#                     mask[:,:,i:] = -100.0
#                 if i > 0:
#                     mask[:,:,:i] = -100.0
#                 if j < 0: 
#                     mask[:,:,:,j:] = -100.0
#                 if j > 0:
#                     mask[:,:,:,:j] = -100.0
#                 masks.append(mask)
#         mask = torch.stack(masks,dim=-1)
#         self.register_buffer("mask",mask)

#     def forward(self, x, input):
#         '''
#         :param input_tensor:[batch,dim,inp_height,inp_width]
#         :param cur_state: [h,c] h:[batch,dim,H,W]
#         :return:
#         '''
            
#         input = torch.where(input[...,0] > 0, torch.ones_like(input[...,-1:,0]), torch.zeros_like(input[...,-1:,0]))
#         input_pool = self.batchnorm(self.pooling(input))
        
#         if not(self.has_memory):
#             self.h = Variable(torch.zeros_like(x))
#             self.c = Variable(torch.zeros_like(x))
#             self.input_pool_past = torch.zeros_like(input_pool)

#         h_cur, c_cur, input_pool_past = self.h, self.c, self.input_pool_past
 
#         combined = torch.cat([x, h_cur], dim=1)  
 
#         combined_conv = self.conv(combined)
#         cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
#         x_afts = []
#         for i in range(-int(self.kernel_size/2), int(self.kernel_size/2) + 1):
#             for j in range(-int(self.kernel_size/2), int(self.kernel_size/2) + 1):
#                 x_afts.append(torch.roll(x, (i,j), dims=(2, 3)))
#         x_aft = torch.stack(x_afts, dim=-1)  #B, C, H, W, size
#         joint = torch.sum(h_cur * x_aft, dim=2)/(self.hidden_dim ** 0.5) #B, H, W, size
#         joint = joint + self.mask
#         y = self.softmax(joint) - 1/9   #B, H, W, size
#         y = y.permute(0,3,1,2).contiguous()   #B, size, H, W

#         features = torch.cat([y, input_pool_past, input_pool - input_pool_past], dim=1)

#         features_conv = self.conv2(features)
#         cc_i, cc_f = torch.split(features_conv, self.hidden_dim, dim=1)

#         i = torch.sigmoid(cc_i)
#         f = torch.sigmoid(cc_f)
#         o = torch.sigmoid(cc_o)
#         g = torch.tanh(cc_g)
 
#         c_next = f * c_cur + i * g
#         h_next = o * torch.tanh(c_next)

#         self.h = h_next
#         self.c = c_next
#         self.input_pool_past = input_pool
#         self.has_memory = True
 
#         return h_next

# class LSTEM1(nn.Module):
#     def __init__(self, H, W, C, L, rate, pool_kernel_size=3, act = "silu"):
#         super(LSTEM1, self).__init__()

#         H = H // rate
#         W = W // rate
#         self.pooling = nn.AvgPool2d(rate, rate)
#         self.batchnorm = nn.BatchNorm2d(1)
        
#         self.conv1 = nn.Conv2d(1, pool_kernel_size * pool_kernel_size + 2, 1, padding=0, bias=False)
#         self.conv2 = nn.Conv2d(1, pool_kernel_size * pool_kernel_size + 1, 1, padding=0, bias=False)
#         # self.conv1 = nn.Conv2d(pool_kernel_size * pool_kernel_size + 2, 1, 1, padding=0)
#         # self.conv2 = nn.Conv2d(pool_kernel_size * pool_kernel_size + 1, 1, 1, padding=0)
#         self.softmax = nn.Softmax(dim = -1)
#         self.sigmoid = nn.Sigmoid()
#         self.conv3 = BaseConv(C * L, C, 1, 1, act = act)

#         self.pool_kernel_size = pool_kernel_size
#         masks = []
#         for i in range(-int(self.pool_kernel_size / 2), int(self.pool_kernel_size / 2) + 1):
#             for j in range(-int(self.pool_kernel_size / 2), int(self.pool_kernel_size / 2) + 1):
#                 mask = torch.zeros((1, 1, H, W))
#                 if i < 0: 
#                     mask[:,:,i:] = -100.0
#                 if i > 0:
#                     mask[:,:,:i] = -100.0
#                 if j < 0: 
#                     mask[:,:,:,j:] = -100.0
#                 if j > 0:
#                     mask[:,:,:,:j] = -100.0
#                 masks.append(mask)
#         mask = torch.stack(masks,dim=-1)
#         self.register_buffer("mask",mask)

#         sur = int((pool_kernel_size ** 2) / 2)

#         base1 = torch.ones((1, 1, 1, 1),dtype=torch.float32)
#         self.register_buffer("base1",base1)

#         weights1 = []
#         weights1.append(-torch.ones((1, sur, 1, 1),dtype=torch.float32))
#         weights1.append(torch.ones((1, 1, 1, 1),dtype=torch.float32))
#         weights1.append(-torch.ones((1, sur, 1, 1),dtype=torch.float32))
#         weights1.append(-torch.ones((1, 2, 1, 1),dtype=torch.float32))
#         weights1 = torch.cat(weights1,dim=1)
#         self.register_buffer("weights1",weights1)

#         self.bias1 = nn.Parameter(torch.zeros((1, 1, 1, 1)))

#         base2 = torch.ones((1, 1, 1, 1),dtype=torch.float32)
#         self.register_buffer("base2",base2)

#         weights2 = []
#         weights2.append(torch.ones((1, sur, 1, 1),dtype=torch.float32))
#         weights2.append(torch.ones((1, 1, 1, 1),dtype=torch.float32))
#         weights2.append(torch.ones((1, sur, 1, 1),dtype=torch.float32))
#         weights2.append(torch.ones((1, 1, 1, 1),dtype=torch.float32))
#         weights2 = torch.cat(weights2,dim=1)
#         self.register_buffer("weights2",weights2)

#         self.bias2 = nn.Parameter(torch.zeros((1, 1, 1, 1)))


#     def forward(self, x, ref, m_ref):    #B, L, C, H, W
#         B, L, C, H, W = x.shape
#         x_pre = x[:,:-1].contiguous()
#         x_aft = x[:,1:].contiguous()
#         x_afts = []
#         for i in range(-int(self.pool_kernel_size / 2), int(self.pool_kernel_size / 2) + 1):
#             for j in range(-int(self.pool_kernel_size / 2), int(self.pool_kernel_size / 2) + 1):
#                 x_afts.append(torch.roll(x_aft,(i,j),dims=(3, 4)))
#         x_aft = torch.stack(x_afts,dim=-1)  #B, L-1, C, H, W, size
#         joint = torch.sum(x_pre[:,:,:,:,:,None] * x_aft,dim=2)/(C**0.5) #B, L-1, H, W, size
#         joint = joint + self.mask
#         y = self.softmax(joint) - 1/(self.pool_kernel_size ** 2)   #B, L-1, H, W, size
#         y = y.permute(0,1,4,2,3).contiguous().view(B * (L-1), self.pool_kernel_size * self.pool_kernel_size, H, W)    #B * (L-1), size, H, W
#         ref = torch.sum(ref,dim=1)[:,None]/2
#         m_ref = torch.sum(m_ref,dim=1)[:,None]/2
#         ref = self.batchnorm(self.pooling(ref))
#         m_ref = self.batchnorm(self.pooling(m_ref))
#         y1 = torch.cat([y, ref - m_ref, m_ref],dim=1)
#         y2 = torch.cat([y, ref - m_ref],dim=1)

#         y1 = torch.sum(torch.square(self.conv1(self.base1)) * self.weights1 * y1 + self.bias1, dim=1)
#         y2 = torch.sum(torch.square(self.conv2(self.base2)) * self.weights2 * y2 + self.bias2, dim=1)
#         y1 = self.sigmoid(y1)[:,None]   #B, L-1, H, W
#         y2 = self.sigmoid(y2)[:,None]   #B, L-1, H, W
        
#         # y1 = self.sigmoid(self.conv1(y1))
#         # y2 = self.sigmoid(self.conv2(y2))
#         y1 = torch.cat([y1, torch.ones_like(y1[:,-1:])], dim=1)[:,:,None]    #B, L, 1, H, W
#         y2 = torch.cat([torch.ones_like(y2[:,:1]),y2], dim=1)[:,:,None]    #B, L, 1, H, W
#         x = x * y1 * y2
#         x = x.view(B, L * C, H, W)
#         return self.conv3(x)