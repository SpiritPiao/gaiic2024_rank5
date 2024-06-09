# from DCNv4.modules.dcnv4 import DCNv4
import torch.nn as nn
from ..layers.pkinet import PKIOwn
from .ops_dcnv3.modules import DCNv3
class DCNV4_YOLO(nn.Module):
    def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if inc != ouc:
            self.stem_conv = Conv(inc, ouc, k=1)
        self.dcnv4 = DCNv4(ouc, kernel_size=k, stride=s, pad=autopad(k, p, d), group=g, dilation=d)
        self.bn = nn.BatchNorm2d(ouc)
        self.act = nn.SiLU()

    def forward(self, x):
        if hasattr(self, 'stem_conv'):
            x = self.stem_conv(x)
        x = self.dcnv4(x, (x.size(2), x.size(3)))
        x = self.act(self.bn(x))
        return x
class DCNV3_YOLO(nn.Module):
    def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        
        if inc != ouc:
            self.stem_conv = Conv(inc, ouc, k=1)
        self.dcnv3 = DCNv3(ouc, kernel_size=k, stride=s, pad=autopad(k, p, d), group=g, dilation=d)
        self.bn = nn.BatchNorm2d(ouc)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    def forward(self, x):
        if hasattr(self, 'stem_conv'):
            x = self.stem_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.dcnv3(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(self.bn(x))
        return x
class DCNV4_CSP(nn.Module):
    def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        ouc_ = int(ouc // 2)
        self.conv1 = Conv(ouc, ouc_)
        self.conv2 = Conv(ouc, ouc_)

        self.dcnv4_1 = DCNV4_YOLO(ouc_, ouc_, k=k, s=s, p=autopad(k, p, d), g=g, d=d)
        self.dcnv4_2 = DCNV4_YOLO(ouc_, ouc_, k=k, s=s, p=autopad(k, p, d), g=g, d=d)

        
        self.act = nn.SiLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        shortcut = x2
        x2 = self.dcnv4_1(x2)
        x2 = self.dcnv4_2(x2)
        x2 = x2 + shortcut
        out = torch.cat([x1, x2], 1)
        
        return out
    

class DCNV3_CSP(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=3, e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=3, e=0.5, s = 1, p=None, d=1):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DCNV3_YOLO(c1, c1, k=k, s=s, p=1, g=g, d=d)
        self.cv2 = DCNV3_YOLO(c1, c1, k=k, s=s, p=1, g=g, d=d)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p



import torch
import torch.nn as nn
class EAEF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp_pool = Feature_Pool(dim)
        self.dwconv = nn.Conv2d(dim*2,dim*2,kernel_size=7,padding=3,groups=dim)
        self.ecse = Channel_Attention(dim*2)
        # self.ccse = Channel_Attention(dim)
        self.sse_r = Spatial_Attention(dim)
        self.sse_t = Spatial_Attention(dim)

        self.dcn1 = DCNv2(dim, dim, 3)
        self.dcn2 = DCNv2(dim, dim, 3)

        self.dcn3 = DCNv2(dim, dim, 3)
        self.dcn4 = DCNv2(dim, dim, 3)
    def forward(self, RGB, T):
        ############################################################################
        # RGB,T = x[0], x[1]
        RGB = self.dcn1(RGB)
        T = self.dcn2(T)


        b, c, h, w = RGB.size()
        rgb_y = self.mlp_pool(RGB)
        t_y = self.mlp_pool(T)
        rgb_y = rgb_y / rgb_y.norm(dim=1, keepdim=True)
        t_y = t_y / t_y.norm(dim=1, keepdim=True)
        rgb_y = rgb_y.view(b, c, 1)
        t_y = t_y.view(b, 1, c)
        logits_per = c * rgb_y @ t_y
        cross_gate = torch.diagonal(torch.sigmoid(logits_per)).reshape(b, c, 1, 1)
        add_gate = torch.ones(cross_gate.shape).cuda() - cross_gate
        ##########################################################################
        New_RGB_e = RGB * cross_gate
        New_T_e = T * cross_gate
        New_RGB_c = RGB * add_gate
        New_T_c = T * add_gate
        x_cat_e = torch.cat((New_RGB_e, New_T_e), dim=1)
        ##########################################################################
        fuse_gate_e = torch.sigmoid(self.ecse(self.dwconv(x_cat_e)))
        rgb_gate_e, t_gate_e = fuse_gate_e[:, 0:c, :], fuse_gate_e[:, c:c * 2, :]
        ##########################################################################
        New_RGB = New_RGB_e * rgb_gate_e + New_RGB_c
        New_T = New_T_e * t_gate_e + New_T_c
        ##########################################################################
        New_fuse_RGB = self.sse_r(New_RGB)
        New_fuse_T = self.sse_t(New_T)
        attention_vector = torch.cat([New_fuse_RGB, New_fuse_T], dim=1)
        attention_vector = torch.softmax(attention_vector, dim=1)
        attention_vector_l, attention_vector_r = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        New_RGB = New_RGB * attention_vector_l
        New_T = New_T * attention_vector_r

        New_RGB = self.dcn3(New_RGB)
        New_T = self.dcn4(New_T)
        New_fuse = New_T + New_RGB
        out = New_RGB, New_T, New_fuse
        ##########################################################################
        return out
class Feature_Pool(nn.Module):
    def __init__(self, dim, ratio=2):
        super(Feature_Pool, self).__init__()
        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.down = nn.Linear(dim, dim * ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim * ratio, dim)
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.up(self.act(self.down(self.gap_pool(x).permute(0,2,3,1)))).permute(0,3,1,2).view(b,c)
        return y
class Channel_Attention(nn.Module):
    def __init__(self, dim, ratio=16):
        super(Channel_Attention, self).__init__()
        self.gap_pool = nn.AdaptiveMaxPool2d(1)
        self.down = nn.Linear(dim, dim//ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim//ratio, dim)
    def forward(self, x):
        max_out = self.up(self.act(self.down(self.gap_pool(x).permute(0,2,3,1)))).permute(0,3,1,2)
        return max_out

class Spatial_Attention(nn.Module):
    def __init__(self, dim):
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1,bias=True)
    def forward(self, x):
        x1 = self.conv1(x)
        return x1
class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, groups=1, act=True, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (autopad(kernel_size, padding), autopad(kernel_size, padding))
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.reset_parameters()

    def forward(self, x):

        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        import math
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()




