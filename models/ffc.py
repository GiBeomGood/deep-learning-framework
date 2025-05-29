import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, fft, nn


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        # bn_layer not used
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels * 2,
                out_channels=out_channels * 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),
        )

    def forward(self, x: Tensor):
        _, _, height, width = x.size()

        x = fft.rfft2(x, norm="ortho")  # (-1 x C x H x W/2+1)
        x = torch.cat([x.real, x.imag], dim=1)  # (-1 x 2C x H x W/2+1)

        x = self.conv_layer(x)  # (-1 x 2C x H x W/2+1)

        real, imag = x.chunk(2, dim=1)  # (-1 x C x H x W/2+1)
        x = torch.complex(real, imag)  # (-1 x C x H x W/2+1)
        output = fft.irfft2(x, s=(height, width), norm="ortho")  # (-1 x C x H x W)

        return output


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, enable_lfu=True):
        # bn_layer not used
        super().__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:  # noqa: PLR2004
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
        )
        self.fourier_unit = FourierUnit(out_channels // 2, out_channels // 2)
        if enable_lfu is True:
            self.local_fourier_unit = FourierUnit(out_channels // 2, out_channels // 2)

        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fourier_unit(x)

        if self.enable_lfu:
            _, channels, height, width = x.shape
            split_no = 2
            split_s_h = height // split_no
            split_s_w = width // split_no
            xs = torch.cat(torch.split(x[:, : channels // 4], split_s_h, dim=2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=3), dim=1).contiguous()
            xs = self.local_fourier_unit(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()

        else:
            xs = 0

        output = self.conv2(x + output + xs)
        return output


class FFConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ratio_in: tuple[float, float],
        ratio_out: tuple[float, float],
        stride=1,
        padding=0,
        bias=False,
        enable_lfu=True,
    ):
        super().__init__()

        assert stride in (1, 2)

        in_cl = int(in_channels * ratio_in[0])
        in_cg = int(in_channels * ratio_in[1])
        out_cl = int(out_channels * ratio_out[0])
        out_cg = int(out_channels * ratio_out[1])

        self.ratio_lout = ratio_out[0]
        self.ratio_gout = ratio_out[1]

        kwargs = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, **kwargs)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, **kwargs)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, **kwargs)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride, enable_lfu)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_lout != 0:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)

        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class FFConvSet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ratio_in: tuple[float, float],
        ratio_out: tuple[float, float],
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        enable_lfu: bool = True,
    ):
        super().__init__()
        self.ffc = FFConv2d(
            in_channels, out_channels, kernel_size, ratio_in, ratio_out, stride, padding, bias, enable_lfu
        )
        self.out_cl = int(out_channels * ratio_out[0])
        self.out_cg = int(out_channels * ratio_out[1])

        if self.out_cl != 0:
            self.bn_local = nn.BatchNorm2d(self.out_cl)

        if self.out_cg != 0:
            self.bn_global = nn.BatchNorm2d(self.out_cg)

        # if stride == 1 and in_channels == out_channels:
        #     self.shortcut_l = nn.Identity()
        #     self.shortcut_g = nn.Identity()

        # else:
        #     self.shortcut_l = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, 1, stride),
        #         nn.BatchNorm2d(self.out_cl),
        #     )
        #     self.shortcut_g = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, 1, stride),
        #         nn.BatchNorm2d(self.out_cg),
        #     )

    def forward(self, x):
        x_local, x_global = self.ffc(x)
        if self.out_cl != 0:
            x_local = F.relu(self.bn_local(x_local))

        if self.out_cg != 0:
            x_global = F.relu(self.bn_global(x_global))

        return x_local, x_global


class FFCBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        hid_channels,
        out_channels,
        ratio_in,
        ratio_out,
        stride=1,
        enable_lfu=True,
    ):
        super().__init__()
        self.conv1 = FFConvSet(
            in_channels,
            hid_channels,
            kernel_size=1,
            ratio_in=ratio_in,
            ratio_out=ratio_out,
            stride=1,
            padding=0,
            bias=False,
            enable_lfu=enable_lfu,
        )
        self.conv2 = FFConvSet(
            hid_channels,
            hid_channels,
            kernel_size=3,
            ratio_in=ratio_out,
            ratio_out=ratio_out,
            stride=stride,
            padding=1,
            bias=False,
            enable_lfu=enable_lfu,
        )
        self.conv3 = FFConvSet(
            hid_channels,
            out_channels,
            kernel_size=1,
            ratio_in=ratio_out,
            ratio_out=ratio_out,
            enable_lfu=enable_lfu,
        )
        self.relu_l = nn.Identity() if ratio_out[0] == 0 else nn.ReLU()
        self.relu_g = nn.Identity() if ratio_out[1] == 0 else nn.ReLU()
        self.stride = stride

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = FFConvSet(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                ratio_in=ratio_in,
                ratio_out=ratio_out,
                enable_lfu=enable_lfu,
            )

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = self.shortcut(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x_l, x_g = self.conv3(x)

        x_l = self.relu_l(x_l + id_l)
        x_g = self.relu_g(x_g + id_g)

        return x_l, x_g


if __name__ == "__main__":
    pass