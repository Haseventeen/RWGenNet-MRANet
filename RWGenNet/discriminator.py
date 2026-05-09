import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

# Mish Activation Function
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# 修改后的NLayerDiscriminator1D
class NLayerDiscriminator1D(nn.Module):
    """1D PatchGAN判别器，使用膨胀卷积和Mish激活函数"""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm1d, use_dilated_conv=False):
        super().__init__()

        use_bias = norm_layer == nn.InstanceNorm1d or (
                isinstance(norm_layer, functools.partial) and norm_layer.func == nn.InstanceNorm1d
        )

        kw = 4  # 卷积核大小
        padw = 1  # padding
        sequence = [
            nn.Conv1d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            Mish()  # 使用Mish激活函数
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if use_dilated_conv:
                sequence += [
                    nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=kw, stride=1, padding=padw, dilation=2, bias=use_bias),  # 膨胀卷积
                    norm_layer(ndf * nf_mult),
                    Mish()  # 使用Mish激活函数
                ]
            else:
                sequence += [
                    nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    Mish()  # 使用Mish激活函数
                ]

        # 最后一层
        sequence += [
            nn.Conv1d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

# 定义一个多尺度判别器（Multi-scale Discriminator）
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.scale1 = NLayerDiscriminator1D(input_nc, ndf, n_layers, norm_layer)
        self.scale2 = NLayerDiscriminator1D(input_nc, ndf*2, n_layers, norm_layer)
        self.scale3 = NLayerDiscriminator1D(input_nc, ndf*4, n_layers, norm_layer)

    def forward(self, x):
        return [self.scale1(x), self.scale2(x), self.scale3(x)]

# 测试
if __name__ == '__main__':
    input_nc = 1
    ndf = 64
    n_layers = 3
    batch_size = 8
    seq_len = 2048

    # 测试膨胀卷积和Mish激活函数
    discriminator = NLayerDiscriminator1D(input_nc, ndf, n_layers, norm_layer=nn.BatchNorm1d, use_dilated_conv=True)
    print(discriminator)

    # 生成随机测试数据
    test_input = torch.randn(batch_size, input_nc, seq_len)
    output = discriminator(test_input)
    print(f"输出形状: {output.shape}")

    # 测试多尺度判别器
    multi_scale_discriminator = MultiScaleDiscriminator(input_nc, ndf, n_layers, norm_layer=nn.BatchNorm1d)
    multi_scale_output = multi_scale_discriminator(test_input)
    print(f"多尺度判别器输出形状: {[out.shape for out in multi_scale_output]}")
