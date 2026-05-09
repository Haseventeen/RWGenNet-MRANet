import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np

from . import discriminator
from dataclasses import dataclass

###############################################################################
# Helper Functions
###############################################################################


class CosineWarmupScheduler(lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        # lr_factor = self.warmup_steps ** 0.5 * min(epoch ** (-0.5), epoch * self.warmup_steps ** (-1.5))

        return lr_factor
    

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm1d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm1d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=3)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    elif opt.lr_policy == 'cosinewarm':
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup=2,       # 预热步数（steps）或轮数（epochs）
            max_iters=20    # 总迭代次数（根据step或epoch模式决定）
        )
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'wavenet':
        net = WaveNetGenerator()
    elif netG == 'RWGenNet':
        net = RWGenNet()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        #net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
        net = NLayerDiscriminator1D(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'dilated':   # 使用膨胀卷积和Mish激活函数的判别器
        net = discriminator.NLayerDiscriminator1D(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer, use_dilated_conv=True)
    elif netD == 'multi_scale':  # 使用多尺度判别器——内含改进后的NLayerDiscriminator1D的调用
        net = discriminator.MultiScaleDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer)


    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=0.9, target_fake_label=0.1):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    # def __call__(self, prediction, target_is_real):
    #     """Calculate loss given Discriminator's output and grount truth labels.

    #     Parameters:
    #         prediction (tensor) - - tpyically the prediction output from a discriminator
    #         target_is_real (bool) - - if the ground truth label is for real images or fake images

    #     Returns:
    #         the calculated loss.
    #     """
    #     if self.gan_mode in ['lsgan', 'vanilla']:
    #         target_tensor = self.get_target_tensor(prediction, target_is_real)
    #         loss = self.loss(prediction, target_tensor)
    #     elif self.gan_mode == 'wgangp':
    #         if target_is_real:
    #             loss = -prediction.mean()
    #         else:
    #             loss = prediction.mean()
    #     return loss

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            prediction (tensor or list of tensors) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        # <<<<<<<<<<<<<<<<<<<< START OF MODIFICATION >>>>>>>>>>>>>>>>>>>>
        
        # If the prediction is a list (from multi-scale discriminator),
        # calculate loss for each item and sum them up.
        if isinstance(prediction, list):
            loss = 0
            for pred_i in prediction:
                loss += self.__call__(pred_i, target_is_real)
            return loss

        # If the prediction is a single tensor, calculate loss as before.
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        
        # <<<<<<<<<<<<<<<<<<<<  END OF MODIFICATION  >>>>>>>>>>>>>>>>>>>>
        return loss



def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm1d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm1d
        else:
            use_bias = norm_layer == nn.InstanceNorm1d

        model = [nn.ReflectionPad1d(3),
                 nn.Conv1d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv1d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            mult = 2 ** n_downsampling
            for i in range(n_blocks):  # add ResNet blocks

                model += [
                    ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                use_bias=use_bias)]

            for i in range(n_downsampling):  # add upsampling layers
                mult = 2 ** (n_downsampling - i)
                model += [nn.ConvTranspose1d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            model += [nn.ReflectionPad1d(3)]
            model += [nn.Conv1d(ngf, output_nc, kernel_size=7, padding=0)]
            model += [nn.Tanh()]

            self.model = nn.Sequential(*model)

        def forward(self, input):
            """Standard forward"""
            return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad1d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad1d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv1d(dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad1d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad1d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv1d(dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm1d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        # """Standard forward"""
        # return self.model(input)
        """Forward pass with automatic length alignment"""
        orig_len = input.shape[-1]  # 原始输入长度
        output = self.model(input)  # 正常前向传播
        
        # 动态对齐长度
        if output.size(-1) > orig_len:
            output = output[..., :orig_len]  # 裁剪超长部分
        elif output.size(-1) < orig_len:
            output = F.pad(output, (0, orig_len - output.size(-1)))  # 填充不足部分
        
        return output

# 根据输入长度是否为偶数动态调整 output_padding
def get_output_padding(input_length):
    return 1 if input_length % 2 == 0 else 0


# 生成器
class WaveNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, residual_channels=64, num_layers=11):
        super().__init__()
        self.start = nn.Conv1d(in_channels, residual_channels, kernel_size=1)
        self.res_blocks = nn.ModuleList()

        for i in range(num_layers):
            dilation = 2 ** i
            conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=3,
                             dilation=dilation, padding=dilation)
            self.res_blocks.append(nn.Sequential(
                conv,
                nn.ReLU(),
                nn.BatchNorm1d(residual_channels)
            ))

        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(residual_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        #x = x.unsqueeze(1)  # [B, 1, 512]
        x = self.start(x)
        for block in self.res_blocks:
            x = x + block(x)  # 残差结构
        x = self.output_layer(x)
        #return x.squeeze(1)  # [B, 512]
        return x  # [B, 512]
    
# 辅助模块: 论文中的 LayerNorm
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    

class TSSA_Attention_NonCausal(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # 对应 Algorithm 1, line 11: self.qkv [cite: 711]
        self.c_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias) 
        # 对应 Algorithm 1, line 13-14: self.to_out [cite: 713-714]
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.temp = nn.Parameter(torch.ones(config.n_head, 1))

        self.attend = nn.Softmax(dim=1) 

    def forward(self, x):
        # x 必须是 (B, L, C) -> (Batch, Length, Channels)
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # 1. 投影 w
        w = self.c_attn(x)
        
        # 2. 拆分多头
        w = w.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # 3. 计算成员概率 Pi (Π)
        w_normed = F.normalize(w, dim=-2) # 非因果: 归一化 (T, hs)
        tmp = torch.sum(w_normed**2, dim=-1) * self.temp # (B, nh, T)
        Pi = self.attend(tmp) # [cite: 112] (B, nh, T)

        # 4. 计算核心统计量 'dots'
        Pi_prob = (Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2) # (B, nh, 1, T)
        dots = torch.matmul(Pi_prob, w**2) # (B, nh, 1, hs)

        # 5. 计算衰减因子 attn
        attn = 1. / (1 + dots) # (B, nh, 1, hs)
        attn = self.attn_dropout(attn)

        # 6. 应用注意力 (TSSA 操作)
        y = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn) # (B, nh, T, hs)
        
        # 7. 合并多头 & 输出
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.resid_dropout(self.c_proj(y))
        return y

class RWGenNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, 
                 residual_channels=64, num_layers=11, n_heads=8):
        super().__init__()
        
        # 我们需要这个来初始化 TSSA_Attention_NonCausal
        @dataclass
        class TSSAConfig:
            n_embd: int = residual_channels
            n_head: int = n_heads
            dropout: float = 0.0 # 你的 WaveNet 里没有, 保持 0
            bias: bool = True
            # (block_size 不是必需的，因为我们非因果)
        
        self.config = TSSAConfig()
        
        # --- 2. 你的 WaveNet 卷积层 ---
        self.start = nn.Conv1d(in_channels, residual_channels, kernel_size=1)
        self.res_blocks = nn.ModuleList()

        for i in range(num_layers):
            dilation = 2 ** i
            conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=3,
                             dilation=dilation, padding=dilation)
            self.res_blocks.append(nn.Sequential(
                conv,
                nn.ReLU(),
                nn.BatchNorm1d(residual_channels)
            ))
            
        # --- 3. TSSA 模块 (放在 WaveNet 之后) ---
        self.tssa_ln = LayerNorm(self.config.n_embd, bias=self.config.bias)
        self.tssa_block = TSSA_Attention_NonCausal(self.config)
        self.gamma = nn.Parameter(torch.ones(self.config.n_embd) * 1e-5, requires_grad=True)

        # --- 4. 你的输出层 ---
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(residual_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # x 形状: [B, C, L] 
        
        x = self.start(x)
        
        # 1. 跑完你所有的 WaveNet 卷积块
        for block in self.res_blocks:
            x = x + block(x)  # 残差结构
            
        # 2. 准备 TSSA
        # TSSA (和 Transformer) 期望 (B, L, C)
        x_permuted = x.permute(0, 2, 1) # [B, L, C]
        
        # 3. 应用 TSSA 模块 (模仿你文件中的 Block 逻辑 [cite: 176])
        x_normed = self.tssa_ln(x_permuted)
        attn_out = self.tssa_block(x_normed)
        
        # 4. 应用残差连接
        # attn_out 已经是负的了，所以加法 = 减法
        # (x = x + gamma * attn_out)
        x_tssa = x_permuted + self.gamma * attn_out
        
        # 5. 转换回 (B, C, L)
        x = x_tssa.permute(0, 2, 1)
        
        # 6. 最后的输出层
        x = self.output_layer(x)

        return x



class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, 
                 norm_layer=nn.BatchNorm1d, use_dropout=False):
        super().__init__()
        
        self.outermost = outermost
        self.upconv = None
        use_bias = norm_layer == nn.InstanceNorm1d or (isinstance(norm_layer, functools.partial) 
                      and norm_layer.func == nn.InstanceNorm1d)
        
        if input_nc is None:
            input_nc = outer_nc

        # 下采样路径 ==============================================
        downconv = nn.Conv1d(
            in_channels=input_nc, 
            out_channels=inner_nc,  # 输出到内层通道数
            kernel_size=4,
            stride=2,
            padding=1,
            bias=use_bias
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        
        # 上采样路径 ==============================================
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            self.upconv = nn.ConvTranspose1d(
                in_channels=inner_nc * 2, 
                out_channels=outer_nc,  # 输出到最终通道数
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=1  # 新增关键参数
            )
            down = [downconv]
            up = [uprelu, self.upconv, nn.Tanh()]
            model = down + [submodule] + up
            
        elif innermost:
            self.upconv = nn.ConvTranspose1d(
                in_channels=inner_nc, 
                out_channels=outer_nc,  # 输出到外层通道数
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=1,  # 新增关键参数
                bias=use_bias
            )
            down = [downrelu, downconv]
            up = [uprelu, self.upconv, upnorm]
            model = down + up
            
        else:
            self.upconv = nn.ConvTranspose1d(
                in_channels=inner_nc * 2, 
                out_channels=outer_nc,  # 输出到外层通道数
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=1,  # 新增关键参数
                bias=use_bias
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, self.upconv, upnorm]
            
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        input_len = x.shape[-1]
        output_pad = get_output_padding(input_len)
        # 修改转置卷积层
        self.upconv.output_padding = output_pad
        if self.outermost:
            return self.model(x)
        else:
            # 获取原始长度
            orig_len = x.shape[-1]
            
            # 前向传播
            out = self.model(x)
            
            # 动态裁剪/填充以匹配原始长度
            if out.shape[-1] != orig_len:
                pad = orig_len - out.shape[-1]
                if pad > 0:
                    out = F.pad(out, (0, pad))  # 末尾填充
                else:
                    out = out[..., :orig_len]   # 裁剪超长部分
            return torch.cat([x, out], 1)  # 通道维度拼接



class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm1d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm1d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm1d
        else:
            use_bias = norm_layer == nn.InstanceNorm1d

        kw = 4
        padw = 1
        sequence = [nn.Conv1d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv1d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class PixelDiscriminator(nn.Module):
    """1x1 PatchGAN判别器（适用于一维信号）"""
    
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm1d):
        super().__init__()
        
        # 判断是否使用偏置项（InstanceNorm不需要）
        use_bias = norm_layer == nn.InstanceNorm1d or (
            isinstance(norm_layer, functools.partial) and 
            norm_layer.func == nn.InstanceNorm1d
        )

        # 网络层定义
        self.net = nn.Sequential(
            # 输入层: (batch, input_nc, seq_len) → (batch, ndf, seq_len)
            nn.Conv1d(in_channels=input_nc, out_channels=ndf, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 中间层: (batch, ndf, seq_len) → (batch, ndf*2, seq_len)
            nn.Conv1d(in_channels=ndf, out_channels=ndf*2, kernel_size=1, bias=use_bias),
            norm_layer(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出层: (batch, ndf*2, seq_len) → (batch, 1, seq_len)
            nn.Conv1d(in_channels=ndf*2, out_channels=1, kernel_size=1, bias=use_bias)
        )

    def forward(self, x):
        return self.net(x)  # 输出形状: (batch, 1, seq_len)


class NLayerDiscriminator1D(nn.Module):
    """1D PatchGAN判别器，用于一维信号"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm1d):
        super().__init__()
        
        use_bias = norm_layer == nn.InstanceNorm1d or (
            isinstance(norm_layer, functools.partial) and norm_layer.func == nn.InstanceNorm1d
        )

        kw = 4  # 卷积核大小
        padw = 1  # padding
        sequence = [
            nn.Conv1d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # 最后一层
        sequence += [
            nn.Conv1d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
