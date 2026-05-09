import torch
from models_2.pix2pix import Pix2PixModel # 假设已经按照之前的建议修改为1D版本
import time
from torch.utils.data import Dataset,DataLoader
import os
from tqdm import tqdm
from scipy.signal import correlate
from torch import optim
from datetime import datetime
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(3407)  # 42是常用的，你可以随便换


class TestOptions:
    # 基础配置
    isG_Perc = True    # 是否启动感知损失。
    #isG_Perc = False    # 是否启动感知损失。
    isTrain = True                  # 是否为训练模式
    name = "ECG2ABP_Experiment_CAPNO_01_wavenet"     # 实验名称（用于保存结果）
    checkpoints_dir = "./checkpoints"  # 模型保存路径
    gpu_ids = [0]                   # 使用的GPU ID（[]表示使用CPU）
    
    # 数据相关参数
    dataroot = "./datasets/ecg_abp" # 数据集根目录
    phase = "train"                 # 当前阶段（train/test/val）
    serial_batches = False          # 是否按顺序读取数据
    num_threads = 1                 # 数据加载线程数
    batch_size = 8                 # 批大小
    seq_length = 2048               # 序列长度
    input_nc = 1                    # 输入通道数（ECG信号）
    output_nc = 1                   # 输出通道数（ABP信号）
    direction = 'AtoB'              # 数据流向（AtoB/BtoA）
    preprocess = "resize_and_crop"  # 预处理方式
    no_flip = False                 # 是否禁止随机翻转
    norm = "batch"          # 归一化方法 [batch | instance | none]
    
    # 网络结构参数
    netG = 'wavenet'               # 生成器架构 [unet_256 | resnet_9blocks | wavenet]
    # netD = 'pixel'                  # 判别器架构 [pixel | n_layers]
    netD = 'multi_scale'                  # 判别器架构 [pixel | n_layers | dilated | multi_scale]
    ngf = 64                        # 生成器基础通道数
    ndf = 64                        # 判别器基础通道数
    n_layers_D = 3                  # 判别器卷积层数
    dropout_rate = 0.2              # Dropout概率
    init_type = 'xavier'            # 权重初始化方法 [normal | xavier | kaiming]
    init_gain = 0.02                # 初始化缩放系数
    use_condition = False
    
    # 优化器参数
    lr = 0.0005                     # 初始学习率
    beta1 = 0.5                     # Adam优化器参数1
    beta2 = 0.999                   # Adam优化器参数2
    lr_policy = 'cosine'            # 学习率策略 [linear | cosine | plateau | cosinewarm]
    lr_decay_iters = 30             # 学习率衰减间隔epoch数
    weight_decay = 0.0001           # 权重衰减系数
    
    # 训练参数
    n_epochs = 50                  # 总训练epoch数
    n_epochs_decay = 50            # 学习率衰减开始epoch
    epoch_count = 1                 # 起始epoch计数
    continue_train = False          # 是否继续训练
    load_iter = 0                   # 加载的迭代次数（0表示最新）
    lambda_L1 = 10               # L1损失权重
    gan_mode = 'lsgan'              # GAN损失类型 [lsgan | vanilla | wgangp]
    pool_size = 50                  # 图像缓冲区大小
    save_epoch_freq = 5             # 模型保存频率（每N个epoch保存一次）
    print_freq = 2000                # 训练日志打印频率（每N个iter打印一次）
    no_dropout= True
    pretrain_epochs = 10   # 前 10 个 epoch 只做 L1
    
    # 高级参数
    fp16 = False                    # 是否使用混合精度训练
    amp_level = 'O1'               # 混合精度模式
    grad_clip = 5.0                 # 梯度裁剪阈值
    spectral_norm = False           # 是否使用谱归一化
    num_test = 50                   # 测试时运行的样本数
    
    # 可视化参数
    display_id = 1                  # Visdom显示窗口ID
    display_winsize = 256           # 显示图像尺寸
    display_port = 8097             # Visdom端口号
    update_html_freq = 1000         # HTML更新频率
    no_html = False                 # 是否禁止保存HTML结果
    
    # 验证参数
    eval_freq = 1                   # 验证频率（每N个epoch验证一次）
    eval_metric = 'MAE'             # 验证指标 [MAE | MSE | SSIM]
    
    # 数据增强参数
    noise_std = 0.02                # 高斯噪声标准差
    scale_range = (0.9, 1.1)       # 随机缩放范围
    shift_range = (-5, 5)          # 随机平移范围（单位：采样点）
    
    # 调试参数
    debug = False                   # 调试模式（减少数据量）
    verbose = False                 # 详细输出模式
    profile = False                 # 性能分析模式

    # WGAN-GP参数
    lambda_gp = 10.0               # 梯度惩罚系数
    n_critic = 5                   # 判别器更新频率（生成器每更新1次判别器更新n次）

    # 自定义参数（根据具体任务添加）
    ecg_filter = True              # 是否应用ECG信号滤波
    abp_normalize = True           # 是否对ABP信号进行标准化


model=None
opt=None

# 生成测试数据
def generate_1d_batch(batch_size, seq_length):
    return {
        'A': torch.randn(batch_size, 1, seq_length),  # 输入信号 (e.g., PPG)
        'B': torch.randn(batch_size, 1, seq_length)   # 目标信号 (e.g., ABP)
    }

# 测试前向传播
def test_forward_pass():
    # 生成模拟数据
    test_data = generate_1d_batch(opt.batch_size, opt.seq_length)
    
    # 设置模型输入
    model.set_input(test_data)
    
    # 执行前向传播
    model.forward()
    
    # 验证输出维度
    assert model.fake_B.shape == (opt.batch_size, opt.output_nc, opt.seq_length), \
        f"生成器输出形状错误，期望{(opt.batch_size, opt.output_nc, opt.seq_length)}，实际{model.fake_B.shape}"
    
    # 验证判别器输入
    if model.isTrain:
        real_AB = torch.cat([model.real_A, model.real_B], dim=1)
        fake_AB = torch.cat([model.real_A, model.fake_B.detach()], dim=1)
        assert real_AB.shape == (opt.batch_size, opt.input_nc+opt.output_nc, opt.seq_length)
        assert fake_AB.shape == (opt.batch_size, opt.input_nc+opt.output_nc, opt.seq_length)
        print("判别器输入维度验证通过")

    print("前向传播测试成功！输出形状:", model.fake_B.shape)

# 测试反向传播
def test_backward():
    model.optimize_parameters()
    assert not torch.isnan(model.loss_G), "生成器损失出现NaN"
    assert not torch.isnan(model.loss_D), "判别器损失出现NaN"
    print(f"反向传播测试通过！损失值 G: {model.loss_G:.4f} D: {model.loss_D:.4f}")


# --- 2. 把 CombinedTimeSpecDataset 改成输出 A/B 键 ---
class Pix2PixDataset(Dataset):
    def __init__(self, pt_file_path): # 可能还有其他参数，比如 transform
        print(f"加载数据集: {pt_file_path}")
        try:
            # 1. 加载包含字典的 .pt 文件
            data_dict = torch.load(pt_file_path)
            
            # 2. 使用字符串键提取数据并存为属性
            self.x_data = data_dict['ecg'] 
            self.y_data = data_dict['breath']
            # self.source_info = data_dict['source_info'] # 可选

            # 检查数据是否加载成功且非空
            if self.x_data is None or self.y_data is None:
                 raise ValueError("加载的数据中 'x' 或 'y' 为空。")
            if len(self.x_data) != len(self.y_data):
                 raise ValueError("加载的 'x' 和 'y' 数据长度不匹配。")
            
            # # --- 新增：检查是否存在 NaN 或 Inf 值 ---
            # print("  正在检查数据中是否存在无效值 (NaN or Inf)...")
            # nan_found = False
            # if torch.isnan(self.x_data).any():
            #     print("  警告：在 'ecg' (x_data) 数据中发现了 NaN 值！")
            #     nan_found = True
            # if torch.isinf(self.x_data).any():
            #     print("  警告：在 'ecg' (x_data) 数据中发现了 Inf 值！")
            #     nan_found = True
                
            # if torch.isnan(self.y_data).any():
            #     print("  警告：在 'breath' (y_data) 数据中发现了 NaN 值！")
            #     nan_found = True
            # if torch.isinf(self.y_data).any():
            #     print("  警告：在 'breath' (y_data) 数据中发现了 Inf 值！")
            #     nan_found = True
            
            # if not nan_found:
            #     print("  数据检查完毕，未发现 NaN 或 Inf 值。")
            # # --- 新增结束 ---
                 
            # 3. 存储数据集长度
            self.length = len(self.x_data) 
            # print(f"数据集加载成功，包含 {self.length} 个样本。")
            # print(f"  x 数据形状: {self.x_data.shape}")
            # print(f"  y 数据形状: {self.y_data.shape}")
            
            # 4. !!! 关键：移除 __init__ 中任何 self.data[i] 形式的访问 !!!
            # 例如，之前错误的行 a, b = self.data[i] 必须删除或移到正确的地方

        except FileNotFoundError:
            print(f"错误：找不到数据集文件 {pt_file_path}")
            raise
        except KeyError as e:
            print(f"错误：加载的 .pt 文件字典中缺少键: {e}。请确保文件包含 'x' 和 'y' 键。")
            raise
        except Exception as e:
            print(f"加载或初始化数据集时发生未知错误: {e}")
            raise

    def __len__(self):
        # 返回数据集的样本总数
        return self.length

    def __getitem__(self, index):
        # 根据索引获取单个数据对
        x_sample = self.x_data[index]
        y_sample = self.y_data[index]

        # --- 在这里添加通道维度 ---
        # unsqueeze(0) 在第 0 维增加一个维度： (L,) -> (1, L)
        x_sample_reshaped = x_sample.unsqueeze(0) 
        y_sample_reshaped = y_sample.unsqueeze(0)
        # --- 添加维度结束 ---

        # (可选) 检查转换后的形状
        # if index == 0: # 只打印第一个样本的形状以供检查
        #     print(f"  __getitem__ reshaped x shape: {x_sample_reshaped.shape}")
        #     print(f"  __getitem__ reshaped y shape: {y_sample_reshaped.shape}")

        # (可选) 在这里进行其他数据预处理或转换 (transform)
        # ...

        # 返回包含 'A' 和 'B' 键以及【正确形状】数据的字典
        # 返回的数据形状将是 (1, L)，例如 (1, 2000)
        return {'A': x_sample_reshaped, 'B': y_sample_reshaped} 




# 设定设备（建议与训练时一致）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train(train_loader, test_loader, opt, epochs, steps_per_epoch=2,early_stop_patience=15,patience=3):
    best_test_loss = float('inf') # 初始化最低测试集损失为正无穷
    best_mae = float('inf')
    no_improve_count = 0 # 连续没有下降的 epoch 数

    # 初始化模型
    model = Pix2PixModel(opt)
    model.setup(opt)  # 初始化网络和优化器

    # 在这里确保目录存在
    os.makedirs(model.save_dir, exist_ok=True)
    
    # 训练循环
    total_steps = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.netG.train()
        if model.isTrain:
            model.netD.train()
        epoch_iter = 0
        
        step = 0
        for i,data in  enumerate(train_loader):
            # print(f"[DEBUG] Batch {i} type: {type(data)}")  # 应该是 dict
            # print(f"[DEBUG] Keys: {data.keys() if isinstance(data, dict) else 'not dict!'}")
            data['A'] = data['A'].to(device)
            data['B'] = data['B'].to(device)
            iter_start_time = time.time()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            # 数据预处理
            model.set_input(data)
            model.optimize_parameters(epoch)

            # 打印训练损失
            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                tcomp = (time.time() - iter_start_time) / opt.batch_size
                print(f'Epoch {epoch}, Step {total_steps}'
                      f' G_GAN: {losses["G_GAN"]:.3f}'
                      f' G_L1: {losses["G_L1"]:.3f}'
                      f' G_Perceptual: {losses.get("G_Perceptual", "无")}'
                      f' D_real: {losses["D_real"]:.3f}'
                      f' D_fake: {losses["D_fake"]:.3f}'
                      f' Time: {tcomp:.3f}s')

        # 验证阶段
        model.eval()
        val_losses = {'G_L1': 0.0, 'G_GAN': 0.0}
        mae_sum = 0.0
        n_samples = 0

        with torch.no_grad():
            for val_data in test_loader:
                model.set_input(val_data)
                model.forward()
                B = val_data['A'].size(0)
                # 原来的 loss 累加
                l1 = model.criterionL1(model.fake_B, model.real_B).item()
                gan = model.criterionGAN(
                    model.netD(torch.cat((model.real_A, model.fake_B),1)), True
                ).item()
                val_losses['G_L1']   += l1 * B
                val_losses['G_GAN']  += gan * B
                # 新增 MAE 累加
                mae_sum += l1 * B      # L1Loss 就是 MAE
                n_samples += B

        # 计算平均
        val_losses = {k: v / n_samples for k,v in val_losses.items()}
        avg_test_loss=val_losses['G_L1']+val_losses['G_GAN']
        mean_mae = mae_sum / n_samples

        print(f"[Validation] MAE: {mean_mae:.4f} | G_L1: {val_losses['G_L1']:.4f} | G_GAN: {val_losses['G_GAN']:.4f}")

        model.save_networks(epoch)
        # 如果当前测试集损失更低，则保存模型
        # 早停判断：如果测试损失在连续 early_stop_patience 个 epoch 内没有下降，则停止训练
        if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                no_improve_count = 0
                # 保存检查点
                model.save_networks(epoch)

                print(f"Model saved with lower test loss: {best_test_loss}")
        else:
                no_improve_count += 1
                print(f"No improvement for {no_improve_count} epoch(s).")
                if no_improve_count >= early_stop_patience:
                    print("Early stopping triggered!")
                    break

        # 每个epoch结束后更新学习率
        model.update_learning_rate()



if __name__ == '__main__':
    # print("开始维度兼容性测试...")
    # test_forward_pass()
    
    # if model.isTrain:
    #     print("\n开始训练流程测试...")
    #     test_backward()

    opt = TestOptions()

    train_loader = DataLoader(Pix2PixDataset('./data/train_CAPNO.pt'),
                          batch_size=opt.batch_size,
                          shuffle=not opt.serial_batches,
                          num_workers=opt.num_threads)

    test_loader = DataLoader(Pix2PixDataset('./data/test_CAPNO.pt'),
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.num_threads)

    train(train_loader,test_loader,opt, epochs=100)

    
    # # 打印网络结构
    # print("\n生成器结构:")
    # print(model.netG)
    
    # print("\n判别器结构:")
    # print(model.netD)