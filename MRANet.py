import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os
from dataclasses import dataclass
from torch.utils.data import WeightedRandomSampler, DataLoader
import random

warnings.filterwarnings('ignore')

MODE_NAME = ''
#'no_ecg', 'no_scg', 'no_resp', 'only_ecg', 'only_scg', 'only_resp' 消融实验选择
save_dir = f""
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(3407)  



def get_target_boost_sampler(dataset):
    """
    创建一个采样器，自动平衡类别，并额外对指定类别进行"强行加练"
    
    Args:
        dataset: 你的训练数据集
    """
    # 1. 获取所有样本的标签 (假设 label 是 One-hot 或 索引)
    # 注意: 这里需要根据你的 dataset 实际结构调整。
    # 如果 dataset.labels 是 tensor，直接用；如果是 list，转 tensor。
    try:
        # 尝试直接读取所有标签 (针对 TensorDataset 或 自定义 Dataset)
        if hasattr(dataset, 'labels'):
            all_labels = dataset.labels
        elif hasattr(dataset, 'tensors'):
            all_labels = dataset.tensors[-1] # 假设标签在最后
        else:
            # 如果都没有，只能遍历一遍 (比较慢，但通用)
            print("正在遍历数据集获取标签以计算采样权重...")
            all_labels = [sample['label'] for sample in dataset]
            all_labels = torch.stack(all_labels)
            
        # 如果是 One-hot (N, 6)，转为索引 (N,)
        if all_labels.ndim > 1 and all_labels.shape[1] > 1:
            target_indices = torch.argmax(all_labels, dim=1)
        else:
            target_indices = all_labels.long()
            
        target_indices = target_indices.cpu().numpy()
        
    except Exception as e:
        print(f"无法自动获取标签: {e}")
        return None

    # 2. 计算每个类别的样本数
    class_counts = np.bincount(target_indices, minlength=6)
    print(f"  - 各类别样本数: {class_counts}")
    
    # 3. 计算基础权重 (倒数频次: 样本越少，权重越大)
    # 加上 1e-6 防止除以 0
    class_weights = 1. / (class_counts + 1e-6)
    
    
    # 4. 给每个样本分配权重
    samples_weights = class_weights[target_indices]
    samples_weights = torch.from_numpy(samples_weights).double()
    
    # 5. 创建采样器
    # num_samples: 建议设为 len(dataset)，这样 epoch 长度不变
    # replacement=True: 允许重复抽样 (必须为True才能实现"多训练几次")
    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights), 
        replacement=True
    )
    
    return sampler

class MultiModalECGDataset(Dataset):
    """多模态心电信号数据集"""

    def __init__(self, data_dict):
        super().__init__()
        self.ecg = data_dict['ECG'].float()
        self.scg = data_dict['SCG'].float()
        self.resp = data_dict['Generated_RESP'].float()
        self.labels = data_dict['label'].float()

    def __len__(self):
        return len(self.ecg)

    def __getitem__(self, idx):
        return {
            'ecg': self.ecg[idx],
            'scg': self.scg[idx],
            'resp': self.resp[idx],
            'label': self.labels[idx]
        }


class BasicBlock1D(nn.Module):
    """1D ResNet 基础块"""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """1D ResNet 用于单模态信号处理"""

    def __init__(self, block, layers, input_channels=1, num_classes=6):
        super().__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet 层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 自适应池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # 特征提取器，不包含最后的分类层
        self.feature_dim = 512 * block.expansion

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

class TSSA_Attention_NonCausal(nn.Module):
    
    def __init__(self, config):
            super().__init__()
            assert config.n_embd % config.n_head == 0
            
            self.c_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias) 
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            
            self.attn_dropout = nn.Dropout(config.dropout)
            self.resid_dropout = nn.Dropout(config.dropout)
            
            self.n_head = config.n_head
            self.n_embd = config.n_embd
            
            self.temp = nn.Parameter(torch.ones(config.n_head, 1))
            self.attend = nn.Softmax(dim=1) 

    def forward(self, x):
        # x: (Batch, 3, 512) -> (B, L, C)
        B, T, C = x.size() 

        # 1. 投影
        w = self.c_attn(x)
        
        # 2. 拆分多头
        w = w.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # 3. 计算成员概率 Pi
        w_normed = F.normalize(w, dim=-2) 
        tmp = torch.sum(w_normed**2, dim=-1) * self.temp # (B, nh, T)
        Pi = self.attend(tmp) # (B, nh, T)

        # 4. 计算核心统计量 'dots'
        Pi_prob = (Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2) # (B, nh, 1, T)
        dots = torch.matmul(Pi_prob, w**2) # (B, nh, 1, hs)

        # 5. 计算衰减因子 attn
        attn = 1. / (1 + dots) # (B, nh, 1, hs)
        attn = self.attn_dropout(attn)

        # 6. 应用注意力 (TSSA 操作) -> 返回负注意力
        y = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn) # (B, nh, T, hs)
        
        # 7. 合并多头 & 输出
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.resid_dropout(self.c_proj(y))
        
        # === 修改点：提取权重用于可视化 ===
        # Pi 的形状是 (Batch, n_head, 3)
        # 我们在 dim=1 (head维度) 上取平均，得到 (Batch, 3)
        attention_weights = Pi.mean(dim=1) 

        return y, attention_weights  # 返回两个值

    
class MultiModalResNet(nn.Module):
    """多模态 ResNet 融合模型"""

    def __init__(self, num_classes=6):
        super().__init__()

        # 三个独立的 ResNet 编码器（移除最后的分类层）
        self.ecg_encoder = ResNet1D(BasicBlock1D, [2, 2, 2, 2], input_channels=1)
        self.scg_encoder = ResNet1D(BasicBlock1D, [2, 2, 2, 2], input_channels=1)
        self.resp_encoder = ResNet1D(BasicBlock1D, [2, 2, 2, 2], input_channels=1)

        # 2. TSSA 注意力模块 (替换了原来的线性注意力)
        # 配置: 输入512维, 8个头 (512/8=64), dropout 0.1
        @dataclass
        class TSSAConfig:
            n_embd: int 
            n_head: int 
            dropout: float = 0.0 # 你的 WaveNet 里没有, 保持 0
            bias: bool = True
            # (block_size 不是必需的，因为我们非因果)
        tssa_config = TSSAConfig(n_embd=512, n_head=8, bias=False, dropout=0.1)
        self.tssa_attention = TSSA_Attention_NonCausal(tssa_config)

        # 模态注意力机制（输入应该是512维特征）
        self.ecg_attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        self.scg_attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        self.resp_attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(512 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, ecg, scg, resp, ablation_mode=None):
        """
        ablation_mode: str, 可选值 'no_ecg', 'no_scg', 'no_resp', 'only_ecg', 'only_scg', 'only_resp'
        """
        
        # --- 核心修改：在源头屏蔽数据 ---
        if ablation_mode == 'no_ecg' or ablation_mode == 'only_scg' or ablation_mode == 'only_resp':
            ecg = torch.zeros_like(ecg) # 把 ECG 变成全 0
            
        if ablation_mode == 'no_scg' or ablation_mode == 'only_ecg' or ablation_mode == 'only_resp':
            scg = torch.zeros_like(scg) # 把 SCG 变成全 0
            
        if ablation_mode == 'no_resp' or ablation_mode == 'only_ecg' or ablation_mode == 'only_scg':
            resp = torch.zeros_like(resp) # 把 RESP 变成全 0

        # 1. 编码各个模态
        ecg_features = self.ecg_encoder(ecg.unsqueeze(1)) 
        scg_features = self.scg_encoder(scg.unsqueeze(1)) 
        resp_features = self.resp_encoder(resp.unsqueeze(1))

        # --- 2. 构建模态序列 ---
        # 将三个向量堆叠，把模态视为序列的时间步 (Length=3)
        # 形状变化: [batch, 512] x 3 -> [batch, 3, 512]
        multi_modal_seq = torch.stack([ecg_features, scg_features, resp_features], dim=1)

        # --- 3. TSSA 注意力交互 ---
        # TSSA 会计算模态之间的关系并更新特征
        # 输出形状: [batch, 3, 512]
        refined_seq, tssa_weights = self.tssa_attention(multi_modal_seq)

        # --- 4. 残差连接 (可选但推荐) ---
        # 你的 TSSA 实现最后返回的是负注意力处理后的 y
        # 通常 Transformer Block 会做 x + Attention(x)
        # 这里我们将 TSSA 的输出与原始特征相加，防止信息丢失
        refined_seq = multi_modal_seq + refined_seq

        # --- 5. 展平与融合 ---
        # [batch, 3, 512] -> [batch, 1536]
        flattened_features = refined_seq.view(refined_seq.size(0), -1)

        # 通过 MLP 进行深度融合
        # [batch, 1536] -> [batch, 512]
        fused_features = self.fusion(flattened_features)

        # --- 6. 分类 ---
        output = self.classifier(fused_features)

        # 注意: TSSA 内部封装了注意力权重，外部无法直接获取简单的权重向量用于可视化
        # 所以这里只返回 output
        return output,fused_features



class ECGClassifier:
    """ECG分类训练器"""

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        # 1. 定义权重 (根据你的数据集统计计算得出)
        # 对应顺序: [Label 1, Label 2, Label 3, Label 4, Label 5, Label 6]
        # 注意：Label 5 (心肌缺血) 的权重给到了 17.0，这是重点
        pos_weights = torch.tensor([12.5, 12.5, 8.0, 6.7, 17.0, 0.8])

        # 2. 必须把权重移动到 GPU 上 (如果你的模型在 GPU 上)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pos_weights = pos_weights.to(device)

        # 3. 初始化 Loss
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights) # 多标签分类使用BCE 
        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, T_max=50, eta_min=1e-6
        # )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            verbose=True,  # 触发时打印日志，方便你看到
            min_lr=1e-6    # 学习率下限
        )
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc='Training', leave=False)
        for batch in pbar:
            ecg = batch['ecg'].to(self.device)
            scg = batch['scg'].to(self.device)
            resp = batch['resp'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            outputs, _ = self.model(ecg, scg, resp, ablation_mode=MODE_NAME)
            loss = self.criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            # 计算预测
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu().detach())
            all_labels.append(labels.cpu().detach())

            pbar.set_postfix({'loss': loss.item()})

        # 计算指标
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        # 对于多标签分类，需要指定多标签评估方式
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        avg_loss = total_loss / len(train_loader)

        return avg_loss, accuracy, f1

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        attention_weights = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation', leave=False)
            for batch in pbar:
                ecg = batch['ecg'].to(self.device)
                scg = batch['scg'].to(self.device)
                resp = batch['resp'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs, att_weights = self.model(ecg, scg, resp, ablation_mode=MODE_NAME)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                preds = torch.sigmoid(outputs) > 0.5
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                attention_weights.append(att_weights.cpu())

                pbar.set_postfix({'loss': loss.item()})

        # 计算指标
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        attention_weights = torch.cat(attention_weights, dim=0)

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        avg_loss = total_loss / len(val_loader)

        return avg_loss, accuracy, f1, attention_weights

    def train(self, train_loader, val_loader, epochs=50):
        print(f"开始训练，使用设备: {self.device}")
        print(f"训练样本: {len(train_loader.dataset)}, 验证样本: {len(val_loader.dataset)}")
        print("=" * 60)

        best_val_f1 = 0
        best_model_state = None

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # 训练
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)

            # 验证
            val_loss, val_acc, val_f1, att_weights = self.validate(val_loader)

            # 更新学习率
            self.scheduler.step(val_loss)

            # 保存历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)

            # 计算平均注意力权重
            avg_att_weights = att_weights.mean(dim=0).numpy()

            # 打印结果
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(
                f"注意力权重 - ECG: {avg_att_weights[0]:.3f}, SCG: {avg_att_weights[1]:.3f}, RESP: {avg_att_weights[2]:.3f}")

            # ==========================================
            # 1. 保存当前轮次模型 (这是你想要的：全部保存)
            # ==========================================
            # 文件名带上 epoch，例如: checkpoints/model_epoch_1.pth
            # 确保文件夹存在 (如果不存在会自动创建，如果存在则什么都不做)
            # exist_ok=True 很重要，防止文件夹已存在时报错
            os.makedirs(save_dir, exist_ok=True)
            current_save_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss, # 建议也把 loss 存进去
                'history': self.history
            }, current_save_path)
            
            print(f"💾 已保存第 {epoch} 轮模型至: {current_save_path}")

            # 保存最佳模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = self.model.state_dict().copy()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': best_val_f1,
                    'history': self.history
                }, 'best_model.pth')
                print(f"✅ 保存最佳模型，F1: {best_val_f1:.4f}")

        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.history

    def plot_history(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # F1 Score
        axes[1, 0].plot(self.history['train_f1'], label='Train F1')
        axes[1, 0].plot(self.history['val_f1'], label='Val F1')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 注意力权重变化（如果有的话）
        axes[1, 1].text(0.5, 0.5, 'Finish Training\nCheck Saved File',
                        ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Training State')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()


def prepare_data(train_path="", 
                 test_path="", 
                 batch_size=32):
    """
    直接读取独立的训练集和测试集文件，并创建 DataLoader
    """
    
    # 1. 检查文件是否存在
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"找不到训练集文件: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到测试集文件: {test_path}")

    print(f"正在加载数据集...")
    print(f"  - 训练集路径: {train_path}")
    print(f"  - 测试集路径: {test_path}")

    # 2. 分别加载 .pt 文件
    train_data_dict = torch.load(train_path)
    test_data_dict = torch.load(test_path)

    # 3. 分别创建 Dataset 实例
    # 假设 MultiModalECGDataset 是你定义好的类，直接传入对应的字典
    train_dataset = MultiModalECGDataset(train_data_dict)
    test_dataset = MultiModalECGDataset(test_data_dict)

    # === 新增: 创建增强采样器 ===
    # 针对 Label 2 (心衰) 和 Label 4 (心律失常) 额外加权 2 倍
    print("正在创建增强采样器 (重点突破心衰和心律失常)...")
    train_sampler = get_target_boost_sampler(
        train_dataset, 
        boost_classes=[0,3], 
        boost_factor=2
    )
    
    # === 修改 Train Loader ===
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        # ⚠️ 注意: 使用 sampler 时，shuffle 必须为 False！
        shuffle=False,       
        sampler=train_sampler, 
        num_workers=0, 
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,       # 测试/验证集通常不需要打乱
        num_workers=0, 
        pin_memory=True,
        drop_last=True
    )

    # 5. 打印统计信息
    print("-" * 30)
    print(f"数据准备完成:")
    print(f"  训练集样本数: {len(train_dataset)}")
    print(f"  测试集样本数: {len(test_dataset)}")
    print("-" * 30)

    return train_loader, test_loader


def print_model_summary(model):
    """打印模型摘要"""
    print("=" * 60)
    print("模型架构摘要:")
    print("=" * 60)

    total_params = 0
    trainable_params = 0

    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        num_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

        print(f"{name}:")
        print(f"  参数数量: {num_params:,}")
        print(f"  可训练参数: {num_trainable:,}")

        total_params += num_params
        trainable_params += num_trainable

    print("=" * 60)
    print(f"总参数数量: {total_params:,}")
    print(f"总可训练参数: {trainable_params:,}")
    print("=" * 60)


def main():
    """主函数"""

    # 1. 准备数据
    print("🚀 开始多模态 ECG 分类任务")
    train_loader, val_loader = prepare_data()

    # 2. 创建模型
    print("\n📦 创建多模态 ResNet 模型...")
    model = MultiModalResNet(num_classes=6)
    print_model_summary(model)

    # 3. 创建训练器
    classifier = ECGClassifier(model)

    # 4. 训练模型（减少epochs以便快速测试）
    print("\n🔥 开始训练模型...")
    history = classifier.train(train_loader, val_loader, epochs=35)

    # 5. 可视化训练历史
    print("\n📊 可视化训练结果...")
    classifier.plot_history()

    # 6. 最终评估
    print("\n📈 最终模型评估...")
    final_val_loss, final_val_acc, final_val_f1, final_att = classifier.validate(val_loader)
    final_att_avg = final_att.mean(dim=0).numpy()

    print("=" * 60)
    print("最终性能指标:")
    print(f"  验证损失: {final_val_loss:.4f}")
    print(f"  验证准确率: {final_val_acc:.4f}")
    print(f"  验证 F1 分数: {final_val_f1:.4f}")
    print(f"  最终注意力权重:")
    print(f"    ECG: {final_att_avg[0]:.3f}")
    print(f"    SCG: {final_att_avg[1]:.3f}")
    print(f"    RESP: {final_att_avg[2]:.3f}")
    print("=" * 60)

    # 7. 保存完整模型
    torch.save({
        'model_state_dict': classifier.model.state_dict(),
        'model_architecture': model.__class__.__name__,
        'num_classes': 6,
        'performance': {
            'val_loss': final_val_loss,
            'val_acc': final_val_acc,
            'val_f1': final_val_f1,
        },
        'attention_weights': final_att_avg.tolist()
    }, 'ecg_classifier_final.pth')

    # 8. 测试预测
    print("\n🧪 测试预测功能...")
    test_preds, test_probs = [], []
    classifier.model.eval()
    with torch.no_grad():
        for batch in val_loader:
            ecg = batch['ecg'].to(classifier.device)
            scg = batch['scg'].to(classifier.device)
            resp = batch['resp'].to(classifier.device)

            outputs, _ = classifier.model(ecg, scg, resp)
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5

            test_preds.append(preds.cpu())
            test_probs.append(probs.cpu())

    test_preds = torch.cat(test_preds, dim=0)
    test_probs = torch.cat(test_probs, dim=0)

    print(f"预测结果形状: {test_preds.shape}")
    print(f"前5个样本的预测概率:")
    for i in range(min(5, len(test_probs))):
        print(f"  样本{i + 1}: {test_probs[i].numpy().round(3)}")

    print("\n✅ 模型训练完成并已保存!")
    print("📁 保存的文件:")
    print("  - best_model.pth (最佳检查点)")
    print("  - ecg_classifier_final.pth (最终模型)")
    print("  - training_history.png (训练历史图表)")



if __name__ == "__main__":
    main()