# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch import nn
from torch.cuda.amp import autocast
from typing import Union, Tuple, List
import os

# 导入 nnU-Net 原始组件
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn

# 导入你修改后的模型获取函数
from nnunetv2.nets.UMambaBot_3d import get_umamba_bot_3d_from_plans
from nnunetv2.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans

# --- 定义多任务 Loss 包装器 ---
class MultiTask_DC_and_CE_loss(nn.Module):
    def __init__(self, task_losses: nn.ModuleList, task_weights: List[float] = None):
        super(MultiTask_DC_and_CE_loss, self).__init__()
        self.task_losses = task_losses
        self.task_weights = task_weights if task_weights is not None else [1.0] * len(task_losses)

    def forward(self, net_outputs: List, target: Union[torch.Tensor, List]):
        total_loss = 0
        for i, loss_func in enumerate(self.task_losses):
            # 获取第 i 个任务的输出 (net_outputs 是一个列表，对应 6 个 Head 的输出)
            task_output = net_outputs[i]
            
            # 从 target 中切分出对应的通道。
            # target shape: [Batch, 6, Z, Y, X] -> task_target: [Batch, 1, Z, Y, X]
            if isinstance(target, (list, tuple)):
                # 深监督开启时，target 是一个列表，每个元素对应一个分辨率
                task_target = [t[:, i:i+1] for t in target]
            else:
                task_target = target[:, i:i+1]
            
            total_loss += self.task_weights[i] * loss_func(task_output, task_target)
        return total_loss


# --- 修改后的训练器类 ---
class nnUNetTrainerUMambaBot(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        
        # 1. 强制设置设备 (保留你的原有逻辑)
        lr = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(lr)
        new_device = torch.device(f'cuda:{lr}')
        
        # 2. 调用父类初始化
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, new_device)
        
        # 3. 初始化配置
        # 指定 6 个任务的类别数（前景+背景）
        self.num_classes_list = [385, 184, 194, 247, 96, 269] 
        
        # 注意：这里你关闭了深监督。如果后续想开启，需要在这里改为 True
        self.enable_deep_supervision = False 

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        # 强制关闭深监督 (保持你原有逻辑)
        enable_deep_supervision = False 
        
        num_classes_list = [385, 184, 194, 247, 96, 269]

        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_bot_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                                 num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            # 调用修改后的 UMambaBot (Shared Decoder 版)
            model = get_umamba_bot_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                                 num_input_channels, deep_supervision=enable_deep_supervision,
                                                 num_classes_list=num_classes_list)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")
        
        print("UMambaBot Multi-Task Architecture Built.")
        return model

    def _build_loss(self):
        task_losses = []
        # 循环创建 6 个任务的 Loss
        for _ in range(6):
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, 
                                   weight_ce=1, weight_dice=1,
                                   ignore_label=self.label_manager.ignore_label, 
                                   dice_class=MemoryEfficientSoftDiceLoss)
            
            if self.enable_deep_supervision:
                deep_supervision_scales = self._get_deep_supervision_scales()
                # 计算深监督权重
                weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
                weights[-1] = 0
                weights = weights / weights.sum()
                loss = DeepSupervisionWrapper(loss, weights)
            
            task_losses.append(loss)

        # --- 核心修改：设置 Loss 权重 ---
        # 任务索引: 0=Task1, 1=Task2, 2=Task3, 3=Task4, 4=Task5, 5=Task6
        # 你指出 Task 2 (index 1) 和 Task 3 (index 2) 是真标签
        # 策略：给予 Task 2 和 3 更高的权重 (5.0)，强制 Shared Decoder 优先拟合它们
        # Task 1, 4, 5, 6 (伪标签) 权重设为 1.0
        weights = [1.0, 5.0, 5.0, 1.0, 1.0, 1.0]
        
        return MultiTask_DC_and_CE_loss(nn.ModuleList(task_losses), task_weights=weights)

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']

        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=True):
            # output 是一个列表，包含 6 个任务的输出
            output = self.network(data)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()
            
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(enabled=True):
            # output: [Head1_out, Head2_out, ..., Head6_out]
            output = self.network(data)
            l = self.loss(output, target)

        all_tp, all_fp, all_fn = [], [], []
        
        # 这里的 output[0] 取决于是否开启深监督
        # 如果关闭深监督，output[0] 是 Tensor(B, C, Z, Y, X)
        # 如果开启深监督，output[0] 是 List[Tensor]，我们需要取最高分辨率 output[0][0]
        # 下面的逻辑自动适配维度
        ref_output = output[0][0] if self.enable_deep_supervision else output[0]
        axes = [0] + list(range(2, ref_output.ndim))

        for i in range(6):
            # 提取第 i 个任务
            out_i = output[i][0] if self.enable_deep_supervision else output[i]
            tgt_i = target[0][:, i:i+1] if self.enable_deep_supervision else target[:, i:i+1]

            # 计算 Dice 指标
            seg_i = out_i.argmax(1)[:, None]
            onehot_i = torch.zeros(out_i.shape, device=out_i.device, dtype=torch.float32)
            onehot_i.scatter_(1, seg_i, 1)
            
            tp, fp, fn, _ = get_tp_fp_fn_tn(onehot_i, tgt_i, axes=axes)
            
            # 排除背景(第0类)，收集所有前景类
            all_tp.append(tp[1:])
            all_fp.append(fp[1:])
            all_fn.append(fn[1:])

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': torch.cat(all_tp, dim=0).detach().cpu().numpy(),
            'fp_hard': torch.cat(all_fp, dim=0).detach().cpu().numpy(),
            'fn_hard': torch.cat(all_fn, dim=0).detach().cpu().numpy()
        }

    def set_deep_supervision_enabled(self, enabled: bool):

        # 获取模型实例 (处理 DDP 包装)
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
            
        # 1. 设置 Shared Decoder 的深监督开关 (控制是否返回中间特征图)
        # 注意：我们在新的 UMambaBot 中定义了 self.decoder
        if hasattr(mod, 'decoder'):
            mod.decoder.deep_supervision = enabled
        
        # 2. 设置每个 Classification Head 的深监督开关
        # 注意：我们在新的 UMambaBot 中定义了 self.heads
        if hasattr(mod, 'heads'):
            for head in mod.heads:
                head.deep_supervision = enabled