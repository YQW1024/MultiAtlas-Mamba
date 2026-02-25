# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch import nn
from torch.cuda.amp import autocast
from typing import Union, Tuple, List

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
            # 获取第 i 个解码器的输出
            task_output = net_outputs[i]
            
            # 从 target 中切分出对应的通道。
            # 如果开启了深监督，target 是一个列表，列表里每个元素是 [B, 2, Z, Y, X]
            if isinstance(target, (list, tuple)):
                task_target = [t[:, i:i+1] for t in target]
            else:
                task_target = target[:, i:i+1]
            
            total_loss += self.task_weights[i] * loss_func(task_output, task_target)
        return total_loss


# --- 修改后的训练器类 ---
class nnUNetTrainerUMambaBot(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        
        
        import os
        # 获取当前进程的 rank
        lr = int(os.environ.get('LOCAL_RANK', 0))
        
        # 1. 核心补丁：强制设置全局当前设备
        torch.cuda.set_device(lr)
        
        # 2. 构造正确的设备对象并强制覆盖传入的参数
        new_device = torch.device(f'cuda:{lr}')
        
        # 3. 调用父类初始化，传入我们修正后的 new_device
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, new_device)
        
        
               
        
#        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # 指定两个任务的类别数（包含背景）
        self.num_classes_list = [385, 184, 194, 247, 96, 269] #1 初始化每个任务的类别数(含背景)
        self.enable_deep_supervision = False #1 关闭深监督

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        enable_deep_supervision = False #1 关闭深监督
        
        # 定义任务类别列表
        num_classes_list = [385, 184, 194, 247, 96, 269]

        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_bot_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                                 num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            # 调用修改后的 UMambaBot，传入 num_classes_list
            model = get_umamba_bot_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                                 num_input_channels, deep_supervision=enable_deep_supervision,
                                                 num_classes_list=num_classes_list)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")
        
        print("UMambaBot Multi-Task: {}".format(model))
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
                weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
                weights[-1] = 0
                weights = weights / weights.sum()
                loss = DeepSupervisionWrapper(loss, weights)
            task_losses.append(loss)

#yuan        return MultiTask_DC_and_CE_loss(nn.ModuleList(task_losses), task_weights=[1.0]*6)
        return MultiTask_DC_and_CE_loss(nn.ModuleList(task_losses), task_weights=[5.0, 1.0, 1.0, 1.0, 1.0, 1.0])#1

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']

        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=True):
            # 模型现在返回的是列表 [output_184, output_90]
            output = self.network(data)
            # 计算组合 Loss
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)#1 12
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)#1 12
            self.optimizer.step()
            
        return {'loss': l.detach().cpu().numpy()}




    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target'] # 假设你的 target 此时有 6 个通道
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(enabled=True):
            output = self.network(data) # 返回 6 个任务的输出列表
            l = self.loss(output, target)

        all_tp, all_fp, all_fn = [], [], []
        axes = [0] + list(range(2, output[0][0].ndim if self.enable_deep_supervision else output[0].ndim))

        for i in range(6):
            # 提取第 i 个任务的预测和标签
            out_i = output[i][0] if self.enable_deep_supervision else output[i]
            tgt_i = target[0][:, i:i+1] if self.enable_deep_supervision else target[:, i:i+1]

            # 计算指标
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
        """
        Override the default behavior because our UMambaBot has 
        'decoders' (ModuleList) instead of a single 'decoder'.
        """
        if self.is_ddp:
            # For Distributed Data Parallel
            for d in self.network.module.decoders:
                d.deep_supervision = enabled
        else:
            # For single GPU training
            for d in self.network.decoders:
                d.deep_supervision = enabled