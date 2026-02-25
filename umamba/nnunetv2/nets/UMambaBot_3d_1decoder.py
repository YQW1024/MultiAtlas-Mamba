# -*- coding: utf-8 -*-
import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from mamba_ssm import Mamba
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

# -------------------------------------------------------------------------
# 1. 基础组件类 (UpsampleLayer, MambaLayer, BasicResBlock, UNetResEncoder)
#    这些类保持不变，但为了代码完整性，全部包含在下面
# -------------------------------------------------------------------------

class UpsampleLayer(nn.Module):
    def __init__(self, conv_op, input_channels, output_channels, pool_op_kernel_size, mode='nearest'):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
    
    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16: x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out

class BasicResBlock(nn.Module):
    def __init__(self, conv_op, input_channels, output_channels, norm_op, norm_op_kwargs, kernel_size=3, padding=1, stride=1, use_1x1conv=False, nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}):
        super().__init__()
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3: x = self.conv3(x)
        y += x
        return self.act2(y)

class UNetResEncoder(nn.Module):
    def __init__(self, input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_blocks_per_stage, conv_bias=False, norm_op=None, norm_op_kwargs=None, nonlin=None, nonlin_kwargs=None, return_skips=False, stem_channels=None, pool_type='conv'):
        super().__init__()
        if isinstance(kernel_sizes, int): kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int): features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int): n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int): strides = [strides] * n_stages
        
        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None
        self.conv_pad_sizes = [[i // 2 for i in krnl] for krnl in kernel_sizes]
        stem_channels = features_per_stage[0]
        self.stem = nn.Sequential(
            BasicResBlock(conv_op, input_channels, stem_channels, norm_op, norm_op_kwargs, 
                          kernel_sizes[0], self.conv_pad_sizes[0], 1, True, nonlin, nonlin_kwargs),
            *[BasicBlockD(conv_op, stem_channels, stem_channels, kernel_sizes[0], 1, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs) for _ in range(n_blocks_per_stage[0] - 1)]
        )
        input_channels = stem_channels
        stages = []

        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(conv_op, input_channels, features_per_stage[s], norm_op, norm_op_kwargs, 
                              kernel_sizes[s], self.conv_pad_sizes[s], strides[s], True, nonlin, nonlin_kwargs),
                *[BasicBlockD(conv_op, features_per_stage[s], features_per_stage[s], kernel_sizes[s], 1, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs) for _ in range(n_blocks_per_stage[s] - 1)]
            )
            stages.append(stage)
            input_channels = features_per_stage[s]
        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None: x = self.stem(x)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips: return ret
        else: return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None: output = self.stem.compute_conv_feature_map_size(input_size)
        else: output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output

# -------------------------------------------------------------------------
# 2. 修改后的 Decoder：只负责提取特征，不再输出类别
# -------------------------------------------------------------------------

class UNetResDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1

        stages = []
        upsample_layers = []
        
        # 记录每层 Decoder 的输出通道数，供后面的 Segmentation Heads 使用
        self.output_channels = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]

            upsample_layers.append(UpsampleLayer(
                conv_op=encoder.conv_op,
                input_channels=input_features_below,
                output_channels=input_features_skip,
                pool_op_kernel_size=stride_for_upsampling,
                mode='nearest'
            ))


            stages.append(nn.Sequential(
                BasicResBlock(encoder.conv_op, 2 * input_features_skip, input_features_skip, 
                              encoder.norm_op, encoder.norm_op_kwargs, encoder.kernel_sizes[-(s + 1)], 
                              encoder.conv_pad_sizes[-(s + 1)], 1, True, encoder.nonlin, encoder.nonlin_kwargs),
                *[
                    BasicBlockD(
                        conv_op=encoder.conv_op,
                        input_channels=input_features_skip,
                        output_channels=input_features_skip,
                        kernel_size=encoder.kernel_sizes[-(s + 1)],
                        stride=1,
                        conv_bias=encoder.conv_bias,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        nonlin=encoder.nonlin,
                        nonlin_kwargs=encoder.nonlin_kwargs,
                    ) for _ in range(n_conv_per_stage[s-1] - 1)
                ]
            ))
            # 记录通道数
            self.output_channels.append(input_features_skip)

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        decoder_outputs = [] # 存储每一层级的特征图
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            # 无论是否深监督，都先把特征图存下来
            decoder_outputs.append(x)
            lres_input = x
        
        # 翻转顺序，让 index 0 对应最高分辨率，index -1 对应最低分辨率
        return decoder_outputs[::-1]

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
        return output

# -------------------------------------------------------------------------
# 3. 新增类：多尺度分割头 (MultiScaleHead)
#    负责将 Decoder 的特征图映射到具体的类别
# -------------------------------------------------------------------------

class MultiScaleHead(nn.Module):
    def __init__(self, conv_op, input_channels_list, num_classes, deep_supervision):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.seg_layers = nn.ModuleList()
        # 为不同分辨率的特征图分别建立 1x1 卷积头
        for ch in input_channels_list:
            self.seg_layers.append(conv_op(ch, num_classes, 1, 1, 0, bias=True))

    def forward(self, decoder_features):
        # decoder_features 是特征图列表: [HighRes, ..., LowRes]
        seg_outputs = []
        if self.deep_supervision:
            # 深监督开启：对所有层都进行预测
            for i, feat in enumerate(decoder_features):
                seg_outputs.append(self.seg_layers[i](feat))
        else:
            # 深监督关闭：只对最高分辨率层(index 0)进行预测
            seg_outputs.append(self.seg_layers[0](decoder_features[0]))
        
        # 保持与 nnUNet 接口一致：如果非深监督，返回 Tensor；如果是，返回 List[Tensor]
        if not self.deep_supervision:
            return seg_outputs[0]
        return seg_outputs

# -------------------------------------------------------------------------
# 4. 主模型 UMambaBot：整合共享 Encoder、共享 Decoder 和 独立 Heads
# -------------------------------------------------------------------------

class UMambaBot(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes_list: List[int], # 传入 6 个任务的类别数
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 stem_channels: int = None
                 ):
        super().__init__()
        
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int): n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int): n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        # Mamba 相关的层数调整 (Bottleneck部分)
        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[s] = 1   
        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[s] = 1

        # 1. 共享 Encoder
        self.encoder = UNetResEncoder(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs,
            return_skips=True, stem_channels=stem_channels
        )

        # 2. 共享 Bottleneck (Mamba)
        self.mamba_layer = MambaLayer(dim = features_per_stage[-1])

        # 3. 共享 Decoder (这是关键：Task 2/3 的真值梯度会优化这个模块)
        self.decoder = UNetResDecoder(self.encoder, n_conv_per_stage_decoder, deep_supervision)
        
        # 获取 Decoder 输出通道顺序 (反转以匹配 [HighRes -> LowRes])
        # decoder.output_channels 存的是 [Layer1, Layer2...] (Deep to Shallow in creation, but implementation matches forward)
        # 我们在 decoder forward 里做了 [::-1]，所以这里通道数也要对应调整。
        # UNetResDecoder 的 output_channels append 顺序是从深层(LowRes)往浅层(HighRes)走的。
        # 所以 reverse 后正好是 [HighRes, ..., LowRes]
        channels_high_to_low = self.decoder.output_channels[::-1]

        # 4. 独立 Heads (Multi-Head)
        # 分别为 6 个任务创建独立的分类头
        self.heads = nn.ModuleList([
            MultiScaleHead(conv_op, channels_high_to_low, nc, deep_supervision)
            for nc in num_classes_list
        ])

    def forward(self, x):
        skips = self.encoder(x)
        skips[-1] = self.mamba_layer(skips[-1])
        
        # 步骤 1: 共享解码，得到特征图列表
        decoder_features = self.decoder(skips)
        
        # 步骤 2: 多头输出，每个 Head 负责一个任务
        # 返回列表: [Output_Task1, Output_Task2, ..., Output_Task6]
        return [head(decoder_features) for head in self.heads]

    def compute_conv_feature_map_size(self, input_size):
        if not hasattr(self.encoder, 'conv_op'): conv_op = convert_dim_to_conv_op(len(input_size))
        else: conv_op = self.encoder.conv_op
        
        size = self.encoder.compute_conv_feature_map_size(input_size)
        size += self.decoder.compute_conv_feature_map_size(input_size)
        return size

# -------------------------------------------------------------------------
# 5. 构建函数 helper
# -------------------------------------------------------------------------

def get_umamba_bot_3d_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True,
        num_classes_list: List[int] = None 
    ):
    # 默认设置 6 个任务类别 (Task 1=385, Task 2=184, Task 3=194...)
    # 你的场景：Task 2 (idx 1) 和 Task 3 (idx 2) 是真标签
    if num_classes_list is None:
        num_classes_list = [385, 184, 194, 247, 96, 269]

    num_stages = len(configuration_manager.conv_kernel_sizes)
    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    segmentation_network_class_name = 'UMambaBot'
    network_class = UMambaBot
    kwargs = {
        'UMambaBot': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }

    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes_list=num_classes_list,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))
    return model