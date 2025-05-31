import torch
from torch import nn
import torch.ao.nn.quantized.functional as F_q
from .irse import iResNet, BasicBlock, Bottleneck, conv1x1

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

# Spatial Pyramid Pooling (SPP): 
# phương pháp sử dụng để xử lý đầu vào với kích thước thay đổi mà không cần resize về một kích thước nhất định.
# cho phép mô hình trích xuất feature hiệu quả ở nhiều mức độ không spatial (không gian) khác nhau.
# Bản chất SPP sử dụng các pooling layer với các kích thước khác nhau và concat lại (kích thước ouput luôn là cố định)


# class MaxPoolingWrapper(nn.Module):
#     def __init__(self, output_size):
#         super().__init__()
#         self.output_size = output_size
#         self.adaptive_max_pool = nn.AdaptiveMaxPool2d(output_size)
        
#         self.quant = torch.ao.quantization.QuantStub()
#         self.dequant = torch.ao.quantization.DeQuantStub()

#     def forward(self, x):
#         x = self.quant(x) # Ensure input is quantized if it enters this custom module as float
#         x = self.adaptive_max_pool(x)
#         x = self.dequant(x) # Dequantize output if it needs to be float for subsequent float ops
#         return x

#     @classmethod
#     def from_float(cls, float_module):
#         q_module = cls(float_module.output_size)
#         return q_module

# class AvgPoolingWrapper(nn.Module):
#     def __init__(self, output_size):
#         super().__init__()
#         self.output_size = output_size
#         self.adaptive_max_pool = nn.AdaptiveAvgPool2d(output_size)
        
#         self.quant = torch.ao.quantization.QuantStub()
#         self.dequant = torch.ao.quantization.DeQuantStub()

#     def forward(self, x):
#         x = self.quant(x) # Ensure input is quantized if it enters this custom module as float
#         x = self.adaptive_max_pool(x)
#         x = self.dequant(x) # Dequantize output if it needs to be float for subsequent float ops
#         return x

#     @classmethod
#     def from_float(cls, float_module):
#         q_module = cls(float_module.output_size)
#         return q_module

class SPPModuleAvg(nn.Module):
    # Sau khi qua các lớp pooling, ta thu được các tensor có kích thước (B, C, 1, 1), (B, C, 2, 2), (B, C, 3, 3), (B, C, 6, 6)
    # Tiến hành làm phẳng thành 1-D vector: (B, C), (B, 4C), (B, 9C), (B, 16C) => cat lại (B, 30C)
    # view lại thành (B, 30C, 1, 1)
    def __init__(self, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.sizes = sizes
        # if pool_mode == 'avg':
        #     pool_layer = AvgPoolingWrapper
        # elif pool_mode == 'max':
        #     pool_layer = MaxPoolingWrapper
        # else:
        #     raise NotImplementedError

        # self.pool_blocks = nn.ModuleList([
        #     nn.Sequential(pool_layer(size), nn.Flatten()) for size in sizes
        # ])
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        xs = [F_q.adaptive_avg_pool2d(x,_size).flatten(start_dim=1) for _size in self.sizes]

        # xs = [block(x) for block in self.pool_blocks]
        x = torch.cat(xs, dim=1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = self.dequant(x)
        return x
    
    @classmethod
    def from_float(cls, float_module):
        # Create an instance of the quantized module
        q_module = cls()
        # No parameters to copy for AdaptiveMaxPool2d, but for other layers
        # you might copy weights/biases if applicable.
        return q_module


class SPPModuleMax(nn.Module):
    def __init__(self, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.sizes = sizes
        # if pool_mode == 'avg':
        #     pool_layer = AvgPoolingWrapper
        # elif pool_mode == 'max':
        #     pool_layer = MaxPoolingWrapper
        # else:
        #     raise NotImplementedError

        # self.pool_blocks = nn.ModuleList([
        #     nn.Sequential(pool_layer(size), nn.Flatten()) for size in sizes
        # ])
        
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        xs = [F_q.max_pool2d(x,_size).flatten(start_dim=1) for _size in self.sizes]
        # xs = [block(x) for block in self.pool_blocks]
        x = torch.cat(xs, dim=1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = self.dequant(x)
        return x
    
    @classmethod
    def from_float(cls, float_module):
        # Create an instance of the quantized module
        q_module = cls()
        # No parameters to copy for AdaptiveMaxPool2d, but for other layers
        # you might copy weights/biases if applicable.
        return q_module


class AttentionModule(nn.Module):
    # reduction: giảm số lượng channel trong Channel Attention
    # Thông qua Conv2d với kernel size = 1, số kênh ban đầu _channels được giảm còn _channels // reduction trước khi được phục hồi về channels
    def __init__(self, channels=512, reduction=16):
        super(AttentionModule, self).__init__()
        kernel_size = 7
        pool_size = (1, 2, 3)
        # Học thông tin bối cảnh từ nhiều cấp độ
        self.avg_spp = SPPModuleAvg(pool_size)
        self.avg_spp.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
        # Học các đặc điểm nổi bật từ nhiều cấp độ
        self.max_spp = SPPModuleMax(pool_size)
        self.max_spp.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
        # Học ma trận attention theo spatial từ đầu vào: input (N,2,H,W), ouput (N,1,H,W) với giá trị xác xuất
        self.spatial = nn.Sequential(
            # in_channels = 2: Max Pooling và Average Pooling theo chiều kênh
            # out_channels = 1: Chỉ xuất ra một kênh, đại diện cho attention map không gian.
            # Kích thước kernel bằng 7 để học các mối quan hệ không gian cục bộ.
            # padding=(kernel_size - 1) // 2: Sử dụng padding để giữ nguyên kích thước chiều cao và chiều rộng của đầu ra.
            # bias=False: Không sử dụng bias vì có Batch Normalization xử lý phía sau.
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2,
                      dilation=1, groups=1, bias=False),
            # Chuẩn hóa giá trị đầu ra của Conv2d theo từng batch, để tăng tốc độ huấn luyện và tính ổn định.
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
            # Áp dụng hàm kích hoạt sigmoid để giới hạn đầu ra trong khoảng [0,1], tạo ra attention map.
            nn.Sigmoid()
        )

        # Tính toán số lượng channel đầu vào dựa trên pool_size
        _channels = channels * int(sum([x ** 2 for x in pool_size]))
        
        # Học ma trận attention mỗi channel bằng cách học mối quan hệ giữa các kênh: (N,_channels,1,1) -> (N,channels,1,1) theo xác xuất chứa trọng số các channel
        # Tăng cường (scale) các kênh quan trọng và giảm ảnh hưởng của các kênh không quan trọng.
        self.channel = nn.Sequential(
            nn.Conv2d(
                _channels, _channels // reduction, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                _channels // reduction, channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channels, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_input = self.avg_spp(x) + self.max_spp(x)
        channel_scale = self.channel(channel_input)

        spatial_input = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        spatial_scale = self.spatial(spatial_input)

        x_non_id = (x * channel_scale + x * spatial_scale) * 0.5

        x_id = x - x_non_id
        
        return x_id, x_non_id


# Backbone trích xuât featuremap
class MIResNet(torch.nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.prelu = nn.PReLU()
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        
        self.spectacles_fsm = AttentionModule()
        self.facial_hair_fsm = AttentionModule()
        self.emotion_fsm = AttentionModule()
        self.pose_fsm = AttentionModule()
        self.gender_fsm = AttentionModule()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')  # Default 'relu'
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.PReLU):  # Initialize PReLU parameters separately
                nn.init.constant_(m.weight, 0.25)


        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 and self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )
        elif stride != 1:
            downsample = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer,
                            start_block=True))
        self.inplanes = planes * block.expansion
        exclude_bn0 = True
        for _ in range(1, (blocks-1)):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer,
                                exclude_bn0=exclude_bn0))
            exclude_bn0 = False

        layers.append(block(self.inplanes, planes, norm_layer=norm_layer, end_block=True, exclude_bn0=exclude_bn0))

        return nn.Sequential(*layers)



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x_non_spectacles, x_spectacles = self.spectacles_fsm(x)
        x_non_facial_hair, x_facial_hair = self.facial_hair_fsm(x_non_spectacles)
        x_non_emotion, x_emotion = self.emotion_fsm(x_non_facial_hair)
        x_non_pose, x_pose = self.pose_fsm(x_non_emotion)
        x_id, x_gender = self.gender_fsm(x_non_pose)
        
        return (
                (x_spectacles, x_non_spectacles),
                (x_facial_hair, x_non_facial_hair),
                (x_emotion, x_non_emotion),
                (x_pose, x_non_pose),
                (x_gender, x_id)
            )


def create_miresnet(model_name, **kwargs):
    configs = {
        "miresnet18": (BasicBlock, [2, 2, 2, 2]),
        "miresnet34": (BasicBlock, [3, 4, 6, 3]),
        "miresnet50": (Bottleneck, [3, 4, 6, 3]),
        "miresnet101": (Bottleneck, [3, 4, 23, 3]),
        "miresnet152": (Bottleneck, [3, 8, 36, 3]),
    }
    if model_name not in configs:
        raise ValueError(f"Model '{model_name}' không được hỗ trợ. Các model hợp lệ: {list(configs.keys())}")
    
    block, layers = configs[model_name]
    return MIResNet(block, layers, **kwargs)


if __name__ == '__main':
    backbone = create_miresnet('miresnet18', num_classes = 1000)