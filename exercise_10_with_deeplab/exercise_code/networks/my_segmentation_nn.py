import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights 
from torch.nn import functional as F

class BasicBlock(nn.Module):
    def __init__(self, input_channels, downsample=True):
        super().__init__()
        self.downsample = None
        if downsample:
            self.conv1 = nn.Conv2d(input_channels, 2*input_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(2*input_channels)
            self.conv2 = nn.Conv2d(2*input_channels, 2*input_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(2*input_channels)
            self.downsample=nn.Sequential(nn.Conv2d(input_channels, 2*input_channels, kernel_size=(1, 1), stride=(2, 2), bias=False), \
                                nn.BatchNorm2d(2*input_channels))
        else:
            self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(input_channels)
            self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(input_channels)
        self.gelu = nn.GELU()

    def forward(self,x):
        identity=x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.gelu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        if self.downsample is not None:
            identity=self.downsample(identity)
        x=x+identity
        x=self.gelu(x)
        return x

class StandardLayer(nn.Module):
    def __init__(self,input_channel,output_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel,output_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.downsample=nn.Sequential(nn.Conv2d(input_channel,output_channel, kernel_size=1, bias=False), \
                                nn.BatchNorm2d(output_channel))
        self.gelu = nn.GELU()

    def forward(self,x):
        identity=x
        x=self.conv1(x)
        x=self.bn1(x)
        identity=self.downsample(identity)
        x=x+identity
        x=self.gelu(x)
        return x
    


class PreBlock(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(3,64,7,1,3,bias=False),nn.BatchNorm2d(64),nn.GELU(),nn.MaxPool2d(3,2,1))


class BackBone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pre_block=PreBlock()
        self.layer1=nn.Sequential(BasicBlock(64,downsample=False),BasicBlock(64,downsample=False))
        self.layer2=nn.Sequential(BasicBlock(64,downsample=True),BasicBlock(128,downsample=False))
        self.layer3=StandardLayer(128,256)
        self.load_resnet_weights()

    def forward(self, x):
        x=self.pre_block(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        return x

    def load_resnet_weights(self):
        res=resnet18(weights=ResNet18_Weights.DEFAULT)
        leaves_res=self.get_leaves(res)
        leaves_module=self.get_leaves(self)
        
        i=0
        while True:
            if self.compare_leaves(leaves_module[i],leaves_res[i]):
                leaves_module[i].load_state_dict(leaves_res[i].state_dict())
                i+=1
            else: 
                break

    def get_leaves(self,module):
        leaves=[]
        def _get_leaves(module,leaves):
            for child in module.children():
                if len(list(child.children())) == 0:  # No sub-modules, so it's a leaf
                    if any(param.requires_grad for param in child.parameters()):
                        leaves.append(child)
                else:
                    _get_leaves(child,leaves)
            return leaves  
        return _get_leaves(module,leaves)

    def compare_leaves(self, module1, module2):
        return str(module1) == str(module2)
    


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates = (12, 24, 36, 48) , out_channels= 256):
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.GELU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res) 

    
class SegmentationNN(nn.Module):

    def __init__(self, encoder, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        # Define image transformations
        
        self.encoder=encoder
        self.head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, num_classes, 1)
        )

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        input_shape=x.shape[2:]
        x=self.encoder(x)
        x=self.head(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x
    
class Encoder(nn.Module):
    def __init__(self, hp=None):
        super().__init__()
        self.hp = hp
        self.backbone=BackBone()
        self.aspp=ASPP(in_channels=256, atrous_rates=(12, 24, 32, 48), out_channels=256)


    def forward(self, x):

        x=self.backbone(x)
        x=self.aspp(x)

        return x

class AutoEnc(nn.Module):
    def __init__(self, encoder, in_channels=256, out_channels=3, hp={}):
        super().__init__()
        self.hp=hp
        inter_channels = in_channels // 4
        self.encoder=encoder
        self.layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1),
        ]   
        self.fcn=nn.Sequential(*self.layers)
        self.set_optimizer()
        self.set_optimizer_scheduler()
        
    def forward(self,x):
        input_shape=x.shape[2:]
        x=self.encoder(x)
        x=self.fcn(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x

    def set_optimizer(self):
        self.optimizer=self.hp.get("optimizer",torch.optim.Adam)
        optim_params = self.optimizer.__init__.__code__.co_varnames[:self.optimizer.__init__.__code__.co_argcount]
        parsed_arguments={ key: value for key,value in self.hp.items() if key in optim_params}
        self.optimizer=self.optimizer(self.parameters(), **parsed_arguments)

    def set_optimizer_scheduler(self):
        self.scheduler=torch.optim.lr_scheduler.StepLR
        scheduler_params = torch.optim.lr_scheduler.StepLR.__init__.__code__.co_varnames[:torch.optim.lr_scheduler.StepLR.__init__.__code__.co_argcount]
        parsed_arguments={ key: value for key,value in self.hp.items() if key in scheduler_params}
        self.optimizer_scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer ,step_size=self.hp.get("step_size",100),**parsed_arguments)

