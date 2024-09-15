"""SegmentationNN"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
class BasicBlock(nn.Module):
    def __init__(self, num_input_channels, hparams, downsample=False):
        super().__init__()
        self.downsample = None
        if downsample:
            self.conv1 = nn.Conv2d(num_input_channels, 2*num_input_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(2*num_input_channels)
            self.dropout1 = nn.Dropout(hparams["dropout_percentage"])
            self.gelu1 = nn.GELU()
            self.conv2 = nn.Conv2d(2*num_input_channels, 2*num_input_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(2*num_input_channels)
            self.dropout2 = nn.Dropout(hparams["dropout_percentage"])
            self.downsample=nn.Sequential(nn.Conv2d(num_input_channels, 2*num_input_channels, kernel_size=(1, 1), stride=(2, 2), bias=False), \
                                nn.BatchNorm2d(2*num_input_channels))
        else:
            self.conv1 = nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(num_input_channels)
            self.dropout1 = nn.Dropout(hparams["dropout_percentage"])
            self.gelu1 = nn.GELU()
            self.conv2 = nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(num_input_channels)
            self.dropout2 = nn.Dropout(hparams["dropout_percentage"])
        self.gelu_out = nn.GELU()
            
    def forward(self,x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)
        out = self.gelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gelu_out(out)
        return out
class BasicBlockUp(nn.Module):
    def __init__(self, num_input_channels, hparams, upsample=False):
        super().__init__()
        self.upsample = None
        if upsample:
            self.conv1 = nn.ConvTranspose2d(num_input_channels, num_input_channels//2, kernel_size=2, stride=2, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(num_input_channels//2)
            self.dropout1 = nn.Dropout(hparams["dropout_percentage"])
            self.gelu1 = nn.GELU()
            self.conv2 = nn.ConvTranspose2d(num_input_channels//2, num_input_channels//2, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(num_input_channels//2)
            self.dropout2 = nn.Dropout(hparams["dropout_percentage"])
            self.upsample=nn.Sequential(nn.ConvTranspose2d(num_input_channels, num_input_channels//2, kernel_size=2, stride=(2, 2), padding=0, bias=False), \
                                nn.BatchNorm2d(num_input_channels//2))
        else:
            self.conv1 = nn.ConvTranspose2d(num_input_channels, num_input_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(num_input_channels)
            self.dropout1 = nn.Dropout(hparams["dropout_percentage"])
            self.gelu1 = nn.GELU()
            self.conv2 = nn.ConvTranspose2d(num_input_channels, num_input_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(num_input_channels)
            self.dropout2 = nn.Dropout(hparams["dropout_percentage"])
        self.gelu_out = nn.GELU()
            
    def forward(self,x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)
        out = self.gelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.gelu_out(out)
        return out
class ConvBlock(nn.Module):
    def __init__(self,num_input_channels,hparams):
        super().__init__()
        self.conv1 = nn.Conv2d(num_input_channels, num_input_channels//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_input_channels//2)
        self.dropout1 = nn.Dropout(hparams["dropout_percentage"])
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(num_input_channels//2, num_input_channels//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_input_channels//2)
        self.dropout2 = nn.Dropout(hparams["dropout_percentage"])
        self.gelu2 = nn.GELU()

        self.downsample=nn.Sequential(nn.Conv2d(num_input_channels, num_input_channels//2, kernel_size=(1, 1), stride=(1, 1), bias=False), \
                                nn.BatchNorm2d(num_input_channels//2))

    def forward(self,x):
        identity=self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)
        out = self.gelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        out += identity
        out = self.gelu2(out)
        return out
class ConvLayer(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(hparams["dropout_percentage"])
        self.gelu1 = nn.GELU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        
        self.down_layer1 = nn.Sequential(BasicBlock(64,hparams,downsample=False),BasicBlock(64,hparams,downsample=False))
        self.down_layer2 = nn.Sequential(BasicBlock(64,hparams,downsample=True),BasicBlock(128,hparams,downsample=False))
        self.down_layer3 = nn.Sequential(BasicBlock(128,hparams,downsample=True),BasicBlock(256,hparams,downsample=False))

        
    def forward(self, x):
        # input height=240
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu1(x)
        x = self.maxpool1(x)
        # x.shape=(N,64,h=60,w=60)
        x = self.down_layer1(x)
        # x.shape=(N,64,h=60,w=60)
        x = self.down_layer2(x)
        # x.shape=(N,128,h=30,w=30)
        x = self.down_layer3(x)
        # x.shape=(N,256,h=15,w=15)

        return x
class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################


        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(hp["dropout_percentage"])
        self.gelu1 = nn.GELU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        
        self.down_layer1 = nn.Sequential(BasicBlock(64,hp,downsample=False),BasicBlock(64,hp,downsample=False))
        self.down_layer2 = nn.Sequential(BasicBlock(64,hp,downsample=True),BasicBlock(128,hp,downsample=False))
        self.down_layer3 = nn.Sequential(BasicBlock(128,hp,downsample=True),BasicBlock(256,hp,downsample=False))

        self.up_layer1 = nn.Sequential(BasicBlockUp(256,hp,upsample=True),BasicBlockUp(128,hp,upsample=False))
        self.convblock1 = ConvBlock(256,hp)
        self.up_layer2 = nn.Sequential(BasicBlockUp(128,hp,upsample=True),BasicBlockUp(64,hp,upsample=False))
        self.convblock2 = ConvBlock(128,hp)
        self.up_layer3 = nn.Sequential(BasicBlockUp(64,hp,upsample=True),BasicBlockUp(32,hp,upsample=False))

        self.outputlayer= nn.Sequential(nn.ConvTranspose2d(64+32, 3, kernel_size=6, stride=2, padding=2, bias=False),nn.Conv2d(3, num_classes, kernel_size=1))
        self.load_resnet_weights()
        self.set_optimizer()
        self.set_optimizer_scheduler()
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
        # x.shape=(N,3,240,240)
        x1 = self.conv1(x)
        # x.shape=(N,64,120,120)
        x2 = self.bn1(x1)
        x3 = self.gelu1(x2)
        x4 = self.maxpool1(x3)
        # x.shape=(N,64,h=60,w=60)
        x5 = self.down_layer1(x4)
        # x.shape=(N,64,h=60,w=60)
        x6 = self.down_layer2(x5)
        # x.shape=(N,128,h=30,w=30)
        x7 = self.down_layer3(x6)
        # x.shape=(N,256,h=15,w=15)
        
        x8 = self.up_layer1(x7)
        # x.shape=(N,128,h=30,w=30)
        x9 = torch.cat((x8,x6),dim=1)
        # x.shape=(N,256,h=30,w=30)
        x10 = self.convblock1(x9)
        # x.shape=(N,128,h=30,w=30)
        x11= self.up_layer2(x10)
        # x.shape=(N,64,h=60,w=60)
        x12 = torch.cat((x11,x5),dim=1)
        # x.shape=(N,128,h=60,w=60)
        x13 = self.convblock2(x12)
        # x.shape=(N,64,h=60,w=60)
        x14 = self.up_layer3(x13)
        # x.shape=(N,32,h=120,w=120)
        x15 = torch.cat((x14,x1),dim=1)
        # x.shape=(N,64+32,h=120,w=120)
        out=self.outputlayer(x15)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return out

    def reset_for_training(self, hp=None):
        self.current_weights=self.state_dict()
        if hp:
            self.__init__(hp)
        else:
            self.__init__(self.hp)
        self.load_state_dict(self.current_weights)


    def set_optimizer(self):
        self.optimizer=self.hp.get("optimizer",torch.optim.Adam)
        optim_params = self.optimizer.__init__.__code__.co_varnames[:self.optimizer.__init__.__code__.co_argcount]
        parsed_arguments={ key: value for key,value in self.hp.items() if key in optim_params}
        self.optimizer=self.optimizer(self.parameters(), **parsed_arguments)

    def set_optimizer_scheduler(self):
        self.scheduler=torch.optim.lr_scheduler.StepLR
        scheduler_params = torch.optim.lr_scheduler.StepLR.__init__.__code__.co_varnames[:torch.optim.lr_scheduler.StepLR.__init__.__code__.co_argcount]
        parsed_arguments={ key: value for key,value in self.hp.items() if key in scheduler_params}
        self.optimizer_scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer ,**parsed_arguments)

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
    
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
    

class Autoencoder(nn.Module):

    def __init__(self, hparams, encoder):
        super().__init__()
        # set hyperparams
        self.hp = hparams
        # Define models
        self.encoder = encoder
        self.decoder = nn.Conv2d(23, 3, kernel_size=1)
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()
        self.set_optimizer_scheduler()

    def forward(self, x):
        reconstruction = None
        ########################################################################
        # TODO: Feed the input image to your encoder to generate the latent    #
        #  vector. Then decode the latent vector and get your reconstruction   #
        #  of the input.                                                       #
        ########################################################################

        latent=self.encoder(x)
        reconstruction=self.decoder(latent)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return reconstruction
    
    def set_optimizer(self):
        self.optimizer=self.hp.get("optimizer",torch.optim.Adam)
        optim_params = self.optimizer.__init__.__code__.co_varnames[:self.optimizer.__init__.__code__.co_argcount]
        parsed_arguments={ key: value for key,value in self.hp.items() if key in optim_params}
        self.optimizer=self.optimizer(self.parameters(), **parsed_arguments)

    def set_optimizer_scheduler(self):
        self.scheduler=torch.optim.lr_scheduler.StepLR
        scheduler_params = torch.optim.lr_scheduler.StepLR.__init__.__code__.co_varnames[:torch.optim.lr_scheduler.StepLR.__init__.__code__.co_argcount]
        parsed_arguments={ key: value for key,value in self.hp.items() if key in scheduler_params}
        self.optimizer_scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer ,**parsed_arguments)


class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

if __name__ == "__main__":
    from torchinfo import summary
    summary(SegmentationNN(hp={"dropout_percentage":0.5}), (1, 3, 240, 240), device="cpu")