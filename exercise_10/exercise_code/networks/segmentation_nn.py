"""SegmentationNN"""
import torch
import torch.nn as nn

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

    # @property
    # def is_cuda(self):
    #     """
    #     Check if model parameters are allocated on the GPU.
    #     """
    #     return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
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
    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")