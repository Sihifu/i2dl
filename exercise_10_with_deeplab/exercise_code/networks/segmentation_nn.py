"""SegmentationNN"""
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision import models, transforms


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        # Define image transformations
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        self.model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        # Freeze all layers in the backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = False
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
        out=self.model(self.preprocess(x))["out"]
        
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