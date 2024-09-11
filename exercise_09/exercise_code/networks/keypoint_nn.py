"""Models for facial keypoint detection"""

import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams=hparams
        n_hidden=hparams["n_hidden"]
        self.ff=nn.Sequential(nn.LayerNorm(n_hidden),nn.Linear(n_hidden,4 *n_hidden),nn.GELU(), \
                              nn.Linear(4*n_hidden,n_hidden),nn.Dropout(p=0.6))

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.ff(x)
        x = x + shortcut  # Add the original input back
        return x

class MLP(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams=hparams
        n_hidden=hparams["n_hidden"]
        self.ff=nn.Sequential(nn.Linear(6400, n_hidden),nn.ELU(), nn.Dropout(p=0.5),
                              nn.Linear(n_hidden, 30))

    def forward(self, x):
        x = self.ff(x)
        return x
class ConvLayer(nn.Module):
    def __init__(self):
        super().__init__()
        num_conv_layer=4
        self.conv_layers=[nn.Conv2d(in_channels=2**(3+num_conv_layer-k), out_channels=2**(4+num_conv_layer-k), \
                               kernel_size=(k+1,k+1)) for k in range(num_conv_layer-1)]
        self.conv_layers.append(nn.Conv2d(in_channels=1, out_channels=2**(5), \
                               kernel_size=(num_conv_layer,num_conv_layer)))
        self.conv_layers[::-1]=self.conv_layers
        self.max_pool_layers=[nn.MaxPool2d(kernel_size=(2,2),stride=2) for _ in range(num_conv_layer)]
        self.dropout_layers=[nn.Dropout2d(p=torch.max(torch.tensor([0.6,(1+k)*1e-1])).item()) for k in range(num_conv_layer)]
        self.activation_layers=[nn.ELU() for _ in range(num_conv_layer)]
        self.conv_ff=[nn.Sequential(a,b,c,d) for a,b,c,d in \
                      zip(self.conv_layers,self.activation_layers,self.max_pool_layers,self.dropout_layers)]
        self.conv_ff=nn.Sequential(*self.conv_ff)
        self.conv_ff=nn.Sequential(self.conv_ff,nn.Flatten(start_dim=-3, end_dim=-1))

    def forward(self, x):
        return self.conv_ff(x)

class KeypointModel(nn.Module):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        """
        super().__init__()
        self.hparams = hparams
        
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        n_hidden=hparams["n_hidden"]
        self.conv_ff=ConvLayer()
        #self.mlp=MLP(hparams)
        self.trf_ll=[TransformerBlock(hparams) for _ in range(hparams["num_trf_ll"])]
        self.trf_ll=nn.Sequential(*self.trf_ll)
        self.model=nn.Sequential(self.conv_ff,nn.Linear(6400,n_hidden),self.trf_ll,nn.Linear(n_hidden,2*15))
        #self.model=nn.Sequential(self.conv_ff,self.mlp)
        self.set_optimizer()
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################


        out=self.model(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return out

    def set_optimizer(self):

        self.optimizer = None
        ########################################################################
        # TODO: Define your optimizer.                                         #
        ########################################################################

        optimizer=torch.optim.Adam
        optim_params = optimizer.__init__.__code__.co_varnames[:optimizer.__init__.__code__.co_argcount]
        parsed_arguments={ key: value for key,value in self.hparams.items() if key in optim_params}
        self.optimizer=optimizer(self.parameters(), **parsed_arguments)


class DummyKeypointModel(nn.Module):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
