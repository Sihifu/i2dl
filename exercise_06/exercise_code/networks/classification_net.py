import numpy as np
import os
import pickle

from exercise_code.networks.layer import affine_forward, affine_backward, Sigmoid, Tanh, LeakyRelu, Relu, Gelu, LayerNorm
from exercise_code.networks.base_networks import Network


class ClassificationNet(Network):
    """
    A fully-connected classification neural network with configurable 
    activation function, number of layers, number of classes, hidden size and
    regularization strength. 
    """

    def __init__(self, activation=Sigmoid, num_layer=2,
                 input_size=3 * 32 * 32, hidden_size=100,
                 std=1e-3, num_classes=10, reg=0, **kwargs):
        """
        :param activation: choice of activation function. It should implement
            a forward() and a backward() method.
        :param num_layer: integer, number of layers. 
        :param input_size: integer, the dimension D of the input data.
        :param hidden_size: integer, the number of neurons H in the hidden layer.
        :param std: float, standard deviation used for weight initialization.
        :param num_classes: integer, number of classes.
        :param reg: float, regularization strength.
        """
        super().__init__("cifar10_classification_net")

        self.activation = activation()
        self.reg_strength = reg

        self.cache = None

        self.memory = 0
        self.memory_forward = 0
        self.memory_backward = 0
        self.num_operation = 0
        
        # Initialize random gaussian weights for all layers and zero bias
        self.num_layer = num_layer
        self.std = std
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.norm_layer=False
        self.reset_weights()

    def forward(self, X):
        """
        Performs the forward pass of the model.

        :param X: Input data of shape N x D. Each X[i] is a training sample.
        :return: Predicted value for the data in X, shape N x 1
                 1-dimensional array of length N with the classification scores.
        """

        self.cache = {}
        self.reg = {}
        X = X.reshape(X.shape[0], -1)
        # Unpack variables from the params dictionary
        for i in range(self.num_layer - 1):
            W, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]

            # Forward i_th layer
            X, cache_affine = affine_forward(X, W, b)
            self.cache["affine" + str(i + 1)] = cache_affine

            # Activation function
            X, cache_sigmoid = self.activation.forward(X)
            self.cache["sigmoid" + str(i + 1)] = cache_sigmoid

            # Store the reg for the current W
            self.reg['W' + str(i + 1)] = np.sum(W ** 2) * self.reg_strength

        # last layer contains no activation functions
        W, b = self.params['W' + str(self.num_layer)],\
               self.params['b' + str(self.num_layer)]
        y, cache_affine = affine_forward(X, W, b)
        self.cache["affine" + str(self.num_layer)] = cache_affine
        self.reg['W' + str(self.num_layer)] = np.sum(W ** 2) * self.reg_strength

        return y

    def backward(self, dy):
        """
        Performs the backward pass of the model.

        :param dy: N x 1 array. The gradient wrt the output of the network.
        :return: Gradients of the model output wrt the model weights
        """

        # Note that last layer has no activation
        cache_affine = self.cache['affine' + str(self.num_layer)]
        dh, dW, db = affine_backward(dy, cache_affine)
        self.grads['W' + str(self.num_layer)] = \
            dW + 2 * self.reg_strength * self.params['W' + str(self.num_layer)]
        self.grads['b' + str(self.num_layer)] = db

        # The rest sandwich layers
        for i in range(self.num_layer - 2, -1, -1):
            # Unpack cache
            cache_sigmoid = self.cache['sigmoid' + str(i + 1)]
            cache_affine = self.cache['affine' + str(i + 1)]

            # Activation backward
            dh = self.activation.backward(dh, cache_sigmoid)

            # Affine backward
            dh, dW, db = affine_backward(dh, cache_affine)

            # Refresh the gradients
            self.grads['W' + str(i + 1)] = dW + 2 * self.reg_strength * \
                                           self.params['W' + str(i + 1)]
            self.grads['b' + str(i + 1)] = db

        return self.grads

    def save_model(self):
        self.eval()
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(model, open(directory + '/' + self.model_name + '.p', 'wb'))

    def get_dataset_prediction(self, loader):
        self.eval()
        scores = []
        labels = []
        
        for batch in loader:
            X = batch['image']
            y = batch['label']
            score = self.forward(X)
            scores.append(score)
            labels.append(y)
            
        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()

        return labels, preds, acc
    
    def eval(self):
        """sets the network in evaluation mode, i.e. only computes forward pass"""
        self.return_grad = False
        
        # Delete unnecessary caches, to mitigate a memory prolbem.
        self.reg = {}
        self.cache = {}
        
    def reset_weights(self):
        self.params = {'W1':self.std * np.random.randn(self.input_size, self.hidden_size),
                       'b1': np.zeros(self.hidden_size)
                       }

        for i in range(self.num_layer - 2):
            self.params['W' + str(i + 2)] = self.std * np.random.randn(self.hidden_size,
                                                                  self.hidden_size)
            self.params['b' + str(i + 2)] = np.zeros(self.hidden_size)

        self.params['W' + str(self.num_layer)] = self.std * np.random.randn(self.hidden_size,
                                                                  self.num_classes)
        self.params['b' + str(self.num_layer)] = np.zeros(self.num_classes)

        self.grads = {}
        self.reg = {}
        for i in range(self.num_layer):
            self.grads['W' + str(i + 1)] = 0.0
            self.grads['b' + str(i + 1)] = 0.0


        

class MyOwnNetwork(ClassificationNet):
    """
    Your first fully owned network!
    
    You can define any arbitrary network architecture here!
    
    As a starting point, you can use the code from ClassificationNet above as 
    reference or even copy it to MyOwnNetwork, but of course you're also free 
    to come up with a complete different architecture and add any additional 
    functionality! (Without renaming class functions though ;))
    """

    def __init__(self, activation=Gelu, num_layer=4,
                 input_size=3 * 32 * 32, hidden_size=2**7,
                 std=1e-3, num_classes=10, reg=0, shortcut=True,norm_layer=True,norm_function=LayerNorm,**kwargs):
        """
        Your network initialization. For reference and starting points, check
        out the classification network above.
        """

        super().__init__()

        ########################################################################
        # TODO:  Your initialization here                                      #
        ########################################################################


        self.activation = activation()
        self.reg_strength = reg

        self.cache = None

        self.memory = 0
        self.memory_forward = 0
        self.memory_backward = 0
        self.num_operation = 0

        # Initialize random gaussian weights for all layers and zero bias
        self.num_layer = num_layer
        self.std = std
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.shortcut=shortcut
        self.norm_layer=norm_layer
        self.norm_function=norm_function()
        self.num_classes = num_classes
        self.reset_weights()

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################



    def forward(self, X):
        out = None
        ########################################################################
        # TODO:  Your forward here                                             #
        ########################################################################

        self.cache = {}
        self.reg = {}
        X = X.reshape(X.shape[0], -1)

        # first layer
        # Norm layer
        if self.norm_layer:
            scale = self.params["scale" + str(1)]
            shift = self.params["shift" + str(1)]
            X, cache_norm = self.norm_function.forward(X,scale=scale, shift=shift)
            self.cache["norm" + str(1)] = cache_norm


        W, b = self.params['W' + str(1)], self.params['b' + str(1)]
        X, cache_affine = affine_forward(X, W, b)
        self.cache["affine" + str(1)] = cache_affine
        # Activation function
        X, cache_sigmoid = self.activation.forward(X)
        self.cache["sigmoid" + str(1)] = cache_sigmoid
        # Store the reg for the current W
        self.reg['W' + str(1)] = np.sum(W ** 2) * self.reg_strength

        # Unpack variables from the params dictionary
        for i in range(1, self.num_layer - 1):
            W, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]

            # Forward i_th layer

            # Norm layer
            if self.norm_layer:
                scale = self.params["scale" + str(i+1)]
                shift = self.params["shift" + str(i+1)]
                X, cache_norm = self.norm_function.forward(X,scale=scale, shift=shift)
                self.cache["norm" + str(i + 1)] = cache_norm

            X_pre=np.copy(X)
            X, cache_affine = affine_forward(X, W, b)

            
            self.cache["affine" + str(i + 1)] = cache_affine

            # Activation function
            X, cache_sigmoid = self.activation.forward(X)
            self.cache["sigmoid" + str(i + 1)] = cache_sigmoid

            # shortcut_connection
            if self.shortcut:
                X=X+X_pre

            # Store the reg for the current W
            self.reg['W' + str(i + 1)] = np.sum(W ** 2) * self.reg_strength

        # last layer contains no activation functions
        W, b = self.params['W' + str(self.num_layer)],\
               self.params['b' + str(self.num_layer)]
        y, cache_affine = affine_forward(X, W, b)
        self.cache["affine" + str(self.num_layer)] = cache_affine
        self.reg['W' + str(self.num_layer)] = np.sum(W ** 2) * self.reg_strength

        out=y

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return out

    def backward(self, dy):
        grads = None
        ########################################################################
        # TODO:  Your backward here                                            #
        ########################################################################

        cache_affine = self.cache['affine' + str(self.num_layer)]
        dh, dW, db = affine_backward(dy, cache_affine)
        self.grads['W' + str(self.num_layer)] = \
            dW + 2 * self.reg_strength * self.params['W' + str(self.num_layer)]
        self.grads['b' + str(self.num_layer)] = db

        # The rest middle layers
        for i in range(self.num_layer - 2, -1, -1):
            # Unpack cache
            cache_sigmoid = self.cache['sigmoid' + str(i + 1)]
            cache_affine = self.cache['affine' + str(i + 1)]
            # Activation backward
            dh1 = self.activation.backward(dh, cache_sigmoid)

            # Affine backward
            dh2, dW, db = affine_backward(dh1, cache_affine)
            if i!=0 and self.shortcut:
                dh2=dh2+dh
            dh=dh2
       
            # Layer Norm backward
            if self.norm_layer:
                cache_norm=self.cache['norm' + str(i + 1)]
                dh, dscale, dshift = self.norm_function.backward(dh, cache_norm)
            #    self.grads['scale' + str(i + 1)] = dscale
            #    self.grads['shift' + str(i + 1)] = dshift
            # Refresh the gradients
            self.grads['W' + str(i + 1)] = dW + 2 * self.reg_strength * \
                                           self.params['W' + str(i + 1)]
            self.grads['b' + str(i + 1)] = db

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return grads

    def reset_weights(self):
        self.params = {'W1':self.std * np.random.randn(self.input_size, self.hidden_size),
                    'b1': np.zeros(self.hidden_size)
                    }
        if self.norm_layer:
            self.params['scale1']=self.std * np.random.randn(1,1)
            self.params['shift1']= np.zeros((1,1))

        for i in range(self.num_layer - 2):
            self.params['W' + str(i + 2)] = self.std * np.random.randn(self.hidden_size,
                                                                self.hidden_size)
            self.params['b' + str(i + 2)] = np.zeros(self.hidden_size)
            if self.norm_layer:
                self.params['scale' + str(i + 2)] = self.std * np.random.randn(1,1)
                self.params['shift' + str(i + 2)] = np.zeros((1,1))
        self.params['W' + str(self.num_layer)] = self.std * np.random.randn(self.hidden_size,
                                                                self.num_classes)
        self.params['b' + str(self.num_layer)] = np.zeros(self.num_classes)

        self.grads = {}
        self.reg = {}
        for i in range(self.num_layer):
            self.grads['W' + str(i + 1)] = 0.0
            self.grads['b' + str(i + 1)] = 0.0
        if self.norm_layer:
            for i in range(self.num_layer-1):
                self.grads['scale' + str(i + 1)] = 0.0
                self.grads['shift' + str(i + 1)] = 0.0
