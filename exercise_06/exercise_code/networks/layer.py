import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)

    :return out: output, of shape (N, M)
    :return cache: (x, w, b)
    """
    out = None
    x_reshaped = np.reshape(x, (x.shape[0], -1))
    out = x_reshaped.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: A numpy array of biases, of shape (M,

    :return dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    :return dw: Gradient with respect to w, of shape (D, M)
    :return db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    dw = np.reshape(x, (x.shape[0], -1)).T.dot(dout)
    dw = np.reshape(dw, w.shape)

    db = np.sum(dout, axis=0, keepdims=False)

    dx = dout.dot(w.T)
    dx = np.reshape(dx, x.shape)
    return dx, dw, db


class LayerNorm:
    def __init__(self):
        super().__init__()
        self.eps = 1e-5

    def forward(self, x, scale, shift):
        mean = np.mean(x,axis=-1, keepdims=True)
        var = np.var(x,axis=-1, keepdims=True)
        std_inv=1/np.sqrt(var + self.eps)
        norm_x = (x - mean) * std_inv
        cache=(norm_x, std_inv, scale)
        scale=1
        shift=0
        return scale * norm_x + shift, cache
    
    def backward(self, dout, cache):
        norm_x, std_inv, scale=cache
        dscale = np.sum(dout * norm_x, axis=-1, keepdims=True)
        dshift = np.sum(dout, axis=-1, keepdims=True)
        # Gradient for the normalized input
        scale=1
        dnorm_x = dout * scale
        # Compute gradient with respect to x
        dx = scale * std_inv * (
            dnorm_x - np.mean(dnorm_x, axis=-1, keepdims=True) 
            - norm_x * np.sum(dnorm_x * norm_x, axis=-1, keepdims=True)
        )
        return dx, dscale, dshift


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape

        :return out: Output, of the same shape as x
        :return cache: Cache, for backward computation, of the same shape as x
        """
        #outputs = 1 / (1 + np.exp(-x))
        outputs=np.zeros_like(x)
        outputs[x >= 0]=np.exp(-x[x >= 0])
        outputs[x >= 0]=1 / (1 + outputs[x >= 0])
        outputs[x < 0]=np.exp(x[x < 0])
        outputs[x < 0]=outputs[x < 0] / (1 + outputs[x < 0])
        cache = outputs
        return outputs, cache

    def backward(self, dout, cache):
        """
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        dx = None
        dx = dout * cache * (1 - cache)
        return dx


class Relu:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape

        :return out: Output, of the same shape as x
        :return cache: Cache, for backward computation, of the same shape as x
        """
        outputs = None
        cache = None
        outputs = np.maximum(x, 0)
        cache = x
        return outputs, cache

    def backward(self, dout, cache):
        """
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        dx = None
        x = cache
        dx = dout
        dx[x < 0] = 0
        return dx


class LeakyRelu:
    def __init__(self, slope=0.01):
        self.slope = slope

    def forward(self, x):
        """
        :param x: Inputs, of any shape

        :return out: Output, of the same shape as x
        :return cache: Cache, for backward computation, of the same shape as x
        """
        outputs = None
        cache = None
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of LeakyRelu activation function          #
        ########################################################################

        outputs = np.copy(x)
        outputs[x<0] = self.slope * outputs[x<0]
        cache = x

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return outputs, cache

    def backward(self, dout, cache):
        """
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        dx = None
        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass of LeakyRelu activation function         #
        ########################################################################

        dx = dout
        dx[cache<0] = self.slope * dout[cache<0]

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dx


class Tanh:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape

        :return out: Output, of the same shape as x
        :return cache: Cache, for backward computation, of the same shape as x
        """
        outputs = None
        cache = None
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of Tanh activation function               #
        ########################################################################

        outputs = np.tanh(x)
        cache = x

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return outputs, cache

    def backward(self, dout, cache):
        """
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        dx = None
        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass of Tanh activation function              #
        ########################################################################

        dx=(1-(np.tanh(cache))**2)*dout

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dx


class Gelu:
    def __init__(self):
        pass

    def forward(self, x):
        """
        Apply the GELU activation function.

        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

        Parameters:
        x : numpy array
            The input array.

        Returns:
        numpy array
            The output after applying GELU.
        """
        cache = x
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))), cache

    def backward(self, dout, x):
        """
        Compute the gradient of the GELU activation with respect to the input x.

        Parameters:
        dout : numpy array
            The gradient of the loss with respect to the output of GELU.
        x : numpy array
            The original input array that was fed to the forward method.

        Returns:
        numpy array
            The gradient of the loss with respect to the input x.
        """
        # Constants
        sqrt_2_pi = np.sqrt(2 / np.pi)
        x_cubed = np.power(x, 3)
        tanh_term = np.tanh(sqrt_2_pi * (x + 0.044715 * x_cubed))

        # Derivative of GELU
        grad = 0.5 * (1 + tanh_term) + \
               0.5 * x * (1 - np.power(tanh_term, 2)) * \
               (sqrt_2_pi * (1 + 3 * 0.044715 * np.power(x, 2)))

        return dout * grad
    

