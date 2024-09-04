from torch import nn
import torch
from ..network import MultiHeadAttention
from ..network import FeedForwardNeuralNetwork

class EncoderBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.0):
        """

        Args:
            d_model: Dimension of Embedding
            d_k: Dimension of Keys and Queries
            d_v: Dimension of Values
            n_heads: Number of Attention Heads
            d_ff: Dimension of hidden layer
            dropout: Dropout probability
        """
        super().__init__()

        self.multi_head = None
        self.layer_norm1 = None
        self.ffn = None
        self.layer_norm2 = None

        ########################################################################
        # TODO:                                                                #
        #   Task 6: Initialize the Encoder Block                               #
        #           You will need:                                             #
        #                           - Multi-Head Self-Attention layer          #
        #                           - Layer Normalization                      #
        #                           - Feed forward neural network layer        #
        #                           - Layer Normalization                      #
        #                                                                      #
        # Hint 6: Check out the pytorch layer norm module                      #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self,
                inputs: torch.Tensor,
                pad_mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            inputs: Inputs to the Encoder Block
            pad_mask: Optional Padding Mask

        Shape:
            - inputs: (batch_size, sequence_length, d_model)
            - pad_mask: (batch_size, sequence_length, sequence_length)
            - outputs: (batch_size, sequence_length, d_model)
        """
        outputs = None

        ########################################################################
        # TODO:                                                                #
        #   Task 6: Implement the forward pass of the encoder block            #
        #   Task 12: Pass on the padding mask                                  #
        #                                                                      #
        # Hint 6: Don't forget the residual connection! You can forget about   #
        #         the pad_mask for now!                                        #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs