from torch import nn
import torch

class FeedForwardNeuralNetwork(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 dropout: float = 0.0):
        """

        Args:
            d_model: Dimension of Embedding
            d_ff: Dimension of hidden layer
            dropout: Dropout probability
        """
        super().__init__()

        self.linear_1 = None
        self.relu = None
        self.linear_2 = None
        self.dropout = None

        ########################################################################
        # TODO:                                                                #
        #   Task 5: Initialize the feed forward network                        #
        #   Task 13: Initialize the dropout layer (torch.nn implementation)    #
        #                                                                      #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self,
                inputs: torch.Tensor) -> torch.Tensor:
        """

        Args:
            inputs: Inputs to the Feed Forward Network

        Shape:
            - inputs: (batch_size, sequence_length_queries, d_model)
            - outputs: (batch_size, sequence_length_queries, d_model)
        """
        outputs = None

        ########################################################################
        # TODO:                                                                #
        #   Task 5: Implement forward pass of feed forward layer               #
        #   Task 13: Pass the output through a dropout layer as a final step   #
        #                                                                      #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs