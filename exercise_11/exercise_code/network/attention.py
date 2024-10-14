from torch import nn
import torch
from ..network import SCORE_SAVER

class ScaledDotAttention(nn.Module):

    def __init__(self,
                 d_k,
                 dropout: float = 0.0):
        """

        Args:
            d_k: Dimension of Keys and Queries
            dropout: Dropout probability
        """
        super().__init__()
        self.d_k = d_k

        self.softmax = None
        self.dropout = None

        ########################################################################
        # TODO:                                                                #
        #   Task 2: Initialize the softmax layer (torch.nn implementation)     #
        #                                                                      #           
        ########################################################################


        self.softmax=torch.nn.Softmax(dim=-1)
        self.dropout=nn.Dropout(dropout)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the scaled dot attention given query, key and value inputs. Stores the scores in SCORE_SAVER for
        visualization

        Args:
            q: Query Inputs
            k: Key Inputs
            v: Value Inputs
            mask: Optional Causal or Padding Boolean Mask

        Shape:
            - q: (*, sequence_length_queries, d_model)
            - k: (*, sequence_length_keys, d_model)
            - v: (*, sequence_length_keys, d_model)
            - mask: (*, sequence_length_queries, sequence_length_keys)
            - outputs: (*, sequence_length_queries, d_v)
        """
        scores = None
        outputs = None

        ########################################################################
        # TODO:                                                                #
        #   Task 2:                                                            #
        #       - Calculate the scores using the queries and keys              #
        #       - Normalise the scores using the softmax function              #
        #       - Compute the updated embeddings and return the output         #
        #   Task 8:                                                            #
        #       - Add a negative infinity mask if a mask is given              #
        #   Task 13:                                                           #
        #       - Add dropout to the scores right BEFORE the final outputs     #
        #         (scores * V) are calculated                                  #
        #                                                                      #
        # Hint 2:                                                              #
        #       - torch.transpose(x, dim_1, dim_2) swaps the dimensions dim_1  #
        #         and dim_2 of the tensor x!                                   #
        #       - Later we will insert more dimensions into *, so how could    #
        #         index these dimensions to always get the right ones?         #
        #       - Also dont forget to scale the scores as discussed!           #
        # Hint 8:                                                              #
        #       - Have a look at Tensor.masked_fill_() or use torch.where()    #
        ########################################################################

        intermediate_result=q@torch.transpose(k,-1,-2)/k.shape[-1]**0.5
        if mask is not None:
            intermediate_result.masked_fill_(~mask,-torch.inf)
        scores=self.softmax(intermediate_result)
        scores=self.dropout(scores)
        outputs=scores@v

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        SCORE_SAVER.save(scores)

        return outputs
