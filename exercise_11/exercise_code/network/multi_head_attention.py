from torch import nn
import torch    

from ..network import ScaledDotAttention

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 n_heads: int,
                 dropout: float = 0.0):
        """

        Args:
            d_model: Dimension of Embedding
            d_k: Dimension of Keys and Queries
            d_v: Dimension of Values
            n_heads: Number of Attention Heads
            dropout: Dropout probability
        """
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.weights_q = None
        self.weights_k = None
        self.weights_v = None
        self.attention = None
        self.project = None
        self.dropout = None

        ########################################################################
        # TODO:                                                                #
        #   Task 3:                                                            #
        #       -Initialize all weight layers as linear layers                 #
        #       -Initialize the ScaledDotAttention                             #
        #       -Initialize the projection layer as a linear layer             #
        #  Task 13:                                                            #
        #       -Initialize the dropout layer (torch.nn implementation)        #
        #                                                                      #
        # Hints 3:                                                             #
        #       - Instead of initializing several weight layers for each head, #
        #         you can create one large weight matrix. This speed up        #
        #         the forward pass, since we dont have to loop through all     #
        #         heads!                                                       #
        #       - All linear layers should only be a weight without a bias!    #
        ########################################################################

        
        self.weights_q = torch.nn.Linear(d_model,n_heads*d_k,bias=False)
        self.weights_k = torch.nn.Linear(d_model,n_heads*d_k,bias=False)
        self.weights_v = torch.nn.Linear(d_model,n_heads*d_v,bias=False)
        self.attention = ScaledDotAttention(d_model)
        self.project = torch.nn.Linear(n_heads*d_v,d_model,bias=False)
        self.dropout = nn.Dropout(dropout)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            q: Query Inputs
            k: Key Inputs
            v: Value Inputs
            mask: Optional Causal or Padding Mask

        Shape:
            - q: (batch_size, sequence_length_queries, d_model)
            - k: (batch_size, sequence_length_keys, d_model)
            - v: (batch_size, sequence_length_keys, d_model)
            - mask: (batch_size, sequence_length_queries, sequence_length_keys)
            - outputs: (batch_size, sequence_length_queries, d_model)
        """

        # You will need these here!
        batch_size, sequence_length_queries, _ = q.size()
        _, sequence_length_keys, _ = k.size()

        outputs = None

        ########################################################################
        # TODO:                                                                #
        #   Task 3:                                                            #
        #       - Pass q,k and v through the linear layer                      #
        #       - Split the last dimensions into n_heads and d_k od d_v        #
        #       - Swap the dimensions so that the shape matches the required   #
        #         input shapes of the ScaledDotAttention layer                 #
        #       - Pass them through the ScaledDotAttention layer               #
        #       - Swap the dimensions of the output back                       #
        #       - Combine the last two dimensions again                        #
        #       - Pass the outputs through the projection layer                #
        #   Task 8:                                                            #
        #       - If a mask is given, add an empty dimension at dim=1          #
        #       - Pass the mask to the ScaledDotAttention layer                #
        #  Task 13:                                                            #
        #       - Add dropout as a final step after the projection layer       #
        #                                                                      #
        # Hints 3:                                                             #
        #       - It helps to write down which dimensions you want to have on  #
        #         paper!                                                       #
        #       - Above the todo, we have already extracted the batch_size and #
        #         the sequence lengths for you!                                #
        #       - Use reshape() to split or combine dimensions                 #
        #       - Use transpose() again to swap dimensions                     #
        # Hints 8:                                                             #
        #       - Use unsqueeze() to add dimensions at the correct location    #
        ########################################################################

        q_out=self.weights_q(q)
        q_out=torch.reshape(q_out,(batch_size, sequence_length_queries, self.n_heads,-1))
        q_out=torch.transpose(q_out,-2,-3)
        k_out=self.weights_k(k)
        k_out=torch.reshape(k_out,(batch_size, sequence_length_keys, self.n_heads,-1))
        k_out=torch.transpose(k_out,-2,-3)
        v_out=self.weights_v(v)
        v_out=torch.reshape(v_out,(batch_size, sequence_length_keys, self.n_heads,-1))
        v_out=torch.transpose(v_out,-2,-3)
        if mask is not None:
            mask = mask.unsqueeze(1)
        attention=self.attention(q_out,k_out,v_out,mask) 
        #attention=torch.softmax(q_out@torch.transpose(k_out,-1,-2)/self.d_k**0.5,-1)@v_out
        attention=torch.transpose(attention,-2,-3)
        attention=torch.reshape(attention,(-1,sequence_length_queries,self.n_heads*self.d_v))
        outputs=self.project(attention)
        outputs=self.dropout(outputs)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs
    