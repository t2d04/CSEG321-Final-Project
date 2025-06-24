from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    """
    TODO: Implement this helper method for the forward function.
      - This function is applied after the multi-head attention layer as well as after the feed forward layer.
      - GPT-2 layer applies dropout to the transformed output of each sub-layer,
        before it is added to the sub-layer input. WE DO NOT APPLY THE LAYER NORM
        IN THIS FUNCTION.
    """
    ### YOUR CODE HERE
    
    # Project the output of the sub-layer using the corresponding dense layer
    output_projected = dense_layer(output)
    
    # Apply dropout to the projected output
    output_dropped = dropout(output_projected)
    
    # Add the original input (residual connection) to the processed output
    result = input + output_dropped
    
    return result


  def forward(self, hidden_states, attention_mask):
    """
    TODO: Implement the forward pass. Some key points to consider:
           - A multi-head attention layer (CausalSelfAttention) that computes self-attention based on masked inputs.
           - Layer normalization applied *before* the attention layer and feed-forward layer.
           - Apply dropout, residual connection, and layer normalization according to the plot in the assignment. (Use self.add)
           - A feed-forward layer that applies transformations to further refine the hidden states.
    """

    ### YOUR CODE HERE
    
    # Sublayer 1: Causal Self-Attention
    ln_output = self.attention_layer_norm(hidden_states)
    attention_output = self.self_attention(ln_output, attention_mask)
    residual_output = self.add(hidden_states, attention_output, self.attention_dense, self.attention_dropout)

    # Sublayer 2: Feed-Forward Network
    ln_output_ffn = self.out_layer_norm(residual_output)
    
    # Apply the GELU activation
    dense_output = self.interm_dense(ln_output_ffn)
    interm_output = F.gelu(dense_output, approximate='tanh')

    # Use the 'add' for the final projection, dropout, and residual connection
    final_output = self.add(residual_output, interm_output, self.out_dense, self.out_dropout)
    
    return final_output

