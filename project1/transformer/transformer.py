from ctypes import Union
from typing import Callable, Optional
from torch.nn import Module, functional as F, Linear, Dropout, ModuleList
import torch
from torch import Tensor

x = Union[str, Callable[[Tensor], Tensor]] 
print(x)
exit()


__all__ = ['Transformer', 'TransformerEncoder', 'TransformerDecoder',
 'TransformerEncoderLayer', 'TransformerDecoderLayer']

# class Transformer(Module):
#     def __init__(self, d_model:int=512, nhead:int=8, num_encoder_layers:int=6,
#                 num_decoder_layers:int=6, dim_feedforward:int=2048, dropout=0.1,
#                 activation:Union[str, Callable[[Tensor], Tensor]] = F.relu,
#                 customer_encoder:Optional) -> None:
#         super().__init__()

