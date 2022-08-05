from re import A


print(A)
def describe(x):
    print(f'x type is{type(x)}')
    print(f'x size is {x.shape}')
    print('x is', x)

import torch
import spacy
x = torch.empty(3 )
describe(x)