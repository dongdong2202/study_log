'''
pytorch 的激活函数继承与 torch.autograd.Function, 要实现2个静态方法
'''
import torch
class Gelu(torch.autograd.Function):

    #gelu = x*simoid(1.7x)
    #gelu' = sigmoid(1.7x) *(1 + 1.7x*(1-simoid(1.7x)))
    @staticmethod
    def forward(ctx, input):
        # 定义向前计算过程
        ctx.input = input
        m = input*torch.sigmoid(1.7*input)
        return m

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.input
        tmp = torch.sigmoid(1.7*input)
        g = grad_output*tmp*(1 + 1.7*input*(1-tmp))
        return g


# 有的时候要知道深度学习某一层的情况，如weight啥样，nan值有有几个等等， hook来修改
# hook 介入的时间是 前向计算前， 前向计算后， 反向传播之后

mode= torch.mode.lstm()
def hook(model, input):
    return input #also return None
def hook1(model, input, output):
    return output #also return NOne
def hook2(mode, grad_input, grad_output):
    return output #also return None

handle = mode.register_forward_pre_hook(hook)
handle2 = mode.register_forward_hook(hook1)
handle3 = mode.register_backward_hook(hook2)

'''
TorchScript 是Python语言的一个静态类型集合
'''


