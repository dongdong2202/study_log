from statistics import mode
from tkinter.messagebox import NO
from matplotlib.cbook import flatten
import numpy as np
import math
import matplotlib.pyplot as plt
from regex import P
import seaborn as sns
from sklearn.model_selection import learning_curve
from zmq import device
import torch



x = np.linspace(-math.pi, math.pi, 2000) # 2000 point in this distance
y = np.sin(x)

a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

l_rate = 1e-6

for t in range(20):
    y_pred = a + b*x+ c *x**2 + d*x**3
    loss = np.square(y_pred - y).sum()
    # if t% 100 == 99:
    #     print(t, loss)
    g_y_pred = 2.0*(y_pred - y)

    grad_a = g_y_pred.sum()
    grad_b = (g_y_pred*x).sum()
    grad_c = (g_y_pred*x**2).sum()
    grad_d = (g_y_pred*x**3).sum()

    a -= l_rate * grad_a
    b -= l_rate * grad_b
    c -= l_rate * grad_c    
    d -= l_rate * grad_d


print(a, b, c, d)

dtype = torch.float
device = torch.device('cpu')

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(2):
    y_pred = a + b*x+ c *x**2 + d*x**3
    loss = (y_pred - y).pow(2).sum()
    # if t %100 == 99 :
    #     print(t, loss.item())
    loss.backward()
    with torch.no_grad(): #两个作用：新增的tensor没有梯度，使带梯度的tensor能够进行原地运算。
        a -= learning_rate*a.grad
        b -= learning_rate*b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

p = torch.tensor([1,2,3])
xx = x.unsqueeze(-1).pow(p)
model = torch.nn.Sequential(
    torch.nn.Linear(3,1)
 ,   torch.nn.Flatten(0, 1) #前0维度保留，其余维度flatten成一维
)


loss_fn = torch.nn.MSELoss(reduction='sum')
for t in range(2000):
    y_pred = model(xx)
    loss = loss_fn(y_pred, y)
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= learning_rate * p.grad
linear_layer = model[0]

print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')


class DynamicNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))
    def forward(self, x):
        y = self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
        for exp in range(4, np.random.randint(4, 6)):
            y = y + self.e * x ** exp
        return y
    
    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + \
         {self.d.item()} x^3 + {self.e.item()} x^4 ? + {self.e.item()} x^5 ?' 

model = DynamicNet()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
for t in range(3000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(model.string())