 
from multiprocessing import context
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1) # 随机种子，固定后生存的随即数是可以复现的
 
 
word_to_ix = {'hello':0, 'world':1}
embeds = nn.Embedding(2,5) #字典表就2个 生存5维度的向量
lookup_tensor = torch.tensor([word_to_ix['hello']], dtype=torch.long, device='cuda:0')
 
hello_embed = embeds(lookup_tensor)


#========================以下是训练ngram模型， word2vec
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

ngrams = [
    ([test_sentence[i-j-1] for j in range(CONTEXT_SIZE)], test_sentence[i]) 
    for i in range(CONTEXT_SIZE, len(test_sentence))
]

vocab = set(test_sentence)
word_to_ix = {word:i for i, word in enumerate(vocab)}

class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size) -> None:
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim) # 词汇表集合， 输出向量维度
        self.linear1 = nn.Linear(context_size*embedding_dim, 128) # 一次处理context_size个向量
        self.linear2 = nn.Linear(128, vocab_size)
    def forward(self, inputs):

        embeds = self.embeddings(inputs).view(1,-1) # 把一次输入的多个向量拼接成1个长向量context_size*embedding_dim
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
  
        return F.log_softmax(out, dim=1)

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(8):
    total_loss =0


    for context, target in ngrams:
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        model.zero_grad()
        log_probs  = model(context_idxs)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    losses.append(total_loss)
for l in losses:
    print(l)
print(model.embeddings.weight[word_to_ix["beauty"]])

exit()

x = torch.tensor([1.,2.,3.], requires_grad=True) # 默认是False
y = torch.tensor([4.,5.,6.], requires_grad=True)
z = torch.pow(x, 2) + y
s = z.sum()
print(s)
print(s.grad_fn, s.grad)
s.backward()
print(x.grad)

x = torch.randn(2,2)
y = torch.randn(2,2)
print(x.requires_grad, y.requires_grad)
x = x.requires_grad_()
y = y.requires_grad_() # 可以改变的
z = x + y
print(z.grad_fn)
print(z.requires_grad)
nz = z.detach() # detach 可以搞掉grad
print(nz, nz.grad_fn)
 
print(x.requires_grad)
print((x**2).requires_grad)

with torch.no_grad(): # 禁止track 梯度
    print(x.requires_grad)
    print((x**2).requires_grad)