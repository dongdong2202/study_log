from curses.ascii import isxdigit
import imp
from turtle import forward
import torch
import datasets

dataset = datasets.load_dataset('tweets_hate_speech_detection')
import re
from  nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk

ss = SnowballStemmer('english')
sw = stopwords.words('english')

def split_tokens(row):                             # STEP
    row['all_tokens'] = [ss.stem(i) for i in       # 5
                     re.split(r" +",               # 3
                     re.sub(r"[^a-z@# ]", "",      # 2
                            row['tweet'].lower())) # 1
                     if (i not in sw) and len(i)]  # 4
    return row
dataset = dataset.map(split_tokens)

from collections import Counter
counts = [i for s in dataset['train']['all_tokens'] for i in s]
counts = Counter(counts)
counts = {k:v for k, v in counts.items() if v > 10}
vocab = list(counts.keys())
n_v = len(vocab)
id2tok = dict(enumerate(vocab))
tok2id = {token:ids for ids,token in id2tok.items()}
def remove_rare_tokens(row):
    row['tokens'] = [t for t in row['all_tokens'] if t in vocab]
    return row

dataset = dataset.map(remove_rare_tokens)

def windowizer(row, wsize=3):

    doc = row['tokens'] 
    wsize = 3
    out = []  
    for i, wd in enumerate(doc):
 
        target = tok2id[wd] 
        window = [i+j for j in range(-wsize, wsize+1, 1)
                  if (i+j>=0) &
                     (i+j<len(doc)) &
                     (j!=0)]
        out+=[(target, tok2id[doc[w]]) for w in window]
    row['moving_window'] = out
    return row

dataset = dataset.map(windowizer)

from torch.utils.data import Dataset, DataLoader

class Word2VecDataset(Dataset):
    def __init__(self, dataset, vocab_size, wsize=3) -> None:
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.data = [i for s in dataset['moving_window'] for i in s]
    def __len__(self):
        return len(self.data)    
    def __getitem__(self, index):
        return self.data[index]

BATCH_SIZE = 2**14
N_LOADER_PROS = 10

dataloader = {}

for key in dataset.keys():

    dataloader = {key:DataLoader(Word2VecDataset(
        dataset[key], vocab_size=n_v),
         batch_size=BATCH_SIZE, shuffle=True, num_workers=8)}

from torch import nn
size = 10
input = 3

def one_hot_encode(input, size):
    vec = torch.zeros(size).float()
    vec[input] = 1.0
 
    return vec

ohe = one_hot_encode(input, size)
linear_layer = nn.Linear(size, 1, bias=False)
embedding_layer = nn.Embedding(size, 1)

with torch.no_grad():
    embedding_layer.weight = nn.Parameter(torch.arange(10, 
        dtype=torch.float).reshape(embedding_layer.weight.shape))

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim=embedding_size)
        self.expand = nn.Linear(embedding_size, vocab_size, bias=False)
    def forward(self, input):
        hidden = self.embed(input)
        logits = self.expand(hidden)
        return logits

    


EMBED_SIZE = 100
model = Word2Vec(n_v, EMBED_SIZE)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
LR = 3e-4
EPOCHS = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

from tqdm import tqdm  # For progress bars

progress_bar = tqdm(range(EPOCHS * len(dataloader['train'])))
running_loss = []
for epoch in range(EPOCHS):
    epoch_loss = 0
    for center, context in dataloader['train']:
        center, context = center.to(device), context.to(device)
        optimizer.zero_grad()
        logits = model(input=context)
        loss = loss_fn(logits, center)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        progress_bar.update(1)
    epoch_loss /= len(dataloader['train'])
    running_loss.append(epoch_loss)
wordvecs = model.expand.weight.cpu().detach().numpy()
tokens = ['good', 'father', 'school', 'hate']
from scipy.spatial import distance
import numpy as np

def get_distance_matrix(wordvecs, metric):
    dist_matrix = distance.squareform(distance.pdist(wordvecs, metric))
    return dist_matrix
def get_k_similar_words(word, dist_matrix, k=10):
    idx = tok2id[word]
    dists = dist_matrix[idx]
    ind = np.argpartition(dists, k)[:k+1]
    ind = ind[np.argsort(dists[ind])][1:]
    out = [(i, id2tok[i], dists[i]) for i in ind]
    return out

dmat = get_distance_matrix(wordvecs, 'cosine')
for word in tokens:
    print(word, [t[1] for t in get_k_similar_words(word, dmat)], "\n")