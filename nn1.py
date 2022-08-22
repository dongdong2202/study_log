import warnings
import os.path
import logging
import sys, codecs
import multiprocessing

import torch
from torch import nn
import numpy as np
import jieba
import pandas as pd
import jieba.posseg



import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
inp = 'corpusSegDone_1.txt'
out_model = 'corpusSegDone_1.model'
out_vector = 'corpusSegDone_1.vector'
def getWordVecs(wordList, model):
    # 返回特征向量
    name = []
    vecs = []
    for word in wordList:
        word = word.replace('\n', '')
        try:
            if word in model:
                name.append(word)
                vecs.append(model[word])
        except KeyError:
            continue
    a = pd.DataFrame(name, columns=['word'])
    b = pd.DataFrame(np.array(vecs, dtype='float'))
    return pd.concat([a, b], axis=1)
def dataPrepos(text, stopKey):
    
    l = []
    pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']
    seg = jieba.posseg.cut(text)

    for i in seg:
        print(i)
        if i.word not in l and i.word not in stopKey and i.flag in pos:
            l.append(i.word)
    return l

def buildAllWordsVecs(data, stopkey, model):
    idList, titleList, abstractList = data['id'], data['title'], data['abstract']
    for index in range(len(idList)):
        id = idList[index]
        title = titleList[index]
        abstract = abstractList[index]

        l_ti = dataPrepos(title, stopkey)
        l_ab = dataPrepos(abstract, stopkey)

        words = np.append(l_ti, l_ab)
        words = list(set(words))
        wordvecs = getWordVecs(words, model)
        data_vecs = pd.DataFrame(wordvecs)
        data_vecs.to_csv('wordvecs_'+str(id)+'.csv', index=False)
        print('document ', id, ' well done')
def main():
    dataFile = 'sampe_data.csv'
    data = pd.read_csv(dataFile)
    stopkey = [ w.strip() for w in codecs.open('stopWord.txt', 'r').readlines()]
    model = gensim.models.KeyedVectors.load_word2vec_format(out_vector, binary=False)
    buildAllWordsVecs(data, stopkey, model)
if __name__ == '__main__':
    main()
exit()
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)
    logger.info('running %s' % ' '.join(sys.argv))



    model = Word2Vec(LineSentence(inp),window=5, min_count=5, workers=multiprocessing.cpu_count())
    model.save(out_model)

    model.wv.save_word2vec_format(out_vector, binary=False)





exit()
filePath = 'a.txt'

fileSegWordDonePath = 'corpusSegDone_1.txt'

def PrintList(list):
    for i in range(len(list)):

        print(list[i])
fileTrainRead = []
with open(filePath, 'r') as fileTrainRaw:
    for line in fileTrainRaw:
        fileTrainRead.append(line)

fileTrainSeg = []
for i in range(len(fileTrainRead)):
    fileTrainSeg.append([' '.join(list(jieba.cut(fileTrainRead[i][0:-11], cut_all=False)))])
with open(fileSegWordDonePath, 'w', encoding='utf-8') as FW:
    for i in range(len(fileTrainSeg)):
        FW.write(fileTrainSeg[i][0])
        FW.write('\n')


