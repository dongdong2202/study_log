import torch
import datasets
dataset = datasets.load_dataset('tweets_hate_speech_detection')
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

ss = SnowballStemmer('english')
sw = stopwords.words('english')
print(type(sw))