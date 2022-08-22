from collections import OrderedDict
from copy import copy
from math import ldexp
from operator import le
from typing import Counter
import numpy as np 
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

docs = ['The faster Harry got to the store, the faster and the faster Harry would get home.']
docs.append('Harry is hairy and faster than Jill')
docs.append('Jill is not as hairy as Harry.')

# doc_tokens = []
# for doc in docs:
#     doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))]
# all = sum(doc_tokens, [])
 
# lexicon = sorted(set(all))
# zero_vector = OrderedDict((token, 0) for token in lexicon)
# doc_vectors = []
# for doc in docs:
#     vec = copy(zero_vector)
#     tokens = tokenizer.tokenize(doc.lower())
#     token_counts = Counter(tokens)
#     for key, value in token_counts.items():
#         vec[key] = value/len(lexicon)
#     doc_vectors.append(vec)

from  nlpia.data.loaders  import get_data

kite_intro = get_data('kite_text')
''.join(kite_intro)
kite_intro = str(kite_intro).lower()
intro_token = tokenizer.tokenize(kite_intro)

kite_history = get_data('kite_history')
''.join(kite_history)
kite_history = str(kite_history).lower()
history_token = tokenizer.tokenize(kite_history)
print(len(history_token), len(intro_token))
intro_tf = {}
hitory_tf  = {}
intro_counts = Counter(intro_token)
