from collections import Counter, defaultdict
from sys import maxsize
from tkinter.messagebox import NO

class Vocab(object):
    UNK = '<unk>'
    # unk 是稀有词的替代符号， pad是首次的替代，编号是0 sos是开始， eos是结束
    def __init__(self, counter, max_size=None, min_freq=1, 
                specials=['<unk>', '<pad>'], specials_first=True) -> None:
        # max_size is a full capcity of vocabulary, 
        # min_freq the minest freq of words
        # self.itos index dict, from index to word
        # self.stos word dict ,from word to index
        self.freqs = counter
        min_freq = max(min_freq, 1)

        self.itos = list()
        self.unk_index = None
        if specials_first:
            self.itos = list(specials)
            max_size = None if max_size is None else max_size + len(specials)

        for tok in specials:
            del counter[tok]
        words_and_frequencies = sorted(counter.items(), key=lambda x:x[0])
        words_and_frequencies.sort(key=lambda x:x[1])

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)
        
        if Vocab.UNK in specials:
            unk_index = specials.index(Vocab.UNK)
            self.unk_index = unk_index if specials_first else len(self.itos) + unk_index
            # self.stoi = defaultdict(list(self.unk_index))
            self.stoi = defaultdict()
        else:
            self.stoi = defaultdict()

        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})
        print(self.stoi)
        print(self.itos)

def build_vocab_from_iterator(iterator):
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    word_vocab = Vocab(counter)
    return word_vocab

x = build_vocab_from_iterator(['you are good', 'you are bad'])
print(x)


