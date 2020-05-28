from typing import List
from collections import defaultdict
import numpy as np


class Vocab:

    def __init__(self, corpus: List[List]):

        word_count = defaultdict(int)

        for paragraph in corpus:
            for word in paragraph:
                word_count[word] += 1

        words = []
        counts = []
        for word, count in word_count.items():
            words.append(word)
            counts.append(count)

        self._all_count = sum(counts)
        self._words = words
        self._counts = counts
        self.freq = np.array(self._counts) / self._all_count
        self.word2idx = dict(zip(self._words, range(len(self._words))))
        self.idx2word = dict(zip(range(len(self._words)), self._words))

    # 随机采样一个单词
    def sample_word(self, encode=True):
        word = np.random.choice(self._words, p=self.freq, k=1)
        if encode:
            return self.word2idx[word]
        else:
            return word

    def encode(self, words):
        return [self.word2idx[word] for word in words]

    def decode(self, ids):
        return [self.idx2word[idx] for idx in ids]

    def __len__(self):
        return len(self._vocab)
