#!/usr/bin/env python2
# -*- coding: utf-8 -*-

class Dictionary(object):

    def __init__(self, id2word, word2id):
        assert len(id2word) == len(word2id)
        self.id2word = id2word
        self.word2id = word2id
        self.check_valid()

    def __len__(self):
        """
        Returns the number of words in the dictionary.
        """
        return len(self.id2word)

    def __getitem__(self, i):
        """
        Returns the word of the specified index.
        """
        return self.id2word[i]

    def __contains__(self, w):
        """
        Returns whether a word is in the dictionary.
        """
        return w in self.word2id

    def __eq__(self, y):
        """
        Compare the dictionary with another one.
        """
        self.check_valid()
        y.check_valid()
        if len(self.id2word) != len(y):
            return False
        return all(self.id2word[i] == y[i] for i in range(len(y)))

    def check_valid(self):
        """
        Check that the dictionary is valid.
        """
        assert len(self.id2word) == len(self.word2id)
        for k, v in self.id2word.items():
            assert self.word2id[v] == k

    def index(self, word):
        """
        Returns the index of the specified word.
        """
        return self.word2id[word]
