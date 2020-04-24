#!/usr/bin/env python3
# coding: utf-8
# File: sentence_similarity.py
# Date: 19-08-17

import gensim
import numpy as np
class SimilarityCompute:
    def __init__(self):
        #self.embedding_file = '../config/token_vector.bin'
        self.embedding_file = './token_vector.bin'
        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.embedding_file, binary=False)

    def get_wordvector(self, word):
        try:
            return self.model[word]
        except:
            return np.zeros(200)

    def similarity_cosine(self, word_list1,word_list2):
        simalrity = 0
        vector1 = np.zeros(200)
        for word in word_list1:
            vector1 += self.get_wordvector(word)

        vector1 = vector1/len(word_list1)
        vector2 = np.zeros(200)

        for word in word_list2:
            vector2 += self.get_wordvector(word)

        vector2 = vector2/len(word_list2)
        cos1 = np.sum(vector1*vector2)
        cos21 = np.sqrt(sum(vector1**2))
        cos22 = np.sqrt(sum(vector2**2))
        similarity = cos1/float(cos21*cos22)
        return similarity

def test():
    newsim = SimilarityCompute()
    text1 = "明天广州天气怎么样？"
    text2 = "广州的明天的气候如何？"
    similarity=newsim.similarity_cosine(text1,text2)
    print("text1","text2",similarity)

test()