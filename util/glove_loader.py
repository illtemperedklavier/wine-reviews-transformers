# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:55:41 2019

@author: alecr
"""

import numpy as np
import pickle



def getGloveVectors(vector_dimension):
    filename = "D:/Data/glovePickle/glovePickle.%s" % vector_dimension
    infile = open(filename,'rb')
    tripledict = pickle.load(infile)
    infile.close()
    word2vec = tripledict['word2vec']
    embedding = tripledict['embedding']
    idx2word = tripledict['idx2word']
    
    return word2vec, embedding, idx2word 
    

