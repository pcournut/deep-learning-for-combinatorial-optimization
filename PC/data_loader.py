from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import itemfreq
from scipy import stats
from scipy.stats import mode
from collections import Counter

import random as rand
import math

import sys
import re

import os



num_roots=12
n_observed_states= num_roots+1 #silence


# Training set is  [ ( (key2(root),key1(scale) , key2(melody) ) ] (file transpose_data/major/title.npy).      
# Ex ( (0,1), 0 ) pour ( (C,m) , C)


def one_hot_melody(m): # convert m to a dummy variable (a keyboard)
    output=np.zeros(num_roots)
    if m!=num_roots: #silence
        output[m]=1
    return output


class TrueDataGenerator(object):

    def __init__(self,max_step=8):
        #data
        self.maj_title=[]
        for element in os.listdir('transpose_data/major/'):
            if element.endswith('.npy'):
                self.maj_title.append(element)

        self.training_seq=[]
        self.target_seq=[]

        self.max_step=max_step #length of each sequence


    def process_tab(self,title,silence_tolerance=0.5):
        data_dir = 'transpose_data/major/'
        path = os.path.join(data_dir, title)
        seq = np.load(path)
        
        nb_c_silence=0

        melody_line = []

        for c,m in seq:

            if m==-1: 
                m=12 #silence
            melody_line.append(one_hot_melody(m))

            #silent state ?
            c_root,c_scale=c
            if c_scale==-1:
                nb_c_silence+=1

        #discard tab if too much silence
        ratio_s=nb_c_silence/np.shape(seq)[0]
        if ratio_s>silence_tolerance:
            pass #too much silence
        else:
            for i in range(len(melody_line)-self.max_step):
                self.training_seq.append(np.asarray(melody_line[i:i+self.max_step]))
                #self.target_seq.append(1) 
                self.target_seq.append(np.asarray([1,0])) # TRUE LABEL

    def process_maj(self):
        print('Processing',np.shape(self.maj_title)[0],'tabs in C major...')
        for title in self.maj_title:
            self.process_tab(title)
        print('Issued',len(self.training_seq),'sequences of length',self.max_step)




class FalseDataGenerator(object):

    def __init__(self,max_step=8):
        self.training_seq=[]
        self.target_seq=[]

        self.max_length=max_step

    def build_fake_data(self,N):
        for _ in range(N):

            fake_line= []
            for i in range(self.max_length):
                fake_line.append(one_hot_melody(rand.randint(0,12)))
            
            self.training_seq.append(np.asarray(fake_line))
            self.target_seq.append(np.asarray([0,1])) # FALSE LABEL

"""
TODO: Build FalseDataGenerator(object) for random melodic sequence generation (label false)
- Create class
- Define init (just need self.training_seq, self.max_length)
- Define build_fake_data(): A fake input sequence is a random sequence of length 'self.max_length' (use one_hot_melody(k) to convert a pitch k to a dummy variable like TrueDataGenerator)
"""


if __name__ == "__main__":
    dataset = TrueDataGenerator(max_step=8)
    dataset.process_maj()
    print(dataset.training_seq[0], dataset.target_seq[0]) # this is what an input sequence should look like for the discriminator

    dataset = FalseDataGenerator(8)
    dataset.build_fake_data(1)
    print(dataset.training_seq[0], dataset.target_seq[0]) # this is what an input sequence should look like for the discriminator

    