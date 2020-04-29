import numpy as np
from nltk.tokenize import word_tokenize
from pyjarowinkler import distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from decimal import Decimal
import numpy as np
from collections import Counter
from math import sqrt

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming_kalimat(kalimat):
    kalimat = kalimat.lower()
    return stemmer.stem(kalimat)

def stemming_list(list_kalimat):
    hasil_stemming=list()
    for i in list_kalimat:
        hasil_stemming.append(stemming_kalimat(i))
    return hasil_stemming
	
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
factoryS = StopWordRemoverFactory()
stopword = factoryS.create_stop_word_remover()
 
# Kalimat
def stopwords_removal(kalimat):
    kalimat = kalimat.lower()
    return stopword.remove(kalimat)
	
def stopwords_list(list_kalimat):
    hasil_stopwords=list()
    for i in list_kalimat:
        hasil_stopwords.append(stopwords_removal(i))
    return hasil_stopwords

import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
import requests
import json
# from modulku import StemNstopW as stm
import qe
sim = qe.QE()



def square_rooted(x):
    return sqrt(sum([a * a for a in x]))
def cosine_similarity(x, y):
    if sum(y) == 0:
        return 0.0
    else:
        numerator = sum(a * b for a, b in zip(x, y))
        denominator = square_rooted(x) * square_rooted(y)
        hasil = numerator / float(denominator)
        return round(hasil,20)

def typo(kalimat, kata_unik, toleransi=0.95):
    kalimat = kalimat.lower()
    kalimat = word_tokenize(kalimat)
    for ix, kata in enumerate(kalimat):
        similarity = list()
        for pembanding in kata_unik:
            similarity.append(distance.get_jaro_distance(kata, pembanding, winkler=True, scaling=0.1))
        if max(similarity)>=toleransi:
            kalimat[ix]=kata_unik[similarity.index(max(similarity))]
    return " ".join(kalimat)

def get_unique_words(list_words):
    list_words_ = list()
    for i in list_words:
        list_words_+=word_tokenize(i)
    return sorted(set(list_words_))

class AES:
    def __init__(self, tf_idf=False, tf=False, binary=False,stop_words=True, ngram=True, n=2, stemming=True, toleransi=0.95, qe = False):
        self.tf_idf = tf_idf
        self.tf = tf
        self.binary = binary
        self.ngram = ngram
        self.n=n
        self.stemming = stemming
        self.toleransi = toleransi
        self.stop_words = stop_words
        self.qe = qe
        
    def train(self, kunci_jawaban, label):
        self.kata_unik = get_unique_words(kunci_jawaban)
        self.kunci_jawaban = kunci_jawaban
        # self.fitur_awal = get_unique_words(kunci_jawaban)
        
        if self.stemming:
            self.kunci_jawaban = stemming_list(kunci_jawaban)
        
        if self.stop_words: 
            self.kunci_jawaban = stopwords_list(self.kunci_jawaban)

        # self.kunci_jawaban = kunci_jawaban
        # print(self.kunci_jawaban)
        
        if self.tf_idf==True and self.ngram==True:
            print("tfidf ngram")
            vectorizer = TfidfVectorizer(ngram_range=(1, self.n)) 
            self.model_w = vectorizer.fit(self.kunci_jawaban)
            self.fitur = vectorizer.get_feature_names()
            self.X = self.model_w.transform(self.kunci_jawaban)
            self.y= np.array(label)
            
        elif self.tf==True and self.ngram==True:
            print("tf ngram")
            vectorizer = CountVectorizer(ngram_range=(1, self.n)) 
            self.model_w = vectorizer.fit(self.kunci_jawaban)
            self.fitur = vectorizer.get_feature_names()
            self.X = self.model_w.transform(self.kunci_jawaban)
            self.y= np.array(label)
            
        elif self.tf==True:
            print("tf")
            vectorizer = CountVectorizer() 
            self.model_w = vectorizer.fit(self.kunci_jawaban)
            self.fitur = vectorizer.get_feature_names()
            self.X = self.model_w.transform(self.kunci_jawaban)
            self.y= np.array(label)
            
        elif self.binary==True and self.ngram==True:
            print("bin ngram")
            vectorizer = CountVectorizer(binary=True, ngram_range=(1, self.n)) 
            self.model_w = vectorizer.fit(self.kunci_jawaban)
            self.fitur = vectorizer.get_feature_names()
            self.X = self.model_w.transform(self.kunci_jawaban)
            self.y= np.array(label)
            
        elif self.binary==True:
            print("bin")
            vectorizer = CountVectorizer(binary=True) 
            self.model_w = vectorizer.fit(self.kunci_jawaban)
            self.fitur = vectorizer.get_feature_names()
            self.X = self.model_w.transform(self.kunci_jawaban)
            self.y= np.array(label)
            
        else:
            vectorizer = CountVectorizer() 
            self.model_w = vectorizer.fit(self.kunci_jawaban)
            self.fitur = vectorizer.get_feature_names()
            self.X = self.model_w.transform(self.kunci_jawaban)
            self.y= np.array(label)
        
    def predict(self, jawaban):
        jawaban = jawaban.lower()
        self.jawaban = typo(jawaban, self.kata_unik, self.toleransi)

        if self.qe:
            # self.jawaban = qe(jawaban, self.kata_unik)
            self.jawaban = sim.qe_process(kunci_jawaban= " ".join(self.kata_unik), jawaban=self.jawaban)

        if self.stemming:
            self.jawaban = stemming_kalimat(self.jawaban)
            
        # print(self.jawaban)  
        self.weight_jawaban = self.model_w.transform([self.jawaban]).A[0]
        
        self.hasil_cosine_similarity = list()
        self.bobot = list()
        for kj in self.X.A:
            self.bobot.append([kj, self.weight_jawaban])
            self.hasil_cosine_similarity.append(cosine_similarity(kj,self.weight_jawaban))
        return self.y[self.hasil_cosine_similarity.index(max(self.hasil_cosine_similarity))]
