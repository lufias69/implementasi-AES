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


import requests
import json
# from modulku import StemNstopW as stm
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

def save(data, nama):
    with open(dir_path+'/data/'+nama, 'w') as f:
    # with open('data/'+nama, 'w') as f:
        json.dump(data, f)
        
def get_value_dict(dict_):
    value_ = list()
    for _, val in dict_.items():
        value_.append(val)
    return value_

def get_data(name):
    with open(dir_path+'/data/'+str(name), "r") as filename:
        return json.load(filename)

kateglo = get_data('kateglo.json')

def fs(cek):
    try:
        list_objek = requests.get('http://kateglo.com/api.php?format=json&phrase='+str(cek)).json()['kateglo']['relation']['s']
        keranjang = dict()
        for _, isi in list_objek.items():
            try:
                keranjang.update({isi['related_phrase']:cek})
            except:
                pass
        return keranjang
    except:
        return {}

def find_sinonim(fitur):
    dict_all = get_data("sinonim_kateglo.json")
    list_kata_ada = set(get_value_dict(dict_all))
    if type(fitur) is not list:
        return "fitur harus bertype list"
    
    
    for i in fitur:
        if i not in list_kata_ada:
            x = fs(i)
            dict_all.update(x)
    save(dict_all, "sinonim_kateglo.json")
    return dict_all 

def qe(s, fitur):
    dict_sinonim = find_sinonim(fitur)
#     print(dict_sinonim)
    s = s.split()
    for ix, i in enumerate(s):
        if i not in fitur and i in dict_sinonim:
            s[ix]=dict_sinonim[i]
    return " ".join(s) 

#qe end
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
    def __init__(self, tf_idf=False, tf=False, binary=False,stop_words=True, ngram=False, n=2, stemming=True, toleransi=0.95, qe = False):
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
        # self.fitur_awal = get_unique_words(kunci_jawaban)
        
        if self.stemming:
            kunci_jawaban = stemming_list(kunci_jawaban)
        
        if self.stop_words: 
            kunci_jawaban = stopwords_list(kunci_jawaban)

        self.kunci_jawaban = kunci_jawaban
        
        if self.tf_idf==True and self.ngram==True:
            print("tfidf ngram")
            vectorizer = TfidfVectorizer(ngram_range=(1, self.n)) 
            self.model_w = vectorizer.fit(kunci_jawaban)
            self.fitur = vectorizer.get_feature_names()
            self.X = self.model_w.transform(kunci_jawaban)
            self.y= np.array(label)
            
        elif self.tf==True and self.ngram==True:
            print("tf ngram")
            vectorizer = CountVectorizer(ngram_range=(1, self.n)) 
            self.model_w = vectorizer.fit(kunci_jawaban)
            self.fitur = vectorizer.get_feature_names()
            self.X = self.model_w.transform(kunci_jawaban)
            self.y= np.array(label)
            
        elif self.tf==True:
            print("tf")
            vectorizer = CountVectorizer() 
            self.model_w = vectorizer.fit(kunci_jawaban)
            self.fitur = vectorizer.get_feature_names()
            self.X = self.model_w.transform(kunci_jawaban)
            self.y= np.array(label)
            
        elif self.binary==True and self.ngram==True:
            print("bin ngram")
            vectorizer = CountVectorizer(binary=True, ngram_range=(1, self.n)) 
            self.model_w = vectorizer.fit(kunci_jawaban)
            self.fitur = vectorizer.get_feature_names()
            self.X = self.model_w.transform(kunci_jawaban)
            self.y= np.array(label)
            
        elif self.binary==True:
            print("bin")
            vectorizer = CountVectorizer(binary=True) 
            self.model_w = vectorizer.fit(kunci_jawaban)
            self.fitur = vectorizer.get_feature_names()
            self.X = self.model_w.transform(kunci_jawaban)
            self.y= np.array(label)
            
        else:
            vectorizer = TfidfVectorizer() 
            self.model_w = vectorizer.fit(kunci_jawaban)
            self.fitur = vectorizer.get_feature_names()
            self.X = self.model_w.transform(kunci_jawaban)
            self.y= np.array(label)
        
    def predict(self, jawaban):
        jawaban = jawaban.lower()
        self.jawaban = typo(jawaban, self.kata_unik, self.toleransi)

        if self.qe:
            self.jawaban = qe(jawaban, self.kata_unik)

        if self.stemming:
            jawaban = stemming_kalimat(self.jawaban)
            
        self.weight_jawaban = self.model_w.transform([self.jawaban]).A[0]
        
        self.hasil_cosine_similarity = list()
        self.bobot = list()
        for kj in self.X.A:
            self.bobot.append([kj, self.weight_jawaban])
            self.hasil_cosine_similarity.append(cosine_similarity(kj,self.weight_jawaban))
        return self.y[self.hasil_cosine_similarity.index(max(self.hasil_cosine_similarity))]
        