import requests
import json

from nltk.tokenize import word_tokenize

# from modulku import StemNstopW as stm
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

def save(data, nama):
    with open(dir_path+'/data/'+nama, 'w') as f:#dir_path+
        json.dump(data, f)

def get_data(name):
    with open(dir_path+"/data/"+str(name), "r") as filename:#dir_path+
        return json.load(filename)

dict_ = get_data('sinonim_kateglo.json')
null = get_data('null.json')

def find_sinonim_kata(cek):
    dict_sinonim_kata = dict()
    try:
        if cek not in null:
            
            list_objek = requests.get('http://kateglo.com/api.php?format=json&phrase='+str(cek)).json()['kateglo']['relation']['s']
            sinonim_list = list()
            for _, i in list_objek.items():
                try:
                    sinonim_list.append(i['related_phrase'])
                except:
                    pass
            dict_sinonim_kata = {cek:sinonim_list}
        else:
            pass
    except:
        dict_sinonim_kata = {cek:[]}
    return dict_sinonim_kata

def find_sinonim_kalimat(kalimat):
    kalimat = word_tokenize(kalimat)
    kalimat = list (set(kalimat))
    sinonim_list=list()
    for i in kalimat:
        if i not in dict_:
            sinonim_list.append(find_sinonim_kata(i))
            null.append(i)
    save(null, "null.json")
    return sinonim_list

def reverser_all(X):
    for i in X:
        for ix, j in i.items():
            for sinonim_ in j:
                dict_.update({sinonim_:ix})

    # null = list(set(null))
    save(dict_, "sinonim_kateglo.json")
    # save(null, "null.json")
    return dict_

class QE:
    def __init__(self):
        pass

    def qe_process(self, kunci_jawaban, jawaban):
        self.corpus = reverser_all(find_sinonim_kalimat(kunci_jawaban))
    #ds
        new_kalimat = list()
        for kata in word_tokenize(jawaban):# jawaban.split():
            if kata in word_tokenize(kunci_jawaban):#kunci_jawaban.split():
                new_kalimat.append(kata)
            elif kata in self.corpus:
                if self.corpus[kata] in word_tokenize(kunci_jawaban):#kunci_jawaban.split():
                    new_kalimat.append(self.corpus[kata])
                else:
                   new_kalimat.append(kata) 
            else:
                new_kalimat.append(kata)
        return " ".join(new_kalimat)

    def delete_corpus(self):
        dict_ = {}
        self.corpus = {}
        save(dict_, "sinonim_kateglo.json")
        print("corpus dihapus", len(dict_))


