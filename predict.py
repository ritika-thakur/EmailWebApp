import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import re
import nltk
from contractions import contractions_dict
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS
from itertools import filterfalse
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score

import pickle as pk


def text_processing(data):
        
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    
    lemmatizer = WordNetLemmatizer()
    
    def strip_titles(text):
        if "Subject: re :" in text:
            return text[13:]
        elif "Subject: news :" in text:
            return text[15:]
        else:
            return text[8:]
    
    
    
    data['text'] = data['text'].apply(lambda x: strip_titles(x))

    data['text'] = data['text'].apply(lambda x: word_tokenize(x))
    
    
    
    def normalize_tokens(list_of_tokens):
        
        return map(lambda x: x.lower(),list_of_tokens)

    
    
    data['text'] = data['text'].apply(lambda x: normalize_tokens(x))

    data['text'] = data['text'].apply(lambda x: list(x))
    
    
    
    def contracted_word_expansion(token):
        if token in contractions_dict.keys():
            return contractions_dict[token]
        else:
            return token
    
    def contractions_expansion(list_of_tokens):
        return map(contracted_word_expansion,list_of_tokens)

    
    
    data['text'] = data['text'].apply(lambda x: contractions_expansion(x))

    data['text'] = data['text'].apply(lambda x: list(x))
    
    
    
    
    regex = r'^@[a-zA-z0-9]|^#[a-zA-Z0-9]|\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*|\W+|\d+|<("[^"]*"|\'[^\']*\'|[^\'">])*>|_+|[^\u0000-\u007f]+'

    
    def waste_word_or_not(token):
        return re.search(regex,token)

    def filter_waste_words(list_of_tokens):
        return filterfalse(waste_word_or_not,list_of_tokens)

    
    
    data['text'] = data['text'].apply(lambda x: filter_waste_words(x))

    data['text'] = data['text'].apply(lambda x: list(x))

    
    
    def split(list_of_tokens):
        return map(lambda x: re.split(regex,x)[0],list_of_tokens)

    
    
    data['text'] = data['text'].apply(lambda x: split(x))

    data['text'] = data['text'].apply(lambda x: list(x))
    
    
    
    en_stop_words = list(set(stopwords.words('english')).union(set(STOP_WORDS)))

    
    def is_stopword(token):
        return not(token in en_stop_words or re.search(r'\b\w\b|[^\u0000-\u007f]+|_+|\W+',token))

    def stopwords_removal(list_of_tokens):
        return filter(is_stopword,list_of_tokens)

    
    
    data['text'] = data['text'].apply(lambda x: stopwords_removal(x))

    data['text'] = data['text'].apply(lambda x: list(x))
    
    
    def get_wnet_pos_tag(treebank_tag):
    
        if treebank_tag[1].startswith('J'):
            return (treebank_tag[0],wordnet.ADJ)
    
        elif treebank_tag[1].startswith('V'):
            return (treebank_tag[0],wordnet.VERB)
    
        elif treebank_tag[1].startswith('N'):
            return (treebank_tag[0],wordnet.NOUN)
    
        elif treebank_tag[1].startswith('R'):
            return (treebank_tag[0],wordnet.ADV)
    
        else:
            (treebank_tag[0],wordnet.NOUN)
        
    def get_pos_tag(list_of_tokens):
        return map(get_wnet_pos_tag,pos_tag(list_of_tokens))

    
    
    
    data['text'] = data['text'].apply(lambda x: get_pos_tag(x))

    data['text'] = data['text'].apply(lambda x: list(x))

    
    def token_lemmatization(token_pos_tuple):
        if token_pos_tuple == None:
            return ""
        else:
            return lemmatizer.lemmatize(word=token_pos_tuple[0],pos=token_pos_tuple[1])
    
    def lemmatization(list_of_tokens):
        if len(list_of_tokens) > 0:
            return map(lambda x: token_lemmatization(x),list_of_tokens)
    
    
    
    data['text'] = data['text'].apply(lambda x: lemmatization(x))

    data['text'] = data['text'].apply(lambda x: list(x))
    
    
    vocab = set()
    
    
    for list_of_tokens in data['text']:
        vocab = vocab.union(set(list_of_tokens))
    
    vocab = list(vocab)

    
    while('' in vocab):
        vocab.remove('')
    
    vocab_dict = dict(zip(vocab,list(range(0,len(vocab)))))
    
    
    
    def join_tokens(list_of_tokens):
        return " ".join(list_of_tokens)

    
    
    data['text'] = data['text'].apply(lambda x: join_tokens(x))
    
    return data
    



