import os
import pandas as pd
import numpy as np
import re
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

twitter_data_path = "C:\\Users\Rodrigo\Documents\Projetos\Twitter Python\data"
emoticon_list = {':))': 'positive_emoticon', ':)': 'positive_emoticon',
                 ':D': 'positive_emoticon', ':(': 'negative_emoticon',
                 ':((': 'negative_emoticon', '8)': 'neutral_emoticon',
                 'xD': 'positive_emoticon', ':-)': 'positive_emoticon'}
noisy_words = ['.', '?', '!', ':', ',', ';', '(', ')', '-']
std_list = {'eh': 'é', 'vc': 'você', 'vcs': 'vocês','tb': 'também',
            'tbm': 'também', 'obg': 'obrigado', 'gnt': 'gente',
            'q': 'que', 'n': 'não', 'ñ': 'não', 'cmg': 'comigo', 'p': 'para',
            'ta': 'está', 'to': 'estou', 'vdd': 'verdade',
            'pa': 'para', 'pq': 'por que', 'mt': 'muito',
            'blz':'beleza', 'tava': 'estava', 'tô': 'estou',
            'dnv':'de novo', 'so': 'só', 'qto': 'quanto'}

def _replace_emotes(data):
    ls=[]
    for linha in data:
        for element in emoticon_list:
            linha = linha.replace(element, emoticon_list[element])
        ls.append(linha)
    return ls

def _remover_regex(data, pattern):
    ls = []
    for linha in data:
        matches =  re.findall(pattern, linha)
        for m in matches:
            linha = linha.replace(m, '')
        ls.append(linha)
    return ls

def _tokenize(data):
    ls =[]
    for linha in data:
        tokens = nltk.tokenize.word_tokenize(linha)
        ls.append(tokens)
    return ls

def _untokenize_text(tokens):
    ls = []

    for tk_line in tokens:
        new_line = ''

        for word in tk_line:
            new_line += word + ' '

        ls.append(new_line)

    return ls

def _estandarizacao(data, std_list):
    ls=[]
    for linha in data:
        novos_tokens = []

        for palavra in linha:
            if palavra.lower() in std_list:
                palavra = std_list[palavra.lower()]
            else:
                palavra = palavra.lower()
            novos_tokens.append(palavra)
        ls.append(novos_tokens)
    return ls

def _get_ptbr_stopwords():
    stopwords_list = nltk.corpus.stopwords.words('portuguese')
    stopwords_list.append('...')
    stopwords_list.append('que')
    stopwords_list.append('tão')
    stopwords_list.append('«')
    stopwords_list.append('➔')
    stopwords_list.append('|')
    stopwords_list.append('»')
    stopwords_list.append('uai')
    return stopwords_list

def _remover_stopwords(tokens, stopword_list):
    ls = []

    for tk_line in tokens:
        new_tokens = []

        for word in tk_line:
            if word.lower() not in stopword_list:
                new_tokens.append(word)

        ls.append(new_tokens)

    return ls

def _apply_stemmer(tokens):
    ls = []
    stemmer = nltk.stem.RSLPStemmer()

    for tk_line in tokens:
        new_tokens = []

        for word in tk_line:
            word = str(stemmer.stem(word))
            new_tokens.append(word)

        ls.append(new_tokens)

    return ls

def _preprocessamento_dados(data):
    regex_mentions = "@[w]*"
    regex_hashtags = "#[w]*"
    regex1 = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    regex2 = re.compile('www?.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    ls = _remover_regex(data, regex_mentions)
    ls = _remover_regex(ls, regex_hashtags)
    ls = _remover_regex(ls, regex1)
    ls = _remover_regex(ls, regex2)
    ls = _replace_emotes(ls)
    ls = _tokenize(ls)
    ls = _estandarizacao(ls,std_list)
    ls = _remover_stopwords(ls, _get_ptbr_stopwords())
    ls = _apply_stemmer(ls)
    ls = _untokenize_text(ls)

    return ls

def _get_text_cloud(tokens):
    text = ''

    for tk_line in tokens:
        new_tokens = []

        for word in tk_line:
            text += word + ' '

    return text

def _print_wordCloud(data):
    text_cloud = _get_text_cloud(data)
    word_cloud = WordCloud(max_font_size=100, width=1520, height=580)
    word_cloud.generate(text_cloud)
    plt.figure(figsize=(16,9))
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.show()

def pre_processamento(input_file):
    input_data = pd.read_csv(input_file, delimiter=";")

    x = _preprocessamento_dados(input_data['tweet_text'].values)
    y = input_data['sentiment'].map({0:'Negative', 1:'Positive', 2:'Neutral'}).values

    vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    x_vect = vect.fit_transform(x)
    x_vect = tfidf_transformer.fit_transform(x_vect)

    return x_vect,y