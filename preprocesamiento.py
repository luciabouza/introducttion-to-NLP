#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: luciabouza
@author: juanmachado
"""

'''
En este Archivo se realizarán los procedimientos para:
    - obtener el archivo de entrenamiento y preprocesarlos para tomarlo como entrada del modelo
      (Data cleaning, Analisis, vectorización)
    - preprocesamiento de los datos de validacion y test para que puedan ser entrada de la red. 
'''    

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re

def cargarEmbeddings(dataPath):
    embeddings = dict()
    f = open(dataPath, "r")
    for line in f:
    	values = line.split()
    	word = values[0]
    	coefs = np.asarray(values[1:], dtype='float32')
    	embeddings[word] = coefs
    f.close()
    return embeddings

def cleanText(text):
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)    
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)    
    
    # convert text to lowercase
    text = text.strip().lower()
    
    # replace punctuation characters with spaces
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text

def cargarEjemplos(dataPath):
    f = open(dataPath, "r")
    content = f.read()
    lines = content.split("\n")
    X_train = []
    y_train = []
    for line in lines:
        X_y = line.split("\t")
        if len(X_y) > 1:
            y_train.append(X_y[1])
            X_train.append(cleanText(X_y[0]))
    return X_train, y_train

def cargarArchivosTest(dataPath, Archivo):
    dataPathArch = dataPath + Archivo
    f = open(dataPathArch, "r")
    content = f.read()
    lines = content.split("\n")
    X_train = []
    for line in lines:
        X_y = line.split("\t")
        X_train.append(cleanText(X_y[0]))
    return X_train


def create_embedding_matrix(dataPath, word_index, embedding_dim):
    word_embeddings = cargarEmbeddings(dataPath + 'fasttext.es.300.txt')
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in word_index.items():
        embedding_vector = word_embeddings.get(word)
        embedding_matrix[i] = np.array(embedding_vector, dtype=np.float32) #[:embedding_dim]

    return embedding_matrix


def preprocesamientoTrain(dataPath, embedding_dim):
    # cargo los datos
    X_train_text, y_train_text = cargarEjemplos(dataPath + 'train.csv')
    X_val_text, y_val_text = cargarEjemplos(dataPath + 'test.csv') 
    
    # tokenizo y hago Matriz de vectores de enteros, representando las palabras de las oraciones (cada fila una oración)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train_text)
    X_train = tokenizer.texts_to_sequences(X_train_text)
    X_val = tokenizer.texts_to_sequences(X_val_text)
    vocab_size = len(tokenizer.word_index) + 1 
   
    # Dejo los vectores todos del mismo largo maxlen
    maxlen = 100
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)
    
    # paso las salidas a numpy
    y_train = np.array(y_train_text, dtype=np.float32)
    y_val = np.array(y_val_text, dtype=np.float32)
  
    # creo la mariz de embeddings precargados
    embedding_matrix = create_embedding_matrix(dataPath, tokenizer.word_index, embedding_dim)
 
    return X_train, y_train, vocab_size, embedding_matrix, tokenizer, X_val, y_val


'Esta función toma como entrada array de nombres de archivos que tienen tweets, pero no están clasificados'
'Devuelve un preprocesamiento de para cada archivo, en forma de lista'
def preprocesamientoTest(dataPath, cantParametros, ArrayParametros, tokenizer):
    ArchivosTest = list()
    for i in range(cantParametros -2):       
        # cargo los datos
        X_test_text = cargarArchivosTest(dataPath, ArrayParametros[i+2])
        
        # tokenizo y hago Matriz de vectores de enteros, representando las palabras de las oraciones (cada fila una oración)
        X_test = tokenizer.texts_to_sequences(X_test_text)
        
        # Dejo los vectores todos del mismo largo maxlen
        maxlen = 100
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
        
        ArchivosTest.append(X_test)
        
    return ArchivosTest
        
        
        
        

