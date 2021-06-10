#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: luciabouza
@author: juanmachado
"""
from kerastuner import HyperModel
from  keras import layers, optimizers
from keras.models import Sequential
from kerastuner.tuners import RandomSearch
import preprocesamiento
import sys


class HyperModel(HyperModel):
    def __init__(self, vocab_size, embedding_dim, input_length, embedding_matrix):
        self.vocab_size =  vocab_size
        self.embedding_dim =  embedding_dim
        self.input_length =  input_length
        self.embedding_matrix =  embedding_matrix

    def build(self, hp):
        model = Sequential()
        
        model.add(layers.Embedding(
            input_dim= self.vocab_size, 
            output_dim= self.embedding_dim, 
            input_length=100, 
            weights=[self.embedding_matrix], 
            trainable=True))
    
        model.add(layers.GlobalMaxPooling1D())
        
        model.add(
            layers.Dense(units=hp.Int(
                    'units',
                    min_value=32,
                    max_value=512,
                    step=32,
                    default=128
                ),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'
                )
            )
        )
        
        model.add(
            layers.Dropout(rate=hp.Float(
                'dropout_2',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )
        
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer= optimizers.Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
            ),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

##################################################
######Lectura de parámetros de entrada###########
##################################################
cantParametros = len(sys.argv)
ArrayParametros = str(sys.argv)
#dataPath = ArrayParametros[0]
dataPath = './recursos_lab/'


##################################################
######Carga y Preprocesamiento de datos###########
##################################################
# Constantes
embedding_dim = 300
"En X_train tendremos la vectorización de la frase, y en Y_train la salida"
X_train, y_train, vocab_size, embedding_matrix, X_val, y_val = preprocesamiento.preprocesamientoTrain(dataPath, embedding_dim)

hypermodel = HyperModel(vocab_size, embedding_dim, X_train.shape[0], embedding_matrix)

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    seed=1,
    max_trials=1,
    executions_per_trial=1,
    directory='random_search',
    project_name='EsOdio'
)

tuner.search_space_summary()

tuner.search(X_train, y_train, epochs=50, verbose=False, validation_data=(X_val, y_val), batch_size=250, use_multiprocessing=True)

# Show a summary of the search
tuner.results_summary()

# Retrieve the best model.
best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters()[0]
best_model.summary()

units = best_hp.get("units")
dense_activation = best_hp.get("dense_activation")
dropout_2 = best_hp.get("dropout_2")
learning_rate = best_hp.get("learning_rate")
print(units)
print(dense_activation)
print(dropout_2)
print(learning_rate)

# Evaluate the best model.
loss, accuracy = best_model.evaluate(X_val, y_val)



