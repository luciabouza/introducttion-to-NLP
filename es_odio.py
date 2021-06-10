#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: luciabouza
@author: juanmachado
"""
### ejemplo ejecución: #######
###python3 es_odio.py './recursos_lab/' test_file1.csv test_file2.csv test_file3.csv
###python3 es_odio.py './recursos_lab/' ./recursos_lab/test.csv


import preprocesamiento
import sys
import time
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras import layers, optimizers


# definición de red con los parametros estipulados
def Inicializar(loss, optimizer, vocab_size, embedding_dim, input_length, embedding_matrix):
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim= embedding_dim, input_length=100, weights=[embedding_matrix], trainable=True))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(96, activation='relu'))
    model.add(layers.Dropout(rate=0.1))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile( optimizer= optimizers.Adam(learning_rate=0.0002965488058019691), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# entrenamiento de la red con los datos preprocesados
def Train(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    model.fit(X_train, y_train, epochs=epochs, verbose=False, validation_data=(X_val, y_val), batch_size=batch_size, use_multiprocessing=True)
    model.summary()
    
 
# prediccion de la red con los datos preprocesados a predecir
def predict(self, X_test):
    return self.model.predict(X_test, use_multiprocessing=True)


# Constantes
embedding_dim = 300

'Inicio del tiempo'
start_time = time.time()

##################################################
######Lectura de parámetros de entrada###########
##################################################
cantParametros = len(sys.argv)
ArrayParametros = sys.argv
dataPath = ArrayParametros[1]
#dataPath = './recursos_lab/'


##################################################
######Carga y Preprocesamiento de datos###########
##################################################
"En X_train tendremos la vectorización de la frase, y en Y_train la salida"
X_train, y_train, vocab_size, embedding_matrix, tokenizer, X_val, y_val = preprocesamiento.preprocesamientoTrain(dataPath, embedding_dim)
"Será una array de arrays donde para cada elemento tendremos la vectorización de los ejemplos de dicho archivo"
ArrayDatosTestNoVistos = preprocesamiento.preprocesamientoTest(dataPath, cantParametros, ArrayParametros, tokenizer) 


##################################################
######Generación y entrenamiento de la Red########
##################################################
"Aqui usamos los parámetros hallados con keras Tuner y RandomSearch ejecutado en el Jupyter Notebook"
Red = Inicializar('binary_crossentropy', 'adam', vocab_size, embedding_dim, X_train.shape[0], embedding_matrix)
Train(Red, X_train, y_train, X_val, y_val, 300, 250)

    
##################################################
######Predicción y métricas#######################
##################################################   
 
"Predicción  y métricas en Validación (o archivo test una vez tengamos los hiperparámetros determinados)"
y_pred = Red.predict(X_val) 
loss, accuracy = Red.evaluate(X_val, y_val, verbose=False)

y_pred_bool = y_pred
y_pred_bool[y_pred_bool<0.5] = 0
y_pred_bool[y_pred_bool>=0.5] = 1
print("###### valores para conjunto validación #######")
print(classification_report(y_val , y_pred_bool))
print("Pérdida Validacion", loss)
print("accuracy general Validacion", accuracy)



"Predicción  y métricas en Test"
y_predTrain = Red.predict(X_train) 
loss, accuracy = Red.evaluate(X_train, y_train, verbose=False)

y_predT_bool = y_predTrain
y_predT_bool[y_predT_bool<0.5] = 0
y_predT_bool[y_predT_bool>=0.5] = 1
print("\n ######valores para conjunto Train #######")
print(classification_report(y_train, y_predT_bool))
print("Pérdida Train", loss)
print("accuracy general train", accuracy)


##################################################
######Predicción archivos no vistos###############
##################################################   

"Para Test de archivos no vistos"
for i in range(cantParametros -2):
    #Predicción para test_filei
    arrayPrediccion = Red.predict(ArrayDatosTestNoVistos[i])
    y_predT_bool = arrayPrediccion
    y_predT_bool[y_predT_bool<0.5] = 0
    y_predT_bool[y_predT_bool>=0.5] = 1
    
    #Genero Archivo
    num = str(i+1)
    NombreArchivoSalida = dataPath + "test_file" + num + ".out"
    with open(NombreArchivoSalida, "w") as txt_file:
        for line in y_predT_bool:
            txt_file.write( str(int(line[0])) + "\n") 

           
'Fin del tiempo'
print("Todo el procedimiento demoró %s seg \n" % (time.time() - start_time))

