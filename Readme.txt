Se implement� un modelo basado en redes neuronales, con keras, de Tensorflow.

################ Sobre el preprocesamiento ###################

Se utilizaron los Word-Embbeding provistos para la tarea, utilizandolos como parte de la red, en la primera capa.
Para ello se cre� la matriz de embeddings, solo con los embeddings de las palabras del corpus. 

En el preprocesamiento se reconfigraron las tweets para que puedan ser entrada de la red (num�ricos y largo fijo).
Primero se realiz� una limpieza del texto eliminando los tags HTML, ciertos caracteres no alfanum�ricos t pasando todo el texto a min�sculas.
Luego se tokeniz� el corpus de entrenamiento y se tradujo cada token a un entero correspondiente dentro del corpus (que luego ser� la entrada de la red).
Esto mismo se realiz� para el corpus de test utilizando el tokenizer obtenido en el entrenamiento.
Como la cantidad de tokens de cada tweet es variable, se hizo un pad de la secuencia a un largo m�ximo de 100.

################ Sobre la red implementada ###################

La red implementada cuenta con es una red secuencial con una capa de embeddings, luego una de pooling, siguiendo una
fully connected con funci�n de activaci�n ReLu , luego una de dropout y la �ltima capa fully connected con activaci�n Sigmoide (clasificacion binaria)

el modelo est� definido de la siguiente manera:

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 100, 300)          3885900   
_________________________________________________________________
global_max_pooling1d_1 (Glob (None, 300)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 96)                28896     
_________________________________________________________________
dropout_1 (Dropout)          (None, 96)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 97        
=================================================================
Total params: 3,914,893
Trainable params: 3,914,893
Non-trainable params: 0

Se prob� con normalizaci�n de los pesos y las salidas de las capas, Capas LSTM y CNN, sin mejorar la performance. 

Se decidi� entonces mantener la arquitectura indicada al inicio de esta secci�n. 


################ Sobre la evaluaci�n del modelo ################

Para las m�tricas de performance se utiliz� accurancy, recall, precision y F1. 
Se utiliz� la funcionalidad de classification report de scikit learn en este punto para la obtencion de metricas.

Se obtuvieron metricas para ambas clases, y tanto para el conjunto de test como validaci�n.
Se observa sobreajuste en el conjunto de entrenamiento. 



################ Sobre la b�squeda de mejores par�metros #############

Para la b�squeda de par�metros de la red se utiliz� RandomSearch con Keras Tuner. 
Se realiz� el m�dulo TunerEsOdio, para esta secci�n. La b�squeda se realiz� en funci�n de:
- cantidad de unidades de la capa densa intermedia
- learning rate del m�todo de optimizaci�n ADAM
- funci�n de activaci�n de la capa densa intermedia
- porcentaje de dropout de la capa de dropout

Los resultados de cada corrida quedan grabados, pero no se realizar� la entrega de ellos ya 
que la carpeta sobrepasa los 2 GB. Si se desea correr el Tuneo de parametros puede ejecutarse dicho script. 

Los mejores hiperpar�metros hallados fueron los siguientes, obteniendo accuracy de 64%:

96 Unidades en la primer capa densa
relu como funci�n de activaci�n en dicha capa
0.1 como porcentaje de dropout
0.0002965488058019691 como learning rate de ADAM

############### Sobre la entrada y salida est�ndard #####################

Los archivos de salida tienen el formato test_fileX.out. ej: test_file1.out
para la creaci�n delo archivo test.out se ejecuto python3 es_odio.py './recursos_lab/' test.csv
y luego se cambi� el nombre al archivo de salida test_file1.out a test.out





