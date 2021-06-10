Se implementó un modelo basado en redes neuronales, con keras, de Tensorflow.

################ Sobre el preprocesamiento ###################

Se utilizaron los Word-Embbeding provistos para la tarea, utilizandolos como parte de la red, en la primera capa.
Para ello se creó la matriz de embeddings, solo con los embeddings de las palabras del corpus. 

En el preprocesamiento se reconfigraron las tweets para que puedan ser entrada de la red (numéricos y largo fijo).
Primero se realizó una limpieza del texto eliminando los tags HTML, ciertos caracteres no alfanuméricos t pasando todo el texto a minúsculas.
Luego se tokenizó el corpus de entrenamiento y se tradujo cada token a un entero correspondiente dentro del corpus (que luego será la entrada de la red).
Esto mismo se realizó para el corpus de test utilizando el tokenizer obtenido en el entrenamiento.
Como la cantidad de tokens de cada tweet es variable, se hizo un pad de la secuencia a un largo máximo de 100.

################ Sobre la red implementada ###################

La red implementada cuenta con es una red secuencial con una capa de embeddings, luego una de pooling, siguiendo una
fully connected con función de activación ReLu , luego una de dropout y la última capa fully connected con activación Sigmoide (clasificacion binaria)

el modelo está definido de la siguiente manera:

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

Se probó con normalización de los pesos y las salidas de las capas, Capas LSTM y CNN, sin mejorar la performance. 

Se decidió entonces mantener la arquitectura indicada al inicio de esta sección. 


################ Sobre la evaluación del modelo ################

Para las métricas de performance se utilizó accurancy, recall, precision y F1. 
Se utilizó la funcionalidad de classification report de scikit learn en este punto para la obtencion de metricas.

Se obtuvieron metricas para ambas clases, y tanto para el conjunto de test como validación.
Se observa sobreajuste en el conjunto de entrenamiento. 



################ Sobre la búsqueda de mejores parámetros #############

Para la búsqueda de parámetros de la red se utilizó RandomSearch con Keras Tuner. 
Se realizó el módulo TunerEsOdio, para esta sección. La búsqueda se realizó en función de:
- cantidad de unidades de la capa densa intermedia
- learning rate del método de optimización ADAM
- función de activación de la capa densa intermedia
- porcentaje de dropout de la capa de dropout

Los resultados de cada corrida quedan grabados, pero no se realizará la entrega de ellos ya 
que la carpeta sobrepasa los 2 GB. Si se desea correr el Tuneo de parametros puede ejecutarse dicho script. 

Los mejores hiperparámetros hallados fueron los siguientes, obteniendo accuracy de 64%:

96 Unidades en la primer capa densa
relu como función de activación en dicha capa
0.1 como porcentaje de dropout
0.0002965488058019691 como learning rate de ADAM

############### Sobre la entrada y salida estándard #####################

Los archivos de salida tienen el formato test_fileX.out. ej: test_file1.out
para la creación delo archivo test.out se ejecuto python3 es_odio.py './recursos_lab/' test.csv
y luego se cambió el nombre al archivo de salida test_file1.out a test.out





