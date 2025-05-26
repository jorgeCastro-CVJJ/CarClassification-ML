## Clasificación de Imágenes de Carros Usando CNN
## 🚗 Clasificación de Imágenes de Carros
Este proyecto se enfoca en el desarrollo de un modelo de aprendizaje automático capaz de clasificar imágenes de diferentes marcas de carros utilizando Redes Neuronales Convolucionales (CNN).

### Selección del Dataset
Dado que se ha trabajado principalmente con datasets de imágenes en clase, se optó por utilizar un dataset de imágenes de carros.

El dataset seleccionado es [Car Images Classification using CNN](https://www.kaggle.com/code/kshitij192/car-images-classification-using-cnn/notebook). de Kaggle. La estructura del dataset es la siguiente:
```
Cars Dataset
├── test
│   ├── Audi (199 imágenes)
│   ├── Hyundai Creta (67 imágenes)
│   ├── Mahindra Scorpio (75 imágenes)
│   ├── Rolls Royce (74 imágenes)
│   ├── Swift (102 imágenes)
│   ├── Tata Safari (106 imágenes)
│   └── Toyota Innova (190 imágenes)
|
└── train
    ├── Audi (814 imágenes)
    ├── Hyundai Creta (271 imágenes)
    ├── Mahindra Scorpio (316 imágenes)
    ├── Rolls Royce (311 imágenes)
    ├── Swift (424 imágenes)
    ├── Tata Safari (441 imágenes)
    └── Toyota Innova (775 imágenes)
```

Las imágenes del dataset tienen una dimensión de 128x128 píxeles.

### Aumento de Datos
Se están considerando técnicas de aumento de datos para cumplir con el objetivo de alcanzar una proporción aproximada de 80% para el conjunto de entrenamiento y 20% para el conjunto de prueba (aproximadamente 820 imágenes de entrenamiento y 200 imágenes de prueba por clase). El dataset ya viene dividido en entrenamiento y prueba.

#### Técnicas de Aumento de Datos

Las técnicas de aumento de datos consideradas, implementadas con la clase ImageDataGenerator de Keras, son:

- **Reescalamiento:** Los valores de los píxeles se dividen por 255 para normalizarlos al rango [0, 1]. Esto es importante para que el modelo pueda procesar los datos de manera más eficiente.

- **Rotación aleatoria:** Se rotan las imágenes hasta 10 grados aleatoriamente. Esto ayuda al modelo a aprender que los objetos pueden aparecer en diferentes orientaciones.

- **Desplazamiento horizontal aleatorio:** Se desplazan las imágenes horizontalmente hasta 0.1 de su ancho. Esto simula ligeros movimientos del objeto en la imagen.

- **Desplazamiento vertical aleatorio:** Se desplazan las imágenes verticalmente hasta 0.1 de su altura. Similar al desplazamiento horizontal, esto ayuda al modelo a ser más robusto a los movimientos del objeto.

- **Corte por cizalladura o Shear:** Se aplica una transformación de cizalladura con un factor de 0.25. Esto distorsiona la forma de la imagen, simulando una perspectiva diferente.

- **Zoom aleatorio:** Se aplica un zoom a las imágenes hasta un factor de 0.3. Esto simula objetos que aparecen más cerca o más lejos de la cámara.

- **Volteo horizontal aleatorio:** Se voltean las imágenes horizontalmente de forma aleatoria. Esto ayuda al modelo a aprender que los objetos pueden aparecer en ambas direcciones.

Los parámetros de aumento de datos se definieron de la siguiente manera:
```
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.25,
    zoom_range=0.3,
    horizontal_flip=True
)
```

### Arquitectura del Modelo
La arquitectura del modelo se basa en una Red Neuronal Convolucional (CNN) con las siguientes capas:

1. **Capa de Convolución 2D:** 32 filtros, tamaño de kernel (3, 3), activación ReLU.
2. **Capa de MaxPooling 2D:** Tamaño de pool (2, 2).
3. **Capa de Convolución 2D:** 64 filtros, tamaño de kernel (3, 3), activación ReLU.
4. **Capa de MaxPooling 2D:** Tamaño de pool (2, 2).
5. **Capa de Dropout:** Tasa de dropout del 0.2. 
6. **Capa de Convolución 2D:** 128 filtros, tamaño de kernel (3, 3), activación ReLU.
7. **Capa de MaxPooling 2D:** Tamaño de pool (2, 2).
8. **Capa de Flatten:** Aplana la salida de la capa anterior.
9. **Capa Densa:** 512 unidades, activación ReLU.
10. **Capa de Dropout:** Tasa de dropout del 0.5.
11. **Capa Densa de Salida:** 7 unidades (una por cada clase de carro), activación softmax.

#### Capa de Convolución
En este modelo, la primera capa de convolución utiliza 32 filtros con un tamaño de kernel de (3, 3) y la función de activación ReLU. Esta configuración permite que el modelo extraiga características fundamentales de las imágenes, como bordes y texturas, desde el inicio del procesamiento.

#### Capa de MaxPooling 2D
A continuación, la capa de MaxPooling 2D opera sobre las características extraídas, utilizando un tamaño de pooling de (2, 2). Así, esta capa reduce la dimensionalidad de los datos generados por la convolución y mantiene las características más importantes, lo que ayuda a hacer el modelo más eficiente y menos susceptible al sobreajuste.


#### Capa de Dropout
En este caso específico, se han incorporado dos capas de Dropout: la primera aplica una tasa de 20% (0.2) justo después de la capa de convolución, y la segunda utiliza una tasa del 50% antes de la capa de salida. Estas capas apagan aleatoriamente un porcentaje de las neuronas en cada paso de entrenamiento. De esta forma, el modelo evita depender demasiado de ciertas neuronas y mejora su capacidad para generalizar a nuevos datos.

#### Capa Dense 
Para finalizar, el modelo utiliza una capa densa con 512 unidades y activación ReLU. Esta capa combina las características que han sido extraídas y procesadas en las etapas anteriores. Dado que es una capa totalmente conectada, cada neurona enlaza con todas las neuronas de la capa anterior, lo que le permite identificar relaciones complejas entre las características aprendidas.

En la salida, se emplea otra capa densa, ahora con 7 unidades (una por cada clase de carro) y activación softmax. Esta última capa convierte la información procesada en probabilidades, indicando la probabilidad de que una imagen corresponda a cada una de las clases posibles en este modelo.

##### Activación ReLU
La función ReLU (Rectified Linear Unit) toma los valores de entrada y deja pasar únicamente los positivos, mientras que los negativos los convierte en cero. Esto permite que la red neuronal sea más rápida de entrenar y ayuda a detectar patrones complejos, ya que añade una “no linealidad” necesaria para el Deep Learning.

##### Activación Softmax
La función Softmax se utiliza en la capa de salida del modelo. Recibe varios valores numéricos y los transforma en probabilidades que suman 1. Así, el modelo puede indicar no solo cuál clase predice, sino también cuánta confianza tiene en cada una de las posibles opciones, facilitando la interpretación de los resultados en tareas de clasificación.



## Compilación del Modelo
El modelo se compila utilizando el optimizador Adam, la función de pérdida categorical_crossentropy y la métrica accuracy. Esto permite que el modelo aprenda a clasificar las imágenes en las diferentes clases de carros.

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Papers Revisados

[1] M. Aamir et al, "An Optimized Architecture of Image Classification Using Convolutional Neural Network," International Journal of Image, Graphics and Signal Processing, vol. 10, (10), pp. 30, 2019. Available: https://www.proquest.com/scholarly-journals/optimized-architecture-image-classification-using/docview/2350539949/se-2. DOI: https://doi.org/10.5815/ijigsp.2019.10.05.

[1] A. Manna et al, "Bird Image Classification using Convolutional Neural Network Transfer Learning Architectures," International Journal of Advanced Computer Science and Applications, vol. 14, (3), 2023. Available: https://www.proquest.com/scholarly-journals/bird-image-classification-using-convolutional/docview/2807222514/se-2. DOI: https://doi.org/10.14569/IJACSA.2023.0140397.