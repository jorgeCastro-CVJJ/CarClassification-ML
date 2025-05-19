## Clasificación de Imágenes de Carros Usando CNN
## 🚗 Clasificación de Imágenes de Carros
Este proyecto se enfoca en el desarrollo de un modelo de aprendizaje automático capaz de clasificar imágenes de diferentes marcas de carros utilizando Redes Neuronales Convolucionales (CNN).

### Selección del Dataset
Inicialmente, se consideraron dos datasets para este proyecto:

1. [Credit Approval](https://archive.ics.uci.edu/dataset/27/credit+approval)
2. [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

Ambos datasets, provenientes del repositorio de UCI Machine Learning, están en formato CSV y son adecuados para tareas de clasificación. Sin embargo, dado que se ha trabajado principalmente con datasets de imágenes en clase, se optó por utilizar un dataset de imágenes de carros.

El dataset seleccionado es[Car Images Classification using CNN](https://www.kaggle.com/code/kshitij192/car-images-classification-using-cnn/notebook). de Kaggle. La estructura del dataset es la siguiente:
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
└── train
|   ├── Audi (814 imágenes)
|   ├── Hyundai Creta (271 imágenes)
│   ├── Mahindra Scorpio (316 imágenes)
│   ├── Rolls Royce (311 imágenes)
│   ├── Swift (424 imágenes)
│   ├── Tata Safari (441 imágenes)
│   └── Toyota Innova (775 imágenes)
```

Las imágenes del dataset tienen una dimensión de 128x128 píxeles.

### Aumento de Datos
Se están considerando técnicas de aumento de datos para cumplir con el objetivo de alcanzar una proporción aproximada de 80% para el conjunto de entrenamiento y 20% para el conjunto de prueba (aproximadamente 820 imágenes de entrenamiento y 200 imágenes de prueba por clase). El dataset ya viene dividido en entrenamiento y prueba.

Las técnicas de aumento de datos consideradas, implementadas con la clase ImageDataGenerator de Keras, son:

- **Reescalamiento:** Los valores de los píxeles se dividen por 255 para normalizarlos al rango [0, 1]. Esto es importante para que el modelo pueda procesar los datos de manera más eficiente.

- **Rotación aleatoria:** Se rotan las imágenes hasta 10 grados aleatoriamente. Esto ayuda al modelo a aprender que los objetos pueden aparecer en diferentes orientaciones.

- **Desplazamiento horizontal aleatorio:** Se desplazan las imágenes horizontalmente hasta 0.1 de su ancho. Esto simula ligeros movimientos del objeto en la imagen.

- **Desplazamiento vertical aleatorio:** Se desplazan las imágenes verticalmente hasta 0.1 de su altura. Similar al desplazamiento horizontal, esto ayuda al modelo a ser más robusto a los movimientos del objeto.

- **Corte por cizalladura:** Se aplica una transformación de cizalladura con un factor de 0.25. Esto distorsiona la forma de la imagen, simulando una perspectiva diferente.

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
