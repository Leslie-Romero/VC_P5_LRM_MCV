# Práctica 5 de VC (Visión por Computador)

## Autores

Leslie Liu Romero Martín
<br>
María Cabrera Vérgez

## Tareas realizadas

Tras mostrar opciones para la detección y extracción de información de caras humanas con deepface, la tarea a entregar consiste en proponer dos escenarios de aplicación y desarrollar dos prototipos de temática libre que provoquen reacciones a partir de la información extraída del rostro. Uno de los prototipos deberá incluir el uso de algún modelo entrenado por ustedes para la extracción de información biometríca, similar al ejemplo del género planteado durante la práctica pero con diferente aplicación (emociones, raza, edad...). El otro es de temática completamente libre.

Los detectores proporcionan información del rostro, y de sus elementos faciales. Ideas inmediatas pueden ser filtros, aunque no hay limitaciones en este sentido. La entrega debe venir acompañada de un gif animado o vídeo de un máximo de 30 segundos con momentos seleccionados de las propuestas. Las propuestas se utilizarán para una posterior votación y elección de las mejores entre el grupo.

## Referencias (dataset)

```bibtex
@inproceedings{karkkainenfairface,
  title={FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation},
  author={Karkkainen, Kimmo and Joo, Jungseock},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2021},
  pages={1548--1558}
}
```

## Instalación

``` python
import os
import numpy as np
import cv2

from matplotlib import pyplot as plt
import matplotlib 

from time import time
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from skimage.feature import hog

from deepface import DeepFace
import json
import random
```

## Tareas

### Minijuego por detección del movimiento de la cara

El segundo filtro era de temática libre, siempre y cuando mantenga el uso del reconocimiento biométrico para realizar un filtro. Para ello, como la mayoría de modelos para la detección de datos biométricos eran muy pesados, se usó la detección de la cara y su movimiento.

La idea general del filtro es sencilla. Similar a filtros existentes, se tiene escrita una pregunta sobre la cabeza del usuario. Por medio de una acción, se es capaz de dar una respuesta, siendo esta dada por válida o incorrecta. Además, se contará con una puntuación para saber qué tal se está haciendo. Son un total de 50 preguntas.

En primer lugar, lo que se hace es captar la cámara del ordenador para poder usar el reconocimiento facial. Se preparan las diferentes variables que serán necesarias en la elaboración del filtro, como una para ir controlando las preguntas, para el puntaje, etc. 

``` python
cap = cv2.VideoCapture(0)


x0 = 0

x_offset = 0

is_selected = None

new_question = False

answer = None

delay = 0

index_pregunta = 0

score = 0
``` 

#### PREGUNTAS, SEÑALO ESTE APARTADO PARA TI LESLIE (para el tema del json, recordarme que eso iba junto acá)

Se realiza el mismo proceso que siempre para ir tomando los frames. Como la cámara se refleja como un espejo, se debe de girar horizontalmente, evitando ese efecto.

``` python
while True:

    # Read frame

    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)  # Flip frame horizontally for mirror effect

    if not ret:

        break
```

Se usará deepface para el reconocimiento de la cara. Se extrae la cara detectada del frame y se usa opencv, esto debido a que es el detector que mejor ha funcionado en las pruebas realizadas.

Si se ha llegado a la pregunta 50 y se responde, se imprime un mensaje avisando de ello para que el usuario pare el programa. Además, se muestra el resultado en porcentaje de la cantidad de preguntas contestadas. 

``` python
if index_pregunta >= len(preguntas):

            text("Se han terminado las preguntas.", frame, 140, (0,0,0))
``` 

Para poder mostrar ese texto por la pantalla, se creó una función. A esta se le debe de pasar el mensaje, el frame en el que se va a dejar el mensaje, la coordenada del centro en y y el color que se va a usar.

Con estos datos, text() usa un fuente por defecto, escribe mensajes con un grosor de 1. El ancho máximo del texto dependerá del contenido mostrado y se elige su escala. Posteriormente se cogen los tamaños del texto cuando se fuese a escribir para ir controlando como es según la escala que se tiene. Si el ancho que se da es menor que la escala dada, se reduce esta última. Se va ir reduciendo hasta que sea menor que el ancho máximo.

Se va a centrar el texto y se escribirá por pantalla.

``` python
def text(content, frame, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    max_width = frame.shape[1] - 40  
    scale = 0.7
 
    (text_width, text_height), baseline = cv2.getTextSize(content, font, scale, thickness)
    while text_width > max_width and scale > 0.1:
        scale -= 0.01
        (text_width, text_height), _ = cv2.getTextSize(content, font, scale, thickness)
 
    frame_width = frame.shape[1]
    x_text = (frame_width - text_width) // 2
    y_text = y
    cv2.putText(frame, content, (x_text, y_text), font, scale, color, thickness)
 
``` 

Para que sea más cómodo para el usuario, se dibuja en la mitad de la pantalla una línea azul que sirve como referencia para el lugar al que debe volver tras cada pregunta.

``` python
        cv2.line(frame, (frame.shape[1]//2, 0), (frame.shape[1]//2, frame.shape[0]), (255, 0, 0), 1)
```

Se debe ir controlando el paso de las preguntas. Esto ayuda a que no se quede guardada la última pregunta procesada en la última ejecución y a que se pase de una pregunta a otra después de responder. Las preguntas se mostrarán sobre un área blanca en la parte superior.

``` python
        if pregunta is None:
            pregunta = preguntas[index_pregunta]

        if new_question:
            index_pregunta += 1
            new_question = False

        if index_pregunta < len(preguntas):
            pregunta = preguntas[index_pregunta]
        else:
            pregunta = None  
```

A ambos lados de la pantalla, en las esquinas de abajo, se mostrarán dos opciones: verdadero o falso. Así el usuario sabe hacia donde moverse para contestar. Las opciones parecerán dos botones, pues tendrán unos rectángulos negros por debajo. 

``` python
        # Show question text
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        max_width = frame.shape[1] - 40  
        scale = 0.7
 
        left_text = "verdadero"
        right_text = "falso"

        (p_width, p_height), p_baseline = cv2.getTextSize(pregunta["pregunta"], font, scale, thickness)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], p_height + p_baseline + 50), (255, 255, 255), -1)
 
        text(pregunta["pregunta"], frame, 50, (0,0,0))

        (left_width, left_height), left_baseline = cv2.getTextSize(left_text, font, scale, thickness)
        (right_width, right_height), right_baseline = cv2.getTextSize(right_text, font, scale, thickness)
        
        # Show True text
        padding = 10
        x_left = 10
        (left_width, left_height), left_baseline = cv2.getTextSize(left_text, font, scale, thickness)
        top_left = frame.shape[0] - 20 - left_height - padding // 2
        bottom_left = frame.shape[0] - 20 + left_baseline + padding // 2
        cv2.rectangle(frame, (x_left - padding, top_left), (x_left + left_width + padding, bottom_left), (0, 0, 0), -1)
        cv2.putText(frame, left_text, (x_left, frame.shape[0] - 20), font, scale, (0, 255, 0), thickness)
 
        # Show False text
        x_right = frame.shape[1] - right_width - 10
        (right_width, right_height), right_baseline = cv2.getTextSize(right_text, font, scale, thickness)
        top_right = frame.shape[0] - 20 - right_height - padding // 2
        bottom_right = frame.shape[0] - 20 + right_baseline + padding // 2
        cv2.rectangle(frame, (x_right - padding, top_right), (x_right + right_width + padding, bottom_right), (0, 0, 0), -1)
        cv2.putText(frame, right_text, (x_right, frame.shape[0] - 20), font, scale, (0, 0, 255), thickness)
``` 

Como se va guardando la puntuación que se va consiguiendo, se imprime en una esquina de la pantalla para que el usuario sea capaz de ver cuantos puntos lleva en cada momento.

``` python
        score_text = f"Puntos: {score}"
        cv2.putText(frame, score_text, (x_left, 100), font, scale, (0, 0, 0), 2)
```

Tras contestar cada pregunta, sale un mensaje avisando de si la respuesta es correcta o, por el contrario, incorrecta.

``` python
        if answer == 0:
            text("Respuesta correcta", frame, 100, (0, 255, 0))
            delay += 1
        elif answer == 1:
            text("Respuesta incorrecta", frame, 100, (0, 0, 255))
            delay += 1
        elif answer == 2:
            text("Respuesta correcta", frame, 100, (0, 255, 0))
            delay += 1
        elif answer == 3:
            text("Respuesta incorrecta", frame, 100, (0, 0, 255))
            delay += 1
```

Para que el mensaje sea visible por el tiempo suficiente, se va controlando una variable que produce un cierto delay, durando el mensaje por unos 25 frames.

``` python
        if delay >= 25:
            answer = None
            delay = 0
```

#### OFFSET, SEPARO EL APARTADO PARA NO LIARME

Si ocurre un error, se imprimirá un mensaje avisando del error. Uno muy común, si uno no se coloca adecuadamente, es que no detecta el rostro, también avisa el programa si la cámara se encuentra apagada. 

``` python
    except Exception as e:

        print("Error:", e)
```

Si se desea salir antes de tiempo, sirve con solo darle al botón Esc en el teclado.

``` python
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # Salir si se presiona 'q'

        break
```