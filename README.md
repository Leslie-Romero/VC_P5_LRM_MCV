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

### Entrenamiento de un modelo para la predicción de edad

Para la primera tarea, hemos decidido entrenar un modelo que detecte la edad, sin embargo, ha sido un proceso largo. En un principio, tuvimos un dataset muy pequeño e intentamos extraer las características a través de HOG, sin embargo, los resultados eran bastante pobres. Luego elegimos el dataset que se especifica en las referencias y que adjuntamos al final con los contenidos, el cual contiene alrededor de 87.000 imágenes entre test de entrenamiento y test de validación, sin embargo, solo trabajamos con las 67.000 imágenes correspondientes al test de entrenamiento, las cuales luego nos encargamos de subdividir manualmente.

Aquí se puede ver como dividimos el dataset en carpetas a razón de un archivo que los dueños adjuntan con las imágenes que indican a qué grupo de edad pertenecen, para nuestro procesamiento, necesitábamos las imágenes divididas en las carpetas cuyo nombre representaba la clase a la que pertenecían, en este caso, el grupo de edad.

```py
# Hay que modificar las siguientes rutas segun donde se tenga el dataset
csv_path = "C:/Users/lllrm/Downloads/dataset_VC/fairface_label_train.csv"
output_dir = "C:/Users/lllrm/Downloads/dataset_VC/ages"
image_path = "C:/Users/lllrm/Downloads/dataset_VC"

df = pd.read_csv(csv_path)

for _, row in df.iterrows():
    age_group = row['age']
    src = os.path.join(image_path, row['file'])
    dst = os.path.join(output_dir, age_group, os.path.basename(row['file']))

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if os.path.exists(src):
        copyfile(src, dst)
```

Tras este paso intermedio, pasamos al entrenamiento, el cual tras varios intentos con distintas técnicas que fueron, o muy pobres o tardaban tanto tiempo que era imposible de ejecutar (ejecución de días), decidimos usar los embeddings de Deepface.

Extraemos los embeddings de todas las imágenes de entrenamiento y los guardamos en archivos (`X_embeddings.npy`, `Y_embeddings.npy`, `data_labels_embeddings.npy`) para poder acelerar las pruebas (una sola ejecución para procesar estos embeddings tardó alrededor de 6 horas):

```py
folder = "C:/Users/lllrm/Downloads/dataset_VC/ages"

X_file = "X_embeddings.npy"
Y_file = "Y_embeddings.npy"
labels_file = "class_labels_embeddings.npy"

# En caso de que ya estén guuardados, cargamos los embeddings
if os.path.exists(X_file) and os.path.exists(Y_file):
    print("Loading cached embeddings...")
    X = np.load(X_file)
    Y = np.load(Y_file)
    classlabels = np.load(labels_file, allow_pickle=True)
    print("Loaded embeddings!")
else:
    X = []
    Y = []
    classlabels = []

    nclasses = 0
    for class_name in os.listdir(folder):
        class_folder = os.path.join(folder, class_name)
        if not os.path.isdir(class_folder):
            continue
        nclasses += 1
        classlabels.append(class_name)

        for file_name in tqdm(os.listdir(class_folder)):
            if not file_name.lower().endswith(".jpg"):
                continue
            img_path = os.path.join(class_folder, file_name)
            image = cv2.imread(img_path)

            # Extraemos el embedding con Deepface
            embedding = DeepFace.represent(img_path=img_path, model_name="Facenet512", enforce_detection=False)
            X.append(embedding[0]["embedding"])
            Y.append(nclasses - 1)

    X = np.array(X, dtype="float32")
    Y = np.array(Y, dtype="int64")

    # Guardamos en archivos para evitar la espera
    np.save(X_file, X)
    np.save(Y_file, Y)
    np.save(labels_file, np.array(classlabels))
    print("Saved embeddings!")

```

Cuando ya teníamos los embeddings cargados y guardados en las variables X e Y, realizamos la división entre train y test set:
```py
if not X or not Y:
    X = np.load("X_embeddings.npy")
    Y = np.load("Y_embeddings.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

print("# samples in training set %d" % X_train.shape[0])
print("# samples in test set %d" % X_test.shape[0])
```

Finalmente, está todo preparado para el entrenamiento del modelo con los embeddings obtenidos:
```py
model_svm = LinearSVC(C=5, dual=False, max_iter=5000)

X = np.load("X_embeddings.npy")

print("Training SVM on embeddings...")
t0 = time()
model_svm.fit(X_train, y_train)
print("Training done in %0.2f seconds" % (time() - t0))

joblib.dump(model_svm, "svm_deepface_embeddings.joblib")
print("Model saved!")

y_pred = model_svm.predict(X_test)
print(classification_report(y_test, y_pred, target_names=classlabels))
print("Precision: %0.3f, Recall: %0.3f" %
      (precision_score(y_test, y_pred, average="macro"),
       recall_score(y_test, y_pred, average="macro")))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
```
Con este entrenamiento obtuvimos también las métricas de resultado, y aunque no son las mejores, son lo más altas que se pudieron obtener con el tiempo que teníamos para entrenar y la técnica que utilizamos (mientras no utilicemos modelos de Deep Learning no se pueden obtener resultados por encima del 70% de precisión para detección de edad, ya que las características de la edad son muy difíciles de detectar con "handcrafted features").

Como último paso, solo queda probar el modelo, que tal y como se puede ver, tiende mucho al grupo de edad 3-9, pero de vez en cuando y con mucho ajuste y preprocesado, se puede ver una detección más o menos decente.

```py
# ... funciones de pre-procesado y cargado del modelo entrenado ...

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    try:
        # Detectamos las caras presentes en el frame
        faces = DeepFace.extract_faces(frame, detector_backend='opencv')
    except:
        faces = []

    for face in faces:
        area = face["facial_area"]
        x, y, w, h = area["x"], area["y"], area["w"], area["h"]
        x2, y2, w2, h2 = expand_box(x, y, w, h)

        # Obtenemos la parte del frame que contiene la cara para pasarsela al modelo
        face_crop = frame[y2:y2+h2, x2:x2+w2]

        # Predecimos la edad
        pred = predict_age_from_face(face_crop)
        # Intentamos coger la predicción más frecuente
        smooth_pred = smooth_prediction(pred)

        age_label = str(classlabels[smooth_pred])

        cv2.rectangle(frame, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2)

        cv2.putText(frame, f"Age: {age_label}",
                    (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    cv2.imshow("Age Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

vid.release()
cv2.destroyAllWindows()

```

![Demostración del modelo entrenado](/examples/detector_edad_recorte.gif)

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

Para las preguntas que utlizamos para este minijuego, decidimos obtener 50 de temas diversos, aunque se limitan a respuestas de Verdadero o Falso para que se puedan contestar con nuestra implementación. Para obtener dichas preguntas y procesarlas en el código, las generamos con el formato de Open Trivia y las pasamos a un archivo JSON.

El procesamiento de dicho archivo JSON se realiza con el módulo `json` nativo de Python que se encarga de convertirlo a un diccionario de Python con el que resulta muy sencillo trabajar:

```py
preguntas = list()
with open("preguntas.json", "r", encoding="utf-8") as f:
    file = json.load(f)
    
preguntas = file["preguntas"]
```

Siguiento con la implementación, se realiza el mismo proceso que siempre para ir tomando los frames. Como la cámara se refleja como un espejo, se debe de girar horizontalmente, evitando ese efecto.

``` python
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
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

padding = 10
x_left = 10
(left_width, left_height), left_baseline = cv2.getTextSize(left_text, font, scale, thickness)
top_left = frame.shape[0] - 20 - left_height - padding // 2
bottom_left = frame.shape[0] - 20 + left_baseline + padding // 2
cv2.rectangle(frame, (x_left - padding, top_left), (x_left + left_width + padding, bottom_left), (0, 0, 0), -1)
cv2.putText(frame, left_text, (x_left, frame.shape[0] - 20), font, scale, (0, 255, 0), thickness)

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

Con respecto a la implementación de la parte central del código, que se encarga de la lógica que detecta el movimiento de la cabeza hacia los lados y el procesamiento de este para traducirlo en una respuesta del usuario, esto funciona de manera muy sencilla. Se establece una variable x_offset que representa la distancia hacia la derecha o izquierda del centro (marcado con una línea) de la cara detectada del usuario. Esta variable se va calculando en cada frame, y si se sobrepasa el umbral que hemos establecido (un 10% del ancho del frame tanto a izquierda como a derecha) se detecta como que el usuario ha seleccionado una opción. Para que todo esto funcione, hay que establecer desde el primer frame, que la variable `x0` se encuentre justo en el centro, donde hemos colocado la línea de referencia. 

Cuando se ha seleccionado la respuesta, se añade un rectángulo alrededor de la opción elegida para aportar "feedback" al usuario y, al volver al centro (se considera centro cuando la coordenada X de la cara detectada del usuario se encuentra dentro de las cercanías del centro con un 10% de margen a cada lado), se determina si la respuesta a la pregunta ha sido correcta o incorrecta, lo cual va a desencadenar que a partir del próximo frame, se muestre el mensaje de respuesta correcta o incorrecta (`answer`) y que se genere la siguiente pregunta (`new_question`).

```py
for face in faces:
    x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
    # Guarda la posición inicial para el primer frame
    if x0 == 0:
        x0 = frame_width // 2
    # Izquierda
    elif x_offset < -(0.1*frame_width):
        # Se considera que la persona ha inclinado la cabeza hacia la izquierda
        x_offset = 0
        # Dibujar un rectángulo sin relleno
        cv2.rectangle(frame, (x_left - padding, top_left), (x_left + left_width + padding, bottom_left), (255, 255, 255), 2)
        is_selected = 0
    # Derecha
    elif x_offset > (0.1*frame_width):
        # Se considera que la persona ha inclinado la cabeza hacia la derecha
        x_offset = 0   
        cv2.rectangle(frame, (x_right - padding, top_right), (x_right + right_width + padding, bottom_right), (255, 255, 255), 2) 
        is_selected = 1
    else:
        x_offset = 0
        if is_selected is not None:
            # Mostramos la respuesta correcta
            if is_selected == 0 and pregunta["respuesta_correcta"] == "Verdadero":
                answer = 0
                score += 1
            elif is_selected == 0 and pregunta["respuesta_correcta"] == "Falso":
                answer = 1
            elif is_selected == 1 and pregunta["respuesta_correcta"] == "Falso":
                answer = 2
                score += 1
            elif is_selected == 1 and pregunta["respuesta_correcta"] == "Verdadero":
                answer = 3
            is_selected = None
            new_question = True                    
    x_offset += (x + w//2) - x0
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 255, 200), 2)
```

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

![Demostración del filtro](/examples/filtro_recorte.gif)

## Recursos

- Link al dataset original: https://github.com/joojs/fairface
- Link a nuestro dataset derivado con las imágenes organizadas en carpetas: https://drive.google.com/drive/folders/1U9pAZhKAfaWCxv68YnpeRSb6IwjxHACL?usp=sharing
- Links a los embeddings
 - X_embeddings.npy → https://drive.google.com/file/d/1rwNr2n4Bf3Nh3QsMhjYRKeYAc7AJ0x4s/view?usp=sharing
 - Y_embeddings.npy → https://drive.google.com/file/d/1Nf8rWUszlKJrpGNREMnCY_iwfkO4ihhW/view?usp=sharing
