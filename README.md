# DeepTox: Modelado Predictivo de Toxicidad Clínica mediante Deep Learning

## 1. Resumen Ejecutivo y definición del problema
Este proyecto trata de farmacovigilancia predictiva. En el desarrollo de fármacos y medicamentos, muchos fallan ya que resultan ser dañinos y y tóxicos para el cuerpo humano en el momento que se llega a la fase clínica de pruebas con personas. El análisis tratará de una clasificación binaria:
- Clase 0: el fármaco superó los ensayos clínicos (es seguro)
- Clase 1: el fármaco no superó los ensayos clínicos (es tóxico)

El modelo deberá de aprender de una nueva molécula y decidir la probabilidad de toxicidad.

## 2. Metodología e implementación
Seguiremos este orden de trabajo:
    1. data_loader.py: leeremos los datos y los transformaremos. 
    2. model.py: definiremos la estructura de nuestra red neuronal (capas, neuronas, ReLUs...)
    3. train.py: unimos data_loader y model.py para crear un bucle de entrenamiento y hacer que la red aprenda.
    4. evaluate.py: tras entrenar el modelo, evaluaremos el modelo para datos desconocidos. 

## 3. Especificaciones del Dataset
El dataset como podemos ver contiene simbología extraña. Esto se denomina SMILES.
Los ordenadores no saben ver imagenes de moléculas 3D, por lo tanto los químicos inventaron este lenguaje de texto para representar estructuras. 
- Las letras son átomos (C es cabrono, O es oxígeno)
- Los números representan dónde se cierran los anillos
- Los símbolos como = o # represetan enlaces dobles o triples.

El problema es que las redes neuronales no entienden estos textos SMILES, por ello en data_loader usaremos RDKit para convertir este texto en un vector de números (ceros y unos) llamado Morgan Fingerprint. Diríamos que es como el DNI numérico de la molécula.

Estos datos son compuestos químicos reales extraidos de ensayos clínicos de la FDA estadounidense.
Abarca moléculas pequeñas como la aspirina y el paracetamol que han sido testeadas en humanos.

Cada fila del dataset es un experimento histórico. Por lo tanto, usaremos décadas de investigación médica para entrenar un modelo que pueda predecir el futuro de nuevas medicinas. 

Como podeis ver, el dataset está compuesto por 3 columnas:
    1. smiles: lo que ya hemos comentado, la cadena de texto que representa la estructura de la molécula.
    2. FDA_APPROVED: indica si la molécula fue aprobada por la FDA o no. Esta sería nuestra etiqueta 1
    3. CT_TOX: indica si la molécula falló específicamente por toxicidad en ensayos clínicos. Esta será nuestra etiqueta objetivo. 

## 4. Conclusiones y Análisis
NOS ENFRENTAMOS A UN GRAN PROBLEMA:
En ClinTox, la gran mayoría de las filas son de clase 0. Hay muy pocas moléculas identificadas como tóxicas (clase 1).
Esto puede hacer que la neurona dando clase 0 siempre, obtenga un éxito del 90%
De este modo, tendremos que usar métricas como precision-recall o el AUC-ROC para demostrar que nuestro modelo realmente indentifica los casos tóxicos y no solo está adivinando lo más frecuente.


## 5. Requisitos Técnicos
* Python 3.x
* Pytorch
* RDKit
* Pandas
* NumPy
* Scikit-Learn
* matplotlib
* stremlit
Se han especificado todas las dependencias y requisitos técnicos necesarios en "requirements.txt"