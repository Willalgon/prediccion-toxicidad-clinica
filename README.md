# DeepTox: Modelado Predictivo de Toxicidad Clínica mediante Deep Learning

## 1. Resumen Ejecutivo
La industria farmacéutica se enfrenta a un desafío crítico en el descubrimiento de fármacos: aproximadamente el 90% de los candidatos a fármacos fallan durante los ensayos clínicos, siendo la toxicidad no detectada la causa principal. Este proyecto implementa una Red Neuronal Profunda (DNN) diseñada para predecir la toxicidad clínica de pequeñas moléculas. Al actuar como una herramienta de cribado virtual, este modelo tiene como objetivo reducir los costes de I+D y mitigar los riesgos antes de que los compuestos entren en la fase clínica.

## 2. Definición del Problema
La evaluación manual de la toxicidad molecular es costosa y requiere mucho tiempo. El objetivo es desarrollar un clasificador binario capaz de distinguir entre compuestos que probablemente superen los estándares de seguridad de la FDA y aquellos con probabilidad de fracaso debido a problemas toxicológicos.

## 3. Especificaciones del Dataset
El modelo utiliza el conjunto de datos **ClinTox**, un referente en el repositorio MoleculeNet.
* **Observaciones:** ~1,500 compuestos químicos.
* **Características:** Descriptores moleculares y huellas dactilares (ECFP4/Morgan Fingerprints) derivados de cadenas SMILES (Simplified Molecular Input Line Entry System).
* **Variable Objetivo:** `CT_TOX` (Binaria: 0 para seguro/no tóxico, 1 para tóxico/fracaso clínico).

## 4. Arquitectura Técnica
La implementación sigue una arquitectura de Perceptrón Multicapa (MLP):

* **Capa de Entrada:** Vector de alta dimensión que representa las huellas dactilares moleculares.
* **Capas Ocultas:** Dos capas totalmente conectadas (fully connected) con funciones de activación **ReLU (Rectified Linear Unit)** para capturar relaciones químicas no lineales.
* **Capa de Salida:** Una única neurona con función de activación **Sigmoide** para la clasificación probabilística.
* **Función de Pérdida:** **Entropía Cruzada Binaria (BCE)**, elegida por su eficiencia en tareas de clasificación binaria.
* **Optimitzador:** Adam, con tasa de aprendizaje adaptativa.

## 5. Metodología e Implementación
1. **Preprocesamiento de Datos:** Conversión de cadenas SMILES en representaciones numéricas utilizando RDKit.
2. **Entrenamiento del Modelo:** Actualización iterativa de pesos mediante retropropagación (backpropagation) para minimizar la función de coste.
3. **Estrategia de Validación:** División de datos 80/20 (entrenamiento/prueba) con muestreo estratificado para corregir posibles desequilibrios de clase.
4. **Métricas de Rendimiento:** Además de la precisión estándar, el modelo se evalúa mediante el Área Bajo la Curva ROC (AUC-ROC) y matrices de confusión para minimizar los Falsos Negativos (críticos en evaluaciones de seguridad).

## 6. Conclusiones y Análisis
* El modelo demuestra la viabilidad del uso de aprendizaje profundo como filtro preliminar en el embudo de descubrimiento de fármacos.
* El análisis de la importancia de las características (gradientes) indica que subestructuras moleculares específicas están altamente correlacionadas con el fracaso clínico.

## 7. Requisitos Técnicos
* Python 3.x
* TensorFlow / Keras
* RDKit
* Pandas / NumPy
* Scikit-Learn
