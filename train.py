import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import mutual_info_regression
import joblib
import numpy as np
import os

os.makedirs("modelos", exist_ok=True)

datos = pd.read_excel("dataset/FGR_dataset.xlsx")
datos.dropna(inplace=True)

caracteristicas = datos.drop(columns=["C31"])
etiquetas = datos["C31"]

escalador_estandar = StandardScaler()
caracteristicas_escaladas = escalador_estandar.fit_transform(caracteristicas)
joblib.dump(escalador_estandar, "modelos/escalador_estandar.pkl")

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    caracteristicas_escaladas, etiquetas, test_size=0.2, random_state=42
)

muestra = datos.sample(n=30, random_state=42)
muestra.to_csv("FGR_dataset_prueba.csv", index=False)

modelo_regresion = LogisticRegression(max_iter=1000)
modelo_regresion.fit(X_entrenamiento, y_entrenamiento)
joblib.dump(modelo_regresion, "modelos/regresion_logistica.pkl")

modelo_red_neuronal = MLPClassifier(hidden_layer_sizes=(30,), max_iter=1000)
modelo_red_neuronal.fit(X_entrenamiento, y_entrenamiento)
joblib.dump(modelo_red_neuronal, "modelos/red_neuronal.pkl")

modelo_svm = SVC(probability=True)
modelo_svm.fit(X_entrenamiento, y_entrenamiento)
joblib.dump(modelo_svm, "modelos/maquina_vectores.pkl")

escalador_minmax = MinMaxScaler()
X_entrenamiento_norm = escalador_minmax.fit_transform(X_entrenamiento)
X_prueba_norm = escalador_minmax.transform(X_prueba)

np.random.seed(42)
pesos_iniciales = np.random.uniform(-1, 1, size=(X_entrenamiento.shape[1], X_entrenamiento.shape[1]))

joblib.dump(escalador_minmax, "modelos/escalador_minmax.pkl")
joblib.dump(pesos_iniciales, "modelos/pesos_iniciales.pkl")

print("Entrenamiento y guardado de modelos completado.")
