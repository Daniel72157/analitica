import http.server
import socketserver
import json
import urllib.parse
import os
from pathlib import Path
import joblib
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import cgi

PUERTO = 8080
RUTA_MODELOS = Path("modelos")

escalador = joblib.load(RUTA_MODELOS / "escalador_estandar.pkl")
modelo_regresion = joblib.load(RUTA_MODELOS / "regresion_logistica.pkl")
modelo_svm = joblib.load(RUTA_MODELOS / "maquina_vectores.pkl")
modelo_red_neuronal = joblib.load(RUTA_MODELOS / "red_neuronal.pkl")
pesos_fcm = joblib.load(RUTA_MODELOS / "pesos_iniciales.pkl")

def predecir_fcm(X, pesos, umbral=0.5):
    activaciones = X @ pesos
    max_activacion = np.max(activaciones, axis=1, keepdims=True)
    salida_binaria = (activaciones >= umbral * max_activacion).astype(int)
    salida_final = salida_binaria.max(axis=1)
    return salida_final

class Manejador(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/predict":
            longitud = int(self.headers['Content-Length'])
            cuerpo = self.rfile.read(longitud)
            datos = urllib.parse.parse_qs(cuerpo.decode())

            entrada = {k: v[0] for k, v in datos.items()}
            valores = [float(entrada[f"C{i}"]) for i in range(1, 31)]
            arreglo = np.array([valores])
            escalado = escalador.transform(arreglo)

            resultado = {
                "regresion_logistica": int(modelo_regresion.predict(escalado)[0]),
                "maquina_vectores": int(modelo_svm.predict(escalado)[0]),
                "red_neuronal": int(modelo_red_neuronal.predict(escalado)[0]),
                "fcm": int(predecir_fcm(escalado, pesos_fcm)[0])
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resultado).encode())

        elif self.path == "/batch_predict":
            tipo_contenido = self.headers.get("Content-Type", "")
            if "multipart/form-data" in tipo_contenido:
                formulario = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD': 'POST'})
                archivo = formulario['csv']
                modelo_solicitado = formulario.getvalue('model').lower()

                if archivo.file:
                    df = pd.read_csv(archivo.file)

                    if "C31" not in df.columns:
                        self.send_error(400, "El dataset debe incluir la columna 'C31'.")
                        return

                    y_real = df["C31"]
                    X = df[[f"C{i}" for i in range(1, 31)]]
                    X_escalado = escalador.transform(X)

                    if modelo_solicitado == "logistic":
                        modelo = modelo_regresion
                    elif modelo_solicitado == "svm":
                        modelo = modelo_svm
                    elif modelo_solicitado == "neural_net":
                        modelo = modelo_red_neuronal
                    elif modelo_solicitado == "fcm":
                        X_array = X.values
                        y_predicho = predecir_fcm(X_array, pesos_fcm)
                        exactitud = round(accuracy_score(y_real, y_predicho) * 100, 2)
                        matriz = confusion_matrix(y_real, y_predicho).tolist()
                    else:
                        self.send_error(400, "Modelo no válido.")
                        return

                    if modelo_solicitado != "fcm":
                        y_predicho = modelo.predict(X_escalado)
                        exactitud = round(accuracy_score(y_real, y_predicho) * 100, 2)
                        matriz = confusion_matrix(y_real, y_predicho).tolist()

                    resultado = {
                        "accuracy": exactitud,
                        "confusion_matrix": matriz
                    }

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(resultado).encode())
                else:
                    self.send_error(400, "Archivo CSV no proporcionado")
            else:
                self.send_error(400, "Tipo de contenido no soportado")
        else:
            self.send_error(404, "Ruta no encontrada")

    def do_GET(self):
        if self.path == "/":
            self.path = "/index.html"
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

if __name__ == "__main__":
    print(f"Servidor iniciado en http://localhost:{PUERTO}")
    with socketserver.TCPServer(("", PUERTO), Manejador) as servidor:
        servidor.serve_forever()
