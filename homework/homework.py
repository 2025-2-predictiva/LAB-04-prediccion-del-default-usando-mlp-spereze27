# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# flake8: noqa: E501
import os
import json
import gzip
import pickle
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- CONFIGURACIÓN Y CONSTANTES ---
INPUT_TRAIN = "files/input/train_data.csv.zip"
INPUT_TEST = "files/input/test_data.csv.zip"
OUTPUT_MODELS = "files/models/"
OUTPUT_METRICS = "files/output/"
MODEL_FILENAME = "model.pkl.gz"
METRICS_FILENAME = "metrics.json"

# --- FUNCIONES DE CARGA Y LIMPIEZA (PASO 1) ---

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Carga el dataset y realiza la limpieza solicitada en el Paso 1.
    
    Acciones:
    - Renombra columna objetivo.
    - Elimina columna ID.
    - Filtra registros inválidos (Marriage/Education != 0).
    - Agrupa niveles educativos superiores a 4.
    """
    # Carga descomprimiendo zip
    df = pd.read_csv(filepath, compression="zip").copy()

    # Renombre de columnas
    if "default payment next month" in df.columns:
        df.rename(columns={"default payment next month": "default"}, inplace=True)

    # Eliminación de ID
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)

    # Filtrado de registros con información no disponible
    df = df[(df["MARRIAGE"] != 0) & (df["EDUCATION"] != 0)].copy()

    # Agrupación de categorías educativas (>4 pasa a ser 4 'others')
    df["EDUCATION"] = df["EDUCATION"].apply(lambda v: 4 if v >= 4 else v)

    return df.dropna()


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Paso 2: Separa las características (X) de la variable objetivo (y).
    """
    X = df.drop(columns=["default"])
    y = df["default"]
    return X, y


# --- PIPELINE Y MODELADO (PASO 3 Y 4) ---

def build_and_optimize_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """
    Construye el pipeline (Paso 3) y configura la búsqueda de hiperparámetros (Paso 4).
    
    Pipeline:
    1. OneHotEncoder (cat) + StandardScaler (num)
    2. SelectKBest
    3. PCA
    4. MLPClassifier
    """
    # Identificación de columnas
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical_features = [col for col in X_train.columns if col not in categorical_features]

    # Preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), categorical_features),
            ("num", StandardScaler(), numerical_features),
        ]
    )

    # Definición del Pipeline
    pipeline = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("selector", SelectKBest(score_func=f_classif)),
            ("pca", PCA()),
            ("mlp", MLPClassifier(max_iter=15000, random_state=21)),
        ]
    )

    # Espacio de búsqueda (Hiperparámetros fijos según lógica original)
    param_grid = {
        "selector__k": [20],
        "pca__n_components": [None],
        "mlp__hidden_layer_sizes": [(50, 30, 40, 60)],
        "mlp__alpha": [0.26],
        "mlp__learning_rate_init": [0.001],
    }

    # Configuración de Cross-Validation
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )
    
    # Ajuste del modelo
    grid_search.fit(X_train, y_train)
    
    return grid_search


# --- GESTIÓN DE ARCHIVOS (PASO 5) ---

def save_model(estimator: GridSearchCV, output_path: str):
    """
    Guarda el modelo comprimido con gzip. Gestiona la limpieza del directorio previo.
    """
    output_dir = Path(output_path)
    
    # Limpieza de directorio existente (Lógica original preservada)
    if output_dir.exists():
        for file in glob(str(output_dir / "*")):
            os.remove(file)
        try:
            os.rmdir(output_dir)
        except OSError:
            pass # Ignorar si no está vacío o hay error de sistema
            
    output_dir.mkdir(parents=True, exist_ok=True)

    # Guardado comprimido
    file_path = output_dir / MODEL_FILENAME
    with gzip.open(file_path, "wb") as f:
        pickle.dump(estimator, f)


def save_metrics(metrics_list: List[Dict], output_path: str):
    """
    Guarda la lista de métricas en un archivo JSON.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = output_dir / METRICS_FILENAME
    with open(file_path, "w", encoding="utf-8") as f:
        for metric in metrics_list:
            f.write(json.dumps(metric) + "\n")


# --- CÁLCULO DE MÉTRICAS (PASO 6 Y 7) ---

def calculate_base_metrics(dataset_name: str, y_true, y_pred) -> Dict:
    """Calcula precision, balanced_accuracy, recall y f1_score."""
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def calculate_confusion_matrix(dataset_name: str, y_true, y_pred) -> Dict:
    """Calcula y formatea la matriz de confusión."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
    }


# --- ORQUESTADOR PRINCIPAL ---

def main():
    # 1. Carga y Limpieza
    print("Cargando y limpiando datos...")
    df_train = load_and_clean_data(INPUT_TRAIN)
    df_test = load_and_clean_data(INPUT_TEST)

    # 2. División X/y
    X_train, y_train = split_features_target(df_train)
    X_test, y_test = split_features_target(df_test)

    # 3. y 4. Construcción y Optimización del Pipeline
    print("Entrenando modelo (esto puede tardar)...")
    estimator = build_and_optimize_pipeline(X_train, y_train)

    # 5. Guardado del Modelo
    print("Guardando modelo...")
    save_model(estimator, OUTPUT_MODELS)

    # 6. y 7. Cálculo de Métricas
    print("Calculando métricas...")
    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)

    metrics_results = [
        calculate_base_metrics("train", y_train, y_train_pred),
        calculate_base_metrics("test", y_test, y_test_pred),
        calculate_confusion_matrix("train", y_train, y_train_pred),
        calculate_confusion_matrix("test", y_test, y_test_pred),
    ]

    save_metrics(metrics_results, OUTPUT_METRICS)
    print("Proceso finalizado exitosamente.")


if __name__ == "__main__":
    main()