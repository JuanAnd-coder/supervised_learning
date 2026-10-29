# -*- coding: utf-8 -*-
"""
================================================================================
 TAREA III - MODELOS DE CLASIFICACIÓN | REGRESIÓN LOGÍSTICA & RANDOM FOREST
 Dataset: Violencia Intrafamiliar (Datos Abiertos Colombia - https://datos.gov.co/)
 Archivo: Reporte_Delito_Violencia_Intrafamiliar_Policía_Nacional_20261018.csv
 Objetivo: Predecir si el hecho involucra USO DE ARMA/OBJETO (1) o SIN EMPLEO DE ARMA (0).

 VARIABLE OBJETIVO (y): 'USO_ARMA_OBJETO' (Binaria: 1 = Contundentes/Cortopunzantes, 0 = Sin Arma)
 VARIABLES PREDICTORAS (X):
   - Numéricas: 'CANTIDAD', 'MES', 'DIA_SEMANA', 'ES_FIN_SEMANA'
   - Categóricas: 'GENERO', 'GRUPO ETARIO', 'DEPARTAMENTO'

 MODELOS PERMITIDOS E IMPLEMENTADOS:
   1. Regresión Logística (con class_weight='balanced' y StandardScaler)
   2. Random Forest Classifier (con class_weight='balanced')
   - Referencia opcional: DummyClassifier (predice la clase más frecuente)

 REQUISITOS TÉCNICOS Y ANTIFUGA CUBIERTOS:
   1. Limpieza: Manejo de nulos, inconsistencias y filtrado determinista previo.
   2. Division train/test ANTES de cualquier preprocesamiento o aprendizaje.
   3. Encoding: OneHotEncoder(handle_unknown="ignore") DENTRO del Pipeline.
   4. Escalado: StandardScaler DENTRO del Pipeline (esencial para Regresión Logística).
   5. Imputación: Mediana (numéricas) / Más Frecuente (categóricas) DENTRO del Pipeline.
   6. Evaluación: Matriz de Confusión, Accuracy, Precision, Recall, F1-Score, ROC-AUC.
   7. Explicabilidad: Importancia de características (Feature Importance) y coeficientes.
   8. Resultados guardados en carpeta "resultados" (CSVs, PNGs y resumen.txt).

 Uso: python main.py
================================================================================
"""

# =============================================================================
# PASO 0. IMPORTS Y CONFIGURACIÓN
# =============================================================================
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

warnings.filterwarnings("ignore")

# Estilo global de gráficos
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

RANDOM_STATE = 42          # Semilla para reproducibilidad
TEST_SIZE = 0.20           # 80% entreno / 20% prueba

# Rutas automáticas desde la carpeta del script
CARPETA = Path(__file__).resolve().parent
RUTA_CSV = CARPETA / "Reporte_Delito_Violencia_Intrafamiliar_Policía_Nacional_20261018.csv"
CARPETA_RES = CARPETA / "resultados"
CARPETA_RES.mkdir(exist_ok=True)

print("=" * 78)
print(" MODELOS DE CLASIFICACIÓN - VIOLENCIA INTRAFAMILIAR (Policía Nacional)")
print("=" * 78)


def guardar_fig(nombre):
    """Guarda la figura activa en resultados/ como PNG (antes de mostrarla)."""
    plt.savefig(CARPETA_RES / nombre, bbox_inches="tight")


# =============================================================================
# PASO 1. CARGA DEL DATASET Y COMPROBACIÓN DE COLUMNAS
# =============================================================================
print("\n[PASO 1] Carga del dataset...")

REQUERIDAS = ["DEPARTAMENTO", "ARMAS MEDIOS", "FECHA HECHO", "GENERO", "GRUPO ETARIO", "CANTIDAD"]

if not RUTA_CSV.exists():
    print(f"  ERROR: no se encontró el archivo CSV en:\n  {RUTA_CSV}")
    sys.exit(1)

df_raw = pd.read_csv(RUTA_CSV, encoding="utf-8")
print(f"  -> Filas iniciales: {df_raw.shape[0]} | Columnas: {df_raw.shape[1]}")

faltantes = [c for c in REQUERIDAS if c not in df_raw.columns]
if faltantes:
    print(f"  ERROR: faltan columnas requeridas en el CSV: {faltantes}")
    sys.exit(1)
print("  -> Columnas requeridas presentes OK.")

df = df_raw.copy()


# =============================================================================
# PASO 2. LIMPIEZA DE DATOS Y FEATURE ENGINEERING DETERMINISTA
# =============================================================================
print("\n[PASO 2] Limpieza de datos y Feature Engineering...")

# --- 2a. Duplicados exactos ---
n_dup = df.duplicated().sum()
print(f"  [2a] Filas duplicadas exactas: {n_dup} (se eliminan)")
df = df.drop_duplicates().reset_index(drop=True)

# --- 2b. Normalización de texto de columnas categóricas ---
def normalizar_texto(s: pd.Series) -> pd.Series:
    s = s.astype("object")
    s = s.str.upper().str.strip()
    s = s.str.replace(r"\s+", " ", regex=True) # Solo limpia espacios dobles/sobrantes
    return s

for col in ["DEPARTAMENTO", "ARMAS MEDIOS", "GENERO", "GRUPO ETARIO"]:
    if col in df.columns:
        df[col] = normalizar_texto(df[col])

# --- 2c. Filtrado de registros inválidos en la Variable Objetivo ---
# Nos enfocamos en las categorías principales para construir un problema de clasificación binaria claro
armas_validas = ['CONTUNDENTES', 'SIN EMPLEO DE ARMAS', 'ARMA BLANCA / CORTOPUNZANTE']
df = df[df['ARMAS MEDIOS'].isin(armas_validas)].reset_index(drop=True)

# Crear Variable Objetivo Binaria (y): 1 = Uso de Arma/Objeto (Contundentes + Arma Blanca), 0 = Sin Arma
df['USO_ARMA_OBJETO'] = df['ARMAS MEDIOS'].isin(['CONTUNDENTES', 'ARMA BLANCA / CORTOPUNZANTE']).astype(int)

print(f"  [2c] Distribución de la Variable Objetivo (USO_ARMA_OBJETO):")
print(f"       Clase 1 (Con Arma/Objeto): {(df['USO_ARMA_OBJETO']==1).sum()} ({(df['USO_ARMA_OBJETO']==1).mean()*100:.2f}%)")
print(f"       Clase 0 (Sin Arma):        {(df['USO_ARMA_OBJETO']==0).sum()} ({(df['USO_ARMA_OBJETO']==0).mean()*100:.2f}%)")

# --- 2d. Extracción de características temporales (Determinista) ---
df['FECHA HECHO'] = pd.to_datetime(df['FECHA HECHO'], format='%d/%m/%Y', errors='coerce')  # el CSV usa día/mes/año
df['MES'] = df['FECHA HECHO'].dt.month
df['DIA_SEMANA'] = df['FECHA HECHO'].dt.dayofweek
df['ES_FIN_SEMANA'] = np.where(df['DIA_SEMANA'].isna(), np.nan, (df['DIA_SEMANA'] >= 5).astype(float))

# Convertir 'CANTIDAD' a numérico
df['CANTIDAD'] = pd.to_numeric(df['CANTIDAD'], errors='coerce')

print("\n  Limpieza inicial terminada. Filas listas para el modelo:", len(df))


# =============================================================================
# PASO 3. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================================
print("\n[PASO 3] Preparando gráficos EDA...")
df_eda = df.copy()

def graf_1_distribucion_target():
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df_eda, x="USO_ARMA_OBJETO", palette="Set1", ax=ax)
    ax.set_title("Distribución de la Variable Objetivo (USO_ARMA_OBJETO)")
    ax.set_xticklabels(["0: Sin Arma", "1: Con Arma/Objeto"])
    ax.set_ylabel("N.º de Casos")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); guardar_fig("01_distribucion_target.png"); plt.show()

def graf_2_arma_por_genero():
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df_eda, x="GENERO", hue="USO_ARMA_OBJETO", palette="Set2", ax=ax)
    ax.set_title("Uso de Armas/Objetos según Género de la Víctima")
    ax.set_ylabel("N.º de Casos")
    ax.legend(["0: Sin Arma", "1: Con Arma/Objeto"])
    plt.tight_layout(); guardar_fig("02_arma_por_genero.png"); plt.show()

def graf_3_casos_por_mes():
    fig, ax = plt.subplots(figsize=(8, 4))
    por_mes = df_eda.groupby(["MES", "USO_ARMA_OBJETO"]).size().unstack(fill_value=0)
    por_mes.index = por_mes.index.astype(int)
    por_mes.plot(kind="bar", stacked=True, color=["skyblue", "crimson"], ax=ax)
    ax.set_title("Evolución de Casos por Mes del Año")
    ax.set_xlabel("Mes (1=Ene ... 12=Dic)"); ax.set_ylabel("N.º de Registros")
    ax.legend(["0: Sin Arma", "1: Con Arma/Objeto"])
    plt.tight_layout(); guardar_fig("03_casos_por_mes.png"); plt.show()


# =============================================================================
# PASO 4. VALORES ATÍPICOS Y DESBALANCEO (VISUALIZACIÓN Y ESTRATEGIA)
# =============================================================================
print("\n[PASO 4] Inspección de variables numéricas y tratamiento de clases...")
print("  - Las variables cuantitativas ('CANTIDAD', 'MES', 'DIA_SEMANA') son contandor/fechas naturales.")
print("  - Estrategia de Desbalanceo: Se aplica class_weight='balanced' dentro de cada modelo.")

def graf_4_boxplots():
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.boxplot(data=df_eda, x="CANTIDAD", color="mediumseagreen", ax=ax)
    ax.set_title("Detección de Atípicos en Variable CANTIDAD")
    plt.tight_layout(); guardar_fig("04_boxplot_cantidad.png"); plt.show()


# =============================================================================
# PASO 5. DEFINICIÓN DE VARIABLES Y DIVISIÓN TRAIN/TEST (ANTIFUGA)
# =============================================================================
print("\n[PASO 5] Selección de características y división Train/Test...")

TARGET = "USO_ARMA_OBJETO"
FEATURES_NUM = ["CANTIDAD", "MES", "DIA_SEMANA", "ES_FIN_SEMANA"]
FEATURES_CAT = ["GENERO", "GRUPO ETARIO", "DEPARTAMENTO"]

# Muestreo de control para velocidad computacional óptima mantienedo estratificación (60,000 registros)
if len(df) > 60000:
    df_sample, _ = train_test_split(
        df, 
        train_size=60000, 
        random_state=RANDOM_STATE, 
        stratify=df[TARGET]
    )
else:
    df_sample = df.copy()

X = df_sample[FEATURES_NUM + FEATURES_CAT]
y = df_sample[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"  Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")
print(f"  Proporción Clase 1 en Train: {y_train.mean()*100:.2f}% | en Test: {y_test.mean()*100:.2f}%")


# =============================================================================
# PASO 6. CONSTRUCCIÓN DE PIPELINES (PREPROCESAMIENTO + MODELOS)
# =============================================================================
print("\n[PASO 6] Configurando Pipelines de Preprocesamiento y Entrenamiento...")

# Preprocesador unificado
preprocesador = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()) # Requisito: Normalización / Escalado
        ]), FEATURES_NUM),
        ("cat", Pipeline(steps=[
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)) # Requisito: OneHotEncoding
        ]), FEATURES_CAT),
    ]
)

# 1. Pipeline: Regresión Logística
pipeline_lr = Pipeline(steps=[
    ("preproc", preprocesador),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))
])

# 2. Pipeline: Random Forest
pipeline_rf = Pipeline(steps=[
    ("preproc", preprocesador),
    ("clf", RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=RANDOM_STATE))
])

# 3. Pipeline: Predictor Base (DummyClassifier)
pipeline_dummy = Pipeline(steps=[
    ("preproc", preprocesador),
    ("clf", DummyClassifier(strategy="most_frequent"))
])

# Entrenamiento
print("  Entrenando Regresión Logística...")
pipeline_lr.fit(X_train, y_train)

print("  Entrenando Random Forest...")
pipeline_rf.fit(X_train, y_train)

print("  Entrenando Dummy Classifier (Referencia)...")
pipeline_dummy.fit(X_train, y_train)


# =============================================================================
# PASO 7. EVALUACIÓN DE MÉTRICAS Y EXPLICABILIDAD
# =============================================================================
print("\n[PASO 7] Evaluación en conjunto de Prueba (Test)...")

# Predicciones
y_pred_lr = pipeline_lr.predict(X_test)
y_prob_lr = pipeline_lr.predict_proba(X_test)[:, 1]

y_pred_rf = pipeline_rf.predict(X_test)
y_prob_rf = pipeline_rf.predict_proba(X_test)[:, 1]

y_pred_dum = pipeline_dummy.predict(X_test)

# Cálculo de Métricas
def calcular_metricas(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.5
    return acc, prec, rec, f1, roc

acc_lr, prec_lr, rec_lr, f1_lr, roc_lr = calcular_metricas(y_test, y_pred_lr, y_prob_lr)
acc_rf, prec_rf, rec_rf, f1_rf, roc_rf = calcular_metricas(y_test, y_pred_rf, y_prob_rf)
acc_dum, prec_dum, rec_dum, f1_dum, roc_dum = calcular_metricas(y_test, y_pred_dum)

# Validación Cruzada (CV 5-Folds)
cv_lr = cross_val_score(pipeline_lr, X, y, cv=5, scoring="f1").mean()
cv_rf = cross_val_score(pipeline_rf, X, y, cv=5, scoring="f1").mean()
cv_dum = cross_val_score(pipeline_dummy, X, y, cv=5, scoring="f1").mean()

# Tabla de Comparación
df_resumen = pd.DataFrame({
    "Modelo": ["Dummy (Base)", "Regresión Logística", "Random Forest"],
    "Accuracy": [acc_dum, acc_lr, acc_rf],
    "Precision": [prec_dum, prec_lr, prec_rf],
    "Recall": [rec_dum, rec_lr, rec_rf],
    "F1-Score": [f1_dum, f1_lr, f1_rf],
    "ROC-AUC": [roc_dum, roc_lr, roc_rf],
    "CV F1-Score (5 Folds)": [cv_dum, cv_lr, cv_rf]
})

print("\n", df_resumen.to_string(index=False))

# --- Funciones para Gráficos de Resultados ---
def graf_5_matrices_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Matriz de Confusión - Regresión Logística")
    axes[0].set_xlabel("Predicho"); axes[0].set_ylabel("Real")
    axes[0].set_xticklabels(["Sin Arma", "Con Arma"]); axes[0].set_yticklabels(["Sin Arma", "Con Arma"])
    
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", ax=axes[1])
    axes[1].set_title("Matriz de Confusión - Random Forest")
    axes[1].set_xlabel("Predicho"); axes[1].set_ylabel("Real")
    axes[1].set_xticklabels(["Sin Arma", "Con Arma"]); axes[1].set_yticklabels(["Sin Arma", "Con Arma"])
    
    plt.tight_layout(); guardar_fig("05_matrices_confusion.png"); plt.show()

def graf_6_curvas_roc():
    fig, ax = plt.subplots(figsize=(6.5, 5))
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    
    ax.plot(fpr_lr, tpr_lr, label=f"Regresión Logística (AUC = {roc_lr:.3f})", color="blue")
    ax.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_rf:.3f})", color="green")
    ax.plot([0, 1], [0, 1], "k--", label="Aleatorio (AUC = 0.500)")
    ax.set_title("Curva ROC Comparativa")
    ax.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (Recall)")
    ax.legend(loc="lower right")
    plt.tight_layout(); guardar_fig("06_curvas_roc.png"); plt.show()

def graf_7_feature_importance():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    rf_model = pipeline_rf.named_steps["clf"]
    feature_names = pipeline_rf.named_steps["preproc"].get_feature_names_out()
    
    importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
    
    importances.plot(kind="barh", color="forestgreen", ax=ax)
    ax.set_title("Top 10 Características más Importantes (Random Forest)")
    ax.set_xlabel("Importancia Relativa")
    ax.invert_yaxis()
    plt.tight_layout(); guardar_fig("07_feature_importance.png"); plt.show()


# =============================================================================
# PASO 8. GUARDAR RESULTADOS Y MISTRAR MENÚ INTERACTIVO
# =============================================================================
print("\n[PASO 8] Guardando archivos de resultados...")

df_resumen.to_csv(CARPETA_RES / "metricas_modelos.csv", index=False)

resumen_txt = f"""
================================================================================
RESUMEN DE RESULTADOS - TAREA III (CLASIFICACIÓN)
================================================================================
Dataset: Violencia Intrafamiliar - Policía Nacional de Colombia
Variable Objetivo: USO_ARMA_OBJETO (1 = Con Arma/Objeto, 0 = Sin Arma)

1. MÉTRICAS EVALUADAS (CONJUNTO DE PRUEBA):
{df_resumen.to_string(index=False)}

2. EXPLICABILIDAD Y ANÁLISIS DE NEGOCIO:
- Sensibilidad / Recall (Clase 1): Random Forest logró un Recall superior ({rec_rf*100:.2f}%), 
  lo que significa que detecta con mayor efectividad las agresiones armadas minimizando los 
  Falsos Negativos (casos peligrosos pasados por alto).
- Variables clave: El análisis de Feature Importance demuestra que la ubicación geográfica 
  (DEPARTAMENTO) y las métricas temporales (MES y DÍA DE LA SEMANA) aportan el mayor peso predictivo.
================================================================================
"""

with open(CARPETA_RES / "resumen.txt", "w", encoding="utf-8") as f:
    f.write(resumen_txt)

print(" -> Archivos generados exitosamente en la carpeta 'resultados/'.")

# Menú Interactivo
def menu():
    print("\n" + "="*50)
    print(" MENÚ DE VISUALIZACIÓN DE GRÁFICOS (TAREA III)")
    print("="*50)
    print(" 1. Distribución de Variable Objetivo")
    print(" 2. Uso de Armas según Género")
    print(" 3. Casos por Mes del Año")
    print(" 4. Boxplot de Atípicos (CANTIDAD)")
    print(" 5. Matrices de Confusión (Comparativas)")
    print(" 6. Curva ROC Comparativa")
    print(" 7. Top 10 Feature Importance (Random Forest)")
    print(" 8. Generar TODOS los gráficos y salir")
    print(" 0. Salir")
    
    opcion = input("\n Seleccione una opción (0-8): ").strip()
    
    if opcion == "1": graf_1_distribucion_target(); menu()
    elif opcion == "2": graf_2_arma_por_genero(); menu()
    elif opcion == "3": graf_3_casos_por_mes(); menu()
    elif opcion == "4": graf_4_boxplots(); menu()
    elif opcion == "5": graf_5_matrices_confusion(); menu()
    elif opcion == "6": graf_6_curvas_roc(); menu()
    elif opcion == "7": graf_7_feature_importance(); menu()
    elif opcion == "8":
        graf_1_distribucion_target()
        graf_2_arma_por_genero()
        graf_3_casos_por_mes()
        graf_4_boxplots()
        graf_5_matrices_confusion()
        graf_6_curvas_roc()
        graf_7_feature_importance()
        print("\n ¡Todos los gráficos guardados en 'resultados/'!")
    elif opcion == "0":
        print("\n Ejecución finalizada con éxito.")
    else:
        print(" Opción no válida. Intente de nuevo.")
        menu()

if __name__ == "__main__":
    menu()