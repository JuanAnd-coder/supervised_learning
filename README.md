# 🛡️ Clasificación de Violencia Intrafamiliar - Uso de Armas/Objetos

Este repositorio contiene la implementación end-to-end de un flujo de **Machine Learning Supervisado para Clasificación Binaria** en Python, utilizando datos abiertos de la Policía Nacional de Colombia sobre Violencia Intrafamiliar.

El objetivo principal es predecir si un hecho registrado involucró el **Uso de Arma/Objeto** (Contundente o Cortopunzante) o si fue **Sin Empleo de Arma**, implementando buenas prácticas metodológicas para evitar la fuga de información (*data leakage*).

---



## 📋 Índice

* [Descripción del Proyecto](#-descripción-del-proyecto)
* [Diferencia Metodológica: Regresión vs. Clasificación](#-diferencia-metodológica-regresión-vs-clasificación)
* [Prevención de Fuga de Información](#-prevención-de-fuga-de-información)
* [Estructura del Repositorio](#-estructura-del-repositorio)
* [Tecnologías Utilizadas](#-tecnologías-utilizadas)
* [Resultados y Métricas del Modelo](#-resultados-y-métricas-del-modelo)
* [Instrucciones de Ejecución](#-instrucciones-de-ejecución)
* [Autor](#-autor)

---

## 🎯 Descripción del Proyecto

* **Variable Objetivo ($y$):** `USO_ARMA_OBJETO` (Binaria)
  * `1`: Presencia de armas contundentes o cortopunzantes.
  * `0`: Sin empleo de armas.
* **Variables Predictoras ($X$):**
  * **Numéricas:** `CANTIDAD`, `MES`, `DIA_SEMANA`, `ES_FIN_SEMANA`.
  * **Categorización:** `GENERO`, `GRUPO ETARIO`, `DEPARTAMENTO`.

---

## 🧠 Diferencia Metodológica: Regresión vs. Clasificación

A diferencia de un modelo de **Regresión** (donde se predice un número continuo como la cantidad de heridos), este problema de **Clasificación** busca asignar probabilidades y categorizar observaciones en clases discretas.

Se implementan dos enfoques supervisados con tratamiento de desbalanceo mediante `class_weight="balanced"`:
1. **Regresión Logística:** Modelo lineal probabilístico con escalado estandarizado.
2. **Random Forest Classifier:** Ensamble de árboles de decisión para capturar patrones no lineales complejos.

---

## 🛡️ Prevención de Fuga de Información (*Data Leakage*)

Para garantizar la rigurosidad científica:
1. **Estratificación y Split Temprano:** División del dataset (80% train / 20% test) preservando la proporción de clases **antes** de ajustar transformers.
2. **Pipelines Autónomos:** Imputación de faltantes (`SimpleImputer`), escalado (`StandardScaler`) y codificación one-hot (`OneHotEncoder`) se entrenan **únicamente** con el conjunto de entrenamiento (`X_train`).
3. **Manejo de Categorías Nuevas:** `OneHotEncoder(handle_unknown="ignore")` previene excepciones ante departamentos o categorías no vistas en entrenamiento.

---

## 📁 Estructura del Repositorio

```text
clasificacion-violencia-intrafamiliar/
├── Reporte_Delito_Violencia_Intrafamiliar_Policía_Nacional_20261018.csv  # Dataset
├── main.py                          # Pipeline principal y evaluación
├── requirements.txt                 # Dependencias
├── resultados/                      # Gráficos (PNG) y reportes (CSV/TXT)
│   ├── metricas_modelos.csv
│   ├── resumen.txt
│   └── *.png                        # Matrices de confusión y curvas ROC
└── README.md                        # Documentación del proyecto
```

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.x
* **Procesamiento de Datos:** Pandas, NumPy
* **Visualización:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (`LogisticRegression`, `RandomForestClassifier`, `DummyClassifier`, `Pipeline`, `ColumnTransformer`)

---

## 📊 Resultados y Métricas del Modelo

Evaluación realizada sobre el conjunto de prueba independiente ($N = 12,000$ registros evaluados):

| Modelo | Accuracy | Precision | Recall (Sensibilidad) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Base)** | 0.5397 | 0.5397 | 1.0000 | 0.7010 | 0.5000 |
| **Regresión Logística** | 0.6171 | 0.6300 | 0.7037 | 0.6648 | 0.6548 |
| **Random Forest** | **0.6226** | **0.6210** | **0.7716** | **0.6881** | **0.6588** |

### 🔍 Conclusiones de Negocio

1. **Sensibilidad / Recall Eleva:** Random Forest logra un **Recall del 77.16%** en la clase de riesgo (uso de armas), minimizando de forma efectiva los falsos negativos en la identificación de casos graves.
2. **Capacidad Discriminativa:** Ambos modelos superan al predictor aleatorio o por frecuencia base (ROC-AUC de ~0.659 frente a 0.500).
3. **Variables Clave:** La ubicación geográfica (`DEPARTAMENTO`) y los factores temporales (`MES` y `DIA_SEMANA`) aportan el mayor peso discriminativo dentro del ensamble de árboles.

---

## 🔧 Instrucciones de Ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/JuanAnd-coder/clasificacion-violencia-intrafamiliar.git
cd clasificacion-violencia-intrafamiliar
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar el pipeline de clasificación
```bash
python main.py
```

---

## 👤 Autor

Desarrollado por **Juan Andrés Cárdenas** ([@JuanAnd-coder](https://github.com/JuanAnd-coder)).
