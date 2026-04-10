# Proyectos de Machine Learning

**Autor:** Juan Camilo Perea Possos

---

## Proyectos

### Adult Income Classification
Clasificacion binaria sobre el dataset Adult (UCI). Predice si una persona gana `>50K` al año.

**Dataset:** 32,561 registros, 15 variables (edad, educacion, ocupacion, etc.)
**Mejor modelo:** XGBoost con GridSearch — F1=0.7289, accuracy=0.8769

#### Fases completadas

| Fase | Descripcion | Resultado |
|------|-------------|-----------|
| 0 | Setup | `df.shape=(32561, 15)` |
| 1 | Limpieza y codificacion | `X.shape=(32561, 104)`, NaN=0 |
| 2 | EDA | Top feature: `marital_status_Married-civ-spouse` (corr=0.44) |
| 3 | Baseline Naive Bayes | F1=0.4759, accuracy=0.8099 |
| 4 | Modelos avanzados (default) | XGBoost F1=0.7181 / RF F1=0.6723 / AdaBoost F1=0.6636 / DT F1=0.6303 |
| 5 | Optimizacion HPO (5 metodos) | GridSearch/XGBoost F1=0.7289 |
| 6 | Reduccion de dimensionalidad | PCA F1=0.6114 / LDA F1=0.5421 / UMAP F1=0.5615 / t-SNE viz |
| 7 | Comparacion final | 18 configuraciones evaluadas, XGBoost domina |
| 8 | Produccion | Pipeline end-to-end guardado en `Adult/models/adult_best_model.joblib` |

#### Estructura
```
Adult/
├── Adults_Classification.ipynb   # Notebook principal con outputs
├── Adults_Project_Plan.md        # Plan de ejecucion y resultados por fase
└── models/
    ├── adult_best_model.joblib   # Pipeline produccion (OHE + XGBoost)
    ├── cm_advanced_models.png
    ├── cm_gaussiannb.png
    ├── eda_heatmap.png
    ├── eda_histogramas.png
    └── eda_target.png
data/
└── adult.csv                     # Dataset UCI Adult (32,561 filas)
```

#### Metodos HPO comparados
| Metodo | RF F1 | XGB F1 |
|--------|-------|--------|
| GridSearchCV | 0.6984 | **0.7289** |
| RandomizedSearchCV | 0.6920 | 0.7141 |
| Optuna (TPE) | 0.6978 | 0.7282 |
| Hyperopt (TPE) | 0.6947 | 0.7223 |
| NNI (simulacion local) | 0.6990 | 0.7274 |

## Requisitos

- Python 3.10+
- Dependencias en `requirements.txt`

## Instalacion y Ejecucion

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook Adult/Adults_Classification.ipynb
```
