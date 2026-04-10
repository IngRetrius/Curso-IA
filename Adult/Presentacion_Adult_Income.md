# Clasificacion Binaria: Adult Income Dataset (UCI)

## Prediccion de ingresos anuales (>50K vs <=50K)

---

# FASE 0 — Setup y Carga de Datos

---

### Diapositiva 1: Descripcion del Problema

- **Objetivo:** Predecir si una persona gana mas de $50,000 al ano
- **Dataset:** UCI Adult Census (32,561 registros, 15 columnas)
- **Tipo de problema:** Clasificacion binaria supervisada
- **Variable target:** `income` → `<=50K` (clase 0) / `>50K` (clase 1)
- **Metrica principal:** F1-Score (binary, pos_label=1)

---

### Diapositiva 2: Variables del Dataset

- **Variables numericas (5):**
  - `age` — Edad
  - `education_num` — Nivel educativo (numerico)
  - `capital_gain` — Ganancias de capital
  - `capital_loss` — Perdidas de capital
  - `hours_per_week` — Horas trabajadas por semana
- **Variables categoricas (8):**
  - `workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country`
- **Variable eliminada:** `fnlwgt` (peso muestral, no predictivo)
- **Variable target:** `income`

---

### Diapositiva 3: Entorno Tecnico

- **Lenguaje:** Python 3.10+
- **Entorno:** Jupyter Notebook
- **Semilla global:** `RANDOM_STATE = 42` (reproducibilidad total)
- **Librerias principales:**
  - pandas, numpy, matplotlib, seaborn
  - scikit-learn, XGBoost, Optuna, Hyperopt, UMAP
- **Resultado de carga:** `df.shape = (32,561 filas x 15 columnas)` — sin errores

---

# FASE 1 — Limpieza de Datos

---

### Diapositiva 4: Valores Nulos Detectados

- **Nulos encontrados (representados como `"?"` en el CSV):**
  - `workclass` → 1,836 nulos
  - `occupation` → 1,843 nulos
  - `native_country` → 583 nulos
- **Estrategia de imputacion:** Moda (valor mas frecuente)
  - `workclass` → imputado con `"Private"`
  - `occupation` → imputado con `"Prof-specialty"`
  - `native_country` → imputado con `"United-States"`
- **Resultado:** 0 valores nulos tras imputacion

---

### Diapositiva 5: Codificacion y Transformacion

- **Target encoding:**
  - `<=50K` → 0
  - `>50K` → 1
- **Features categoricas:** One-Hot Encoding (`OneHotEncoder`, `handle_unknown="ignore"`)
- **Features numericas:** Sin transformacion (passthrough)
- **Resultado final:**
  - `X.shape = (32,561 x 104)` — 104 features tras OHE
  - `NaN en X = 0` | `NaN en y = 0`
  - Clases en y: `{0, 1}`

---

# FASE 2 — Analisis Exploratorio de Datos (EDA)

---

### Diapositiva 6: Estadisticas Descriptivas

- **age:** media = 38.58, std = 13.64, rango = [17, 90]
- **education_num:** media = 10.08, std = 2.57, rango = [1, 16]
- **capital_gain:** media = 1,077.65 (muy sesgado, max = 99,999)
- **capital_loss:** media = 87.30 (muy sesgado, max = 4,356)
- **hours_per_week:** media = 40.44, std = 12.35

> Imagen: `models/eda_histogramas.png`
> Histogramas de las 5 variables numericas. Se observa sesgo positivo marcado en capital_gain y capital_loss, distribucion aproximadamente normal en age centrada en ~35 anos, y concentracion de hours_per_week en torno a 40 horas.

---

### Diapositiva 7: Distribucion del Target (Desbalance de Clases)

- **<=50K (clase 0):** 24,720 registros (75.9%)
- **>50K (clase 1):** 7,841 registros (24.1%)
- **Ratio de desbalance:** ~3:1
- El modelo debe ser evaluado con F1-Score, no solo accuracy, debido al desbalance

> Imagen: `models/eda_target.png`
> Grafico de barras mostrando 24,720 registros en <=50K (azul) vs 7,841 en >50K (rojo/naranja). Desbalance claro de clases 76% vs 24%.

---

### Diapositiva 8: Correlaciones con el Target

- **Top 10 features mas correlacionadas con income:**
  - `marital_status_Married-civ-spouse` → 0.4447
  - `relationship_Husband` → 0.4010
  - `education_num` → 0.3352
  - `marital_status_Never-married` → 0.3184
  - `age` → 0.2340
  - `hours_per_week` → 0.2297
  - `relationship_Own-child` → 0.2285
  - `capital_gain` → 0.2233
  - `sex_Male / sex_Female` → 0.2160
- **Hallazgo clave:** El estado civil (casado) es el predictor mas fuerte, por encima de la educacion

> Imagen: `models/eda_heatmap.png`
> Heatmap de correlacion con las 20 features de mayor correlacion absoluta con income. Se observan bloques de correlacion entre variables de estado civil y relacion familiar.

---

# FASE 3 — Baseline: Naive Bayes

---

### Diapositiva 9: Modelo Baseline — GaussianNB

- **Split:** 80% train / 20% test (estratificado)
- **Preprocesado especifico:** StandardScaler + OrdinalEncoder (NB asume distribucion gaussiana)
- **Resultados en test:**

| Metrica   | Valor  |
|-----------|--------|
| Accuracy  | 0.8099 |
| Precision | 0.7078 |
| Recall    | 0.3584 |
| F1-Score  | 0.4759 |

- **Analisis:**
  - Accuracy aparentemente alta (81%) pero engana por el desbalance de clases
  - **Recall = 0.36** → el modelo solo detecta el 36% de las personas que realmente ganan >50K
  - F1 = 0.48 confirma que el balance precision/recall es deficiente
  - Sirve como linea base para superar en fases posteriores

> Imagen: `models/cm_gaussiannb.png`
> Matriz de confusion: 4,713 verdaderos negativos, 232 falsos positivos, 1,006 falsos negativos, 562 verdaderos positivos. El modelo predice bien <=50K pero falla gravemente en detectar >50K (1,006 casos perdidos).

---

# FASE 4 — Modelos Avanzados

---

### Diapositiva 10: Configuracion de Modelos

- **Split:** 60% train / 20% validation / 20% test (estratificado)
  - Train: 19,536 | Validation: 6,512 | Test: 6,513
- **4 modelos evaluados con parametros por defecto:**
  - Decision Tree
  - Random Forest
  - AdaBoost
  - XGBoost
- Todos evaluados en validation y test de forma independiente

---

### Diapositiva 11: Resultados Modelos Avanzados (Test)

| Modelo        | Accuracy | Precision | Recall | F1     |
|---------------|----------|-----------|--------|--------|
| Decision Tree | 0.8240   | 0.6377    | 0.6231 | 0.6303 |
| Random Forest | 0.8501   | 0.7099    | 0.6384 | 0.6723 |
| AdaBoost      | 0.8552   | 0.7530    | 0.5931 | 0.6636 |
| **XGBoost**   | **0.8719** | **0.7640** | **0.6773** | **0.7181** |

- **Mejor modelo:** XGBoost (F1 = 0.7181)
- **Mejora vs baseline:** F1 de 0.4759 → 0.7181 (+50.9% relativo)
- XGBoost supera a todos en accuracy, precision, recall y F1 simultaneamente
- AdaBoost tiene la mayor precision (0.7530) pero el peor recall (0.5931)

> Imagen: `models/cm_advanced_models.png`
> Matrices de confusion de los 4 modelos avanzados. XGBoost (F1=0.7181) muestra el mejor equilibrio entre verdaderos positivos y falsos negativos. Decision Tree tiene el mayor numero de falsos positivos.

---

# FASE 5 — Optimizacion de Hiperparametros (HPO)

---

### Diapositiva 12: Metodos de Optimizacion Utilizados

- Se optimizaron **2 modelos** (Random Forest y XGBoost) con **5 metodos** distintos:
  1. **GridSearchCV** — Busqueda exhaustiva en grilla predefinida (CV=3)
  2. **RandomizedSearchCV** — Muestreo aleatorio de distribuciones (15 iteraciones, CV=3)
  3. **Optuna** — Busqueda bayesiana con TPE Sampler (20 trials)
  4. **Hyperopt** — Busqueda bayesiana con Tree of Parzen Estimators (20 evals)
  5. **NNI** — Simulacion local de busqueda estilo NNI (10 iteraciones)
- **Total:** 5 metodos x 2 modelos = 10 configuraciones evaluadas

---

### Diapositiva 13: Resultados HPO — F1 en Test

| Metodo     | RF F1  | XGB F1     |
|------------|--------|------------|
| GridSearch | 0.6984 | **0.7289** |
| Random     | 0.6920 | 0.7141     |
| Optuna     | 0.6978 | 0.7282     |
| Hyperopt   | 0.6947 | 0.7223     |
| NNI        | 0.6990 | 0.7274     |

- **Mejor modelo global:** GridSearch + XGBoost → **F1 = 0.7289**
- **Mejores hiperparametros:** `max_depth=6`, `learning_rate=0.1`, `n_estimators=200`

---

### Diapositiva 14: Analisis de Resultados HPO

- **XGBoost supera a Random Forest en los 5 metodos** sin excepcion
- La HPO mejora XGBoost default de 0.7181 → 0.7289 (+0.0108 F1)
- RF se mueve en un rango estrecho: 0.6920 – 0.6990
- XGB en un rango tambien acotado: 0.7141 – 0.7289
- **Conclusion:** El espacio de busqueda estaba bien disenado; todos los metodos convergieron en soluciones cercanas
- GridSearch con espacios pequenos (8 configuraciones para XGB) encontro el optimo global

---

# FASE 6 — Reduccion de Dimensionalidad

---

### Diapositiva 15: Tecnicas Aplicadas

- **Objetivo:** Evaluar si se puede reducir el espacio de 104 features sin perder rendimiento
- 4 tecnicas evaluadas sobre datos escalados con StandardScaler:

| Tecnica | Componentes | Var. Explicada | RF F1 (test) |
|---------|-------------|----------------|--------------|
| PCA     | 86 / 104    | 95.71%         | 0.6114       |
| LDA     | 1           | —              | 0.5421       |
| t-SNE   | 2           | —              | (solo viz)   |
| UMAP    | 2           | —              | 0.5615       |

---

### Diapositiva 16: Conclusiones de Reduccion Dimensional

- **PCA** retiene 95.7% de varianza con 86 de 104 componentes → reduccion minima (18 dims)
  - RF sobre PCA pierde -0.0609 F1 vs RF default (0.6723 → 0.6114)
  - Las 18 dimensiones eliminadas contenian senal predictiva
- **LDA** a 1 dimension sufre degradacion severa (-0.1302 F1)
  - Comprime demasiado para un problema con desbalance 75/25
- **UMAP** 2D: F1 = 0.5615, mejor que LDA pero lejos del espacio completo
- **Conclusion general:** El espacio completo de 104 features es critico para este dataset
  - La reduccion dimensional dana la clasificacion en todos los casos
  - Las features OHE dispersas contienen informacion complementaria que no se puede comprimir sin perdida

---

# FASE 7 — Comparacion Final

---

### Diapositiva 17: Ranking General — Top 10 por F1 (Test)

| # | Metodo     | Modelo       | Accuracy | Precision | Recall | F1     |
|---|------------|--------------|----------|-----------|--------|--------|
| 1 | GridSearch | XGBoost      | 0.8769   | 0.7755    | 0.6875 | **0.7289** |
| 2 | Optuna     | XGBoost      | 0.8761   | 0.7716    | 0.6894 | 0.7282 |
| 3 | NNI        | XGBoost      | 0.8750   | 0.7659    | 0.6926 | 0.7274 |
| 4 | Hyperopt   | XGBoost      | 0.8743   | 0.7712    | 0.6792 | 0.7223 |
| 5 | Default    | XGBoost      | 0.8719   | 0.7640    | 0.6773 | 0.7181 |
| 6 | Random     | XGBoost      | 0.8716   | 0.7699    | 0.6658 | 0.7141 |
| 7 | NNI        | RandomForest | 0.8684   | 0.7780    | 0.6346 | 0.6990 |
| 8 | GridSearch | RandomForest | 0.8676   | 0.7736    | 0.6365 | 0.6984 |
| 9 | Optuna     | RandomForest | 0.8658   | 0.7621    | 0.6435 | 0.6978 |
|10 | Hyperopt   | RandomForest | 0.8649   | 0.7618    | 0.6384 | 0.6947 |

- **Total de modelos evaluados:** 18 (1 baseline + 4 default + 10 HPO + 3 dim. reduction)

---

### Diapositiva 18: Hallazgos Clave de la Comparacion

- **XGBoost ocupa los 6 primeros puestos** del ranking sin excepcion
- Random Forest aparece a partir del puesto 7
- Las tecnicas de reduccion dimensional quedan fuera del top 10 (F1 < 0.62)
- **Recall en test ~0.69** indica que aun se pierden ~31% de los >50K reales
- **Progresion del proyecto:**
  - Baseline (Naive Bayes): F1 = 0.4759
  - Mejor modelo default (XGBoost): F1 = 0.7181
  - Mejor modelo optimizado (GridSearch + XGBoost): F1 = **0.7289**
  - Mejora total: **+53.2% relativo** desde el baseline

---

# FASE 8 — Produccion

---

### Diapositiva 19: Pipeline de Produccion

- **Arquitectura del pipeline final:**
  1. `ColumnTransformer` — Preprocesamiento (passthrough numericas + OneHotEncoder categoricas)
  2. `XGBClassifier` — Clasificador con hiperparametros optimizados por GridSearch
- **Hiperparametros del modelo final:**
  - `max_depth = 6`
  - `learning_rate = 0.1`
  - `n_estimators = 200`
- **Reentrenamiento:** Pipeline entrenado sobre el 100% del dataset (32,561 registros)
- **Artefacto guardado:** `adult_best_model.joblib`

---

### Diapositiva 20: Smoke Test y Validacion

- **Smoke test:** Prediccion sobre las primeras 5 filas del dataset
  - Predicciones: `[0, 0, 0, 0, 1]`
  - Ground truth: `[0, 0, 0, 0, 0]`
  - Fila 5 mal clasificada → esperado: XGBoost con regularizacion no memoriza el training set
- **Validaciones superadas:**
  - Archivo `.joblib` existe en disco
  - Pipeline carga correctamente con `joblib.load()`
  - `predict()` retorna array de longitud correcta con valores en {0, 1}
  - Pipeline end-to-end acepta datos crudos (pre-OHE) directamente

---

# CONCLUSIONES Y TRABAJO FUTURO

---

### Diapositiva 21: Resumen de Resultados

- **Mejor modelo:** XGBoost optimizado con GridSearchCV
- **Metricas finales en test:**
  - Accuracy = 0.8769
  - Precision = 0.7755
  - Recall = 0.6875
  - **F1-Score = 0.7289**
- **Feature mas importante:** `marital_status_Married-civ-spouse` (correlacion = 0.4447)
- **Insight principal:** El estado civil, la relacion familiar y el nivel educativo son los predictores mas fuertes de ingresos altos
- La reduccion de dimensionalidad no aporta mejora; las 104 features completas son necesarias

---

### Diapositiva 22: Mejoras Pendientes

- **Alta prioridad:**
  - Ajuste de threshold de decision (actualmente 0.5, probar 0.35–0.40 para mejorar recall)
  - Tratamiento del desbalance de clases (`class_weight='balanced'`, SMOTE, undersampling)
  - Validacion cruzada estratificada (5-fold) para estimar varianza real del F1
  - Curvas de aprendizaje para diagnosticar underfitting vs overfitting
- **Media prioridad:**
  - UMAP con mas componentes (10, 20, 30) en lugar de solo 2D
  - Feature importance y seleccion de variables
  - Ampliar espacio HPO de Optuna/Hyperopt (100 trials vs 20)
  - Stacking/Voting ensemble (XGBoost + RF + AdaBoost)
  - Calibracion de probabilidades (`CalibratedClassifierCV`)
- **Baja prioridad:**
  - Correccion bug GPU en Fase 8 (`isinstance` en lugar de string matching)
  - Persistencia de figuras EDA a disco
  - Logging con MLflow o W&B
  - Tests de contrato del pipeline de produccion

---

### Diapositiva 23: Stack Tecnologico Completo

| Categoria              | Herramientas                                      |
|------------------------|---------------------------------------------------|
| Lenguaje               | Python 3.10+                                      |
| Entorno                | Jupyter Notebook                                   |
| Datos                  | pandas, numpy                                      |
| Visualizacion          | matplotlib, seaborn                                |
| ML Core                | scikit-learn (NB, DT, RF, AdaBoost, PCA, LDA)     |
| Gradient Boosting      | XGBoost                                            |
| HPO                    | GridSearchCV, RandomizedSearchCV, Optuna, Hyperopt |
| Reduccion Dimensional  | PCA, LDA, t-SNE, UMAP                             |
| Produccion             | joblib (serializacion de pipelines)                |
