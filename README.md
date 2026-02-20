# ETL - Base de Datos de Empresas Activas
## Camara de Comercio de Ibague - Corte a 31 de Diciembre de 2025

**Autor:** Juan Camilo Perea Possos

---

## Descripcion

Proceso completo de **ETL (Extract, Transform, Load)** sobre la base de datos publica de empresas y/o entidades activas registradas en la jurisdiccion de la Camara de Comercio de Ibague, Colombia.

El dataset contiene **20,280 registros** con **83 variables** que incluyen informacion como razon social, NIT, actividad economica (CIIU), tamano empresarial, municipio, participacion de mujeres, entre otros.

## Estructura del Proyecto

```
Curso-IA/
├── README.md
├── requirements.txt
├── etlcamara.ipynb
├── .gitignore
└── data/
    └── BASE_DE_DATOS_DE_EMPRESAS_Y_O_ENTIDADES_ACTIVAS_...csv
```

## Contenido del Notebook

| Paso | Fase ETL | Descripcion |
|------|----------|-------------|
| 1 | - | Importacion de librerias |
| 2 | **Extract** | Carga del CSV original |
| 3 | - | Exploracion inicial del dataset |
| 4 | **Transform** | Limpieza de datos (nulos, duplicados, tipos, texto) |
| 5 | **Transform** | Analisis de correlacion y seleccion de variables |
| 6 | **Transform** | Creacion de variables derivadas (municipio, antiguedad, CIIU) |
| 7.1 | EDA | Histogramas (antiguedad, mujeres, socios) |
| 7.2 | EDA | Nube de palabras (razones sociales) |
| 7.3 | EDA | Graficos de barras (organizacion y tamano empresarial) |
| 8 | **Load** | Exportacion a CSV limpio |
| 9 | - | Resumen y conclusiones |

## Transformaciones Realizadas

- Reemplazo de valores "No reporta" y "No aplica" por NaN
- Eliminacion de filas duplicadas
- Conversion de fechas (YYYYMMDD → datetime)
- Conversion de columnas numericas (texto → numerico)
- Estandarizacion de texto (mayusculas, sin espacios extra)
- Eliminacion de columnas con >80% de valores nulos
- Eliminacion de variables redundantes por correlacion (>0.85)
- Creacion de variables: NOMBRE_MUNICIPIO, ANTIGUEDAD_ANOS, CIIU_CODIGO, SECCION_ECONOMICA

## Requisitos

- Python 3.12+
- Librerias listadas en `requirements.txt`

## Instalacion y Ejecucion

```bash
# Crear ambiente virtual
python -m venv venv

# Activar ambiente virtual (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt

# Abrir el notebook
jupyter notebook etlcamara.ipynb
```

## Fuente de Datos

Dataset publico de la Camara de Comercio de Ibague. Base de datos de empresas y/o entidades activas con corte al 31 de diciembre de 2025.
