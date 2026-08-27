# Tarea 3: Polars vs Pandas

**Curso:** Computación Paralela
**Profesor:** Johansell Villalobos Cubillo
**Fecha:** Junio 2026

## Descripción del Problema

Este proyecto construye un pipeline reproducible para predecir si un vuelo se retrasara. El objetivo es comparar el rendimiento de Polars y Pandas al ejecutar exactamente la misma logica de lectura, filtrado, agregacion, join e ingenieria de caracteristicas. Tambien se comparan Logistic Regression, Random Forest y XGBoost.

## Dataset

- Nombre: Airlines Dataset to Predict a Delay.
- Fuente: [Kaggle](https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay).
- Archivo esperado: `data/Airlines.csv`.
- Registros: 539,383.
- Columnas: 9.
- Target: `Delay` (`0` = sin retraso, `1` = retrasado).
- Valores faltantes observados: 0.

| Variable        | Tipo              | Descripcion                    |
| --------------- | ----------------- | ------------------------------ |
| `id`          | Numerica          | Identificador del registro     |
| `Airline`     | Categorica        | Codigo de la aerolinea         |
| `Flight`      | Numerica          | Numero de vuelo                |
| `AirportFrom` | Categorica        | Aeropuerto de origen           |
| `AirportTo`   | Categorica        | Aeropuerto de destino          |
| `DayOfWeek`   | Numerica discreta | Dia de la semana, de 1 a 7     |
| `Time`        | Numerica          | Minutos desde medianoche       |
| `Length`      | Numerica          | Duracion programada en minutos |
| `Delay`       | Binaria           | Variable objetivo              |

### Descarga y ubicación del dataset

El dataset utilizado es **Airlines Dataset to Predict a Delay**, disponible públicamente en Kaggle:

[Descargar dataset desde Kaggle](https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay)

Para ejecutar el proyecto:

1. Ingrese al enlace de Kaggle.
2. Seleccione **Download** para descargar el dataset.
3. Descomprima el archivo descargado.
4. Copie el archivo `Airlines.csv` dentro de la carpeta `data/` del proyecto.
5. Confirme que la ruta final sea:

```text
CP_Polars/data/Airlines.csv
```


## Decisiones de ingenieria

- Se validan `DayOfWeek` entre 1 y 7, `Time` entre 0 y 1439, `Length > 0` y `Delay` dentro de `{0, 1}`.
- El dataset no contiene nulos. El pipeline conserva una estrategia reproducible: moda para categoricas y mediana para numericas, por si aparecen nulos en ejecuciones futuras.
- El filtrado eliminó 4 registros con duración igual a cero, equivalentes aproximadamente al 0.0007 % del dataset. No se encontraron días, horas ni valores del target fuera de los rangos permitidos.
- Se crean `Hour`, `IsWeekend` y `DurationBin` mediante expresiones vectorizadas.
- `Length` y `Time` se estandarizan.
- Las categoricas se codifican con mappings deterministas y `replace_strict`.
- `AirlineFrequency` se calcula con `group_by` y se incorpora mediante `left join`.
- Logistic Regression utiliza `StandardScaler` para asegurar convergencia.
- El benchmark ejecuta cinco repeticiones y reporta el promedio.

## Requisitos e instalacion

Python 3.11.7 y las dependencias de `requirements.txt`. En PowerShell:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En VS Code, seleccione `.venv\Scripts\python.exe` como interprete y kernel.

## Ejecucion

Coloque `Airlines.csv` dentro de `data/` y ejecute desde la raiz:

```powershell
python src/polars_pipeline.py
python src/analysis.py
```

Como alternativa, el notebook puede abrirse en VS Code y ejecutarse directamente con `Run All`, ya que ejecuta el pipeline y el análisis en el orden requerido.

## Estructura del proyecto

```text
CP_Polars/
├── data/
│   └── Airlines.csv
├── figures/
│   ├── benchmark_comparison.png
│   ├── confusion_logistic_regression.png
│   ├── confusion_random_forest.png
│   ├── confusion_xgboost.png
│   ├── correlations.png
│   ├── eda_additional_distributions.png
│   ├── eda_distributions.png
│   ├── missing_values.png
│   ├── model_comparison.png
│   └── scalability.png
├── notebooks/
│   └── analysis.ipynb
├── report/
│   ├── analysis_report.md
│   └── report.pdf
├── results/
│   ├── airlines_processed.parquet
│   ├── benchmark_results.csv
│   ├── correlations.csv
│   ├── environment.json
│   ├── lazy_execution_results.csv
│   ├── missing_values.csv
│   ├── model_results.csv
│   └── scalability_results.csv
├── src/
│   ├── analysis.py
│   └── polars_pipeline.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Resultados Polars frente a Pandas

| Operación          | Polars (s) | Pandas (s) | Speedup |
| ------------------- | ---------: | ---------: | ------: |
| Lectura             |     0.0168 |     0.4875 |  29.03x |
| Filtrado            |     0.0067 |     0.0159 |   2.39x |
| Agregación          |     0.0032 |     0.0149 |   4.61x |
| Join                |     0.0097 |     0.0792 |   8.17x |
| Feature engineering |     0.0629 |     0.4664 |   7.41x |
| Pipeline total      |     0.0759 |     1.0039 |  13.23x |

El speedup pasó de 5.42x con el 25 % de los datos a 9.32x con el dataset completo. Lazy execution redujo marginalmente el tiempo, de 0.0250 segundos con `read_csv` a 0.0225 segundos con `scan_csv().collect()`, pero aumentó el pico de memoria de 20.69 MB a 24.08 MB. El archivo ocupa 17.66 MB y la consulta utiliza todas las columnas requeridas por la agregación.

## Resultados de Machine Learning

| Modelo              | Accuracy | F1 weighted |    AUC | Entrenamiento (s) |
| ------------------- | -------: | ----------: | -----: | ----------------: |
| Logistic Regression |   0.6256 |      0.6159 | 0.6563 |            0.5840 |
| Random Forest       |   0.6170 |      0.6164 | 0.6576 |            9.0964 |
| XGBoost             |   0.6620 |      0.6512 | 0.7134 |            2.9642 |

XGBoost produjo el mejor desempeño. Polars fue más rápido en las seis operaciones y logró un speedup de 13.23x en el pipeline total.

## Reproducibilidad

`results/environment.json` documenta CPU, RAM, sistema operativo, tamano del dataset, versiones y repeticiones. Los CSV de `results/` contienen los valores utilizados en tablas y graficas. El informe final integra pipeline, experimentos, diez preguntas de analisis y conclusiones.

## Referencias

- [Polars Documentation](https://docs.pola-rs.com/)
- [Pandas vs Polars](https://docs.pola-rs.com/user-guide/migration/pandas/)
- [Feature Engineering Guide](https://scikit-learn.org/stable/modules/preprocessing.html)

---

**Autor:** Carolina Salas
**Institución:** LEAD University  
**Fecha de entrega:** Junio 2026
