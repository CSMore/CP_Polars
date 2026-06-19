# CP_Polars
# Tarea 3: Polars - Airline Flight Status Prediction

**Curso:** Computación Paralela  
**Profesor:** Johansell Villalobos Cubillo  
**Fecha:** Junio 2026

## Descripción del Problema

Este proyecto implementa un pipeline completo de análisis de datos y aprendizaje automático usando **Polars** para predecir el estado de vuelos de aerolíneas (On Time, Delayed, Cancelled). Se compara el rendimiento de Polars vs Pandas en diferentes etapas del procesamiento.

## Dataset

**Nombre:** Airline Flight Status Dataset  
**Registros:** 98,619  
**Variables:** 15 (numéricas y categóricas)  
**Target:** Flight Status (3 clases balanceadas: ~33% cada una)

### Variables principales:
- **Numéricas:** Age, Passenger ID
- **Categóricas:** Gender, Nationality, Airport Name, Airport Country Code, Airport Continent, Pilot Name
- **Temporal:** Departure Date, Arrival Airport
- **Target:** Flight Status (On Time, Delayed, Cancelled)

## Requisitos

```bash
pip install -r requirements.txt
```

## Estructura del Repositorio

```
tarea3_airlines/
├── data/
│   └── Airline_Dataset.csv          # Dataset original
├── src/
│   └── polars_pipeline.py            # Pipeline principal
├── notebooks/
│   └── analysis.ipynb               # Notebook Jupyter (opcional)
├── results/
│   └── benchmark_results.json       # Resultados de benchmarking
├── report/
│   └── report.pdf                   # Informe final
├── figures/
│   └── *.png                        # Gráficas generadas
├── requirements.txt
└── README.md
```

## Ejecución

### Opción 1: Script Python (recomendado - rápido)
```bash
cd tarea3_airlines
python src/polars_pipeline.py
```

### Opción 2: Google Colab (si necesitas GPU)
```python
# En una celda:
!git clone <tu-repo>
%cd tarea3_airlines
!pip install -r requirements.txt
!python src/polars_pipeline.py
```

## Workflow Implementado

### Parte 1: Exploración Explorativa (EDA) con Polars
- Distribución del target (Flight Status)
- Estadísticas descriptivas (Age)
- Análisis de valores nulos
- Conteo de categorías únicas

### Parte 2: Feature Engineering
- **Extracción temporal:** Day of Week, Month, Is Weekend
- **Encoding:** Label encoding para Gender, Nationality, Airport Continent
- **Nuevas características:**
  - `Pilot_Frequency`: Número de vuelos por piloto (grupo y agregación)
  - `Airport_Frequency`: Número de vuelos por aeropuerto (join implícito)
- **Manejo de datos:** Limpieza de valores faltantes

### Parte 3: Machine Learning
Entrenamiento de 3 modelos clasificadores:

1. **Logistic Regression**
   - Modelo lineal baseline
   - Rápido de entrenar

2. **Random Forest**
   - Maneja no-linealidades
   - Feature importance automático

3. **XGBoost**
   - Mejor rendimiento predictivo
   - Regularización integrada

**Métricas reportadas:**
- Accuracy
- Precision, Recall, F1-Score (weighted)
- Matriz de confusión
- Tiempo de entrenamiento

### Parte 4: Benchmark Polars vs Pandas
Se comparan operaciones clave:
- **Lectura:** `pl.read_csv()` vs `pd.read_csv()`
- **Filtrado:** Expresiones vectorizadas
- **Agregación:** `group_by()` + `agg()`

## Resultados Esperados

### Rendimiento de Modelos
```
Logistic Regression:
  ✓ Accuracy: ~0.33-0.35 (baseline - son 3 clases)
  ✓ F1-Score: ~0.33
  ✓ Tiempo: ~0.1-0.2s

Random Forest:
  ✓ Accuracy: ~0.35-0.40
  ✓ F1-Score: ~0.35-0.38
  ✓ Tiempo: ~1-2s

XGBoost:
  ✓ Accuracy: ~0.40-0.45 (mejor)
  ✓ F1-Score: ~0.39-0.42
  ✓ Tiempo: ~0.5-1s
```

### Speedup Polars vs Pandas
Esperado: **1.5x - 3x más rápido** en:
- Lectura: 1.5-2x
- Filtrado: 2-3x
- Agregación: 2-4x

## Análisis de Resultados

**Preguntas respondidas:**
1. ✅ Ventajas de Polars observadas
2. ✅ Operaciones con mayor speedup
3. ✅ Beneficios de Lazy Execution
4. ✅ Limitaciones encontradas
5. ✅ Comparación Polars vs Pandas
6. ✅ Justificación de migración
7. ✅ Efecto del tamaño de dataset
8. ✅ Mejor modelo (XGBoost probablemente)
9. ✅ Recomendaciones futuras

## Control de Versiones

Repository commits incluyen:
- Inicialización del proyecto
- EDA y análisis exploratorio
- Feature engineering
- Entrenamiento de modelos
- Benchmark final
- Análisis de resultados

## Trabajo Opcional

Para mejora de 5% en nota:
- Implementar Lazy Execution explícitamente
- Agregar Dask o RAPIDS cuDF
- Visualizaciones avanzadas (plotly, altair)

## Requisitos del Sistema

- **RAM:** 4 GB mínimo (recomendado 8 GB)
- **CPU:** 4+ cores (para paralelización efectiva)
- **Python:** 3.8+
- **Tiempo ejecución:** ~3-5 minutos (local), ~30s-1min (Colab con GPU)

## Referencias

- [Polars Documentation](https://docs.pola-rs.com/)
- [Pandas vs Polars](https://docs.pola-rs.com/user-guide/migration/pandas/)
- [Feature Engineering Guide](https://scikit-learn.org/stable/modules/preprocessing.html)

---

**Autor:** Caro  
**Instituto:** Lead University  
**Fecha entrega:** Junio 18, 2026
