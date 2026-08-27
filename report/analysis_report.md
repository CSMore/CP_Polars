# Tarea 3: Polars vs Pandas

**Curso:** Computación Paralela  
**Profesor:** Johansell Villalobos Cubillo  
**Estudiante:** Carolina Salas  
**Institución:** LEAD University  
**Fecha:** Junio 2026

## Análisis de resultados: Polars frente a Pandas

## Resumen experimental

El dataset publico de Kaggle contiene 539,383 registros y 9 columnas, incluida la variable objetivo binaria `Delay`. Se identificaron 0 valores faltantes. La clase 0 contiene 299,119 registros y la clase 1 contiene 240,264, por lo que existe una diferencia moderada, pero no un desbalance extremo.

El entorno utilizado fue Windows-10-10.0.26200-SP0, con 10 nucleos fisicos, 16 nucleos logicos y 31.63 GB de RAM. El archivo ocupa 17.66 MB. Se utilizaron Polars 1.44.1 y Pandas 3.0.5. Cada operacion del benchmark se midio 5 veces y se reporto el promedio.

## Preguntas de analisis

### 1. ¿Que ventajas se observaron al utilizar Polars?

Polars redujo el tiempo en todas las operaciones medidas. El pipeline total paso de 1.0039 segundos en Pandas a 0.0759 segundos en Polars, equivalente a un speedup de 13.23x. La API de expresiones permitio realizar filtrado, escalado, binning, encoding, agregaciones y joins sin ciclos sobre las filas.

### 2. ¿Que operaciones obtuvieron el mayor speedup?

La mayor aceleracion se observo en `Read`, con 29.03x. El resultado muestra que la arquitectura columnar y la implementacion paralela de Polars producen una diferencia especialmente clara en esta operacion.

### 3. ¿En cuales operaciones la diferencia fue pequena?

La menor diferencia se presento en `Filter`, aunque Polars todavia obtuvo 2.39x. Esto indica que las operaciones simples tienen menor margen de optimizacion que un pipeline con varias transformaciones encadenadas.

### 4. ¿Que beneficios aporto Lazy Execution?

En este experimento, lazy execution redujo marginalmente el tiempo: `read_csv` tardo 0.0250 segundos y aumento el pico de memoria aproximadamente 20.69 MB; `scan_csv().collect()` tardo 0.0225 segundos y aumento el pico aproximadamente 24.08 MB. Sin embargo, el modo lazy uso mas memoria en esta medicion. El pipeline evaluado es corto, utiliza todas las columnas requeridas por la agregacion y el archivo solo ocupa 17.66 MB, por lo que la diferencia temporal es pequena. Lazy execution seria mas beneficioso en pipelines largos, con proyeccion de pocas columnas o filtros capaces de descartar gran parte de los datos.

### 5. ¿Que limitaciones se encontraron en Polars?

La sintaxis y algunos nombres de metodos cambian entre versiones, por lo que fue necesario adaptar el codigo a Polars 1.44.1. Ademas, scikit-learn trabaja principalmente con arreglos NumPy o DataFrames de Pandas, de modo que los datos finales debieron convertirse antes del entrenamiento. Las mediciones de memoria de operaciones nativas tambien son aproximadas porque se basan en el RSS del proceso.

### 6. ¿Que ventajas mantiene Pandas?

Pandas conserva un ecosistema muy amplio, abundante documentacion y compatibilidad directa con numerosas bibliotecas estadisticas y de aprendizaje automatico. Para datasets pequenos o exploraciones sencillas, su familiaridad puede ser mas importante que una diferencia reducida de tiempo.

### 7. ¿La aceleracion justifica migrar un proyecto existente?

Para este caso, una aceleracion total de 13.23x si justifica considerar Polars cuando el pipeline se ejecuta repetidamente o tiene restricciones de tiempo. Una migracion real debe valorar tambien el costo de reescribir, validar y mantener el codigo. No se recomienda migrar un proyecto pequeno solo por preferencia tecnologica.

### 8. ¿Como afecto el tamano del dataset al beneficio?

Con 134,845 registros, el speedup fue 5.42x. Con 539,383 registros, fue 9.32x. La ventaja crecio inicialmente conforme aumento el volumen y luego se estabilizo, lo que evidencia que Polars aprovecha mejor el procesamiento columnar cuando el costo de la operacion supera la sobrecarga inicial.

### 9. ¿Que modelo produjo el mejor desempeno predictivo?

El mejor modelo fue XGBoost, con Accuracy de 0.6620, F1 weighted de 0.6512 y AUC de 0.7134. Su ventaja sugiere relaciones no lineales e interacciones entre aerolinea, ruta, horario, dia y duracion que Logistic Regression no captura completamente.

### 10. ¿Que recomendaciones se darian para proyectos futuros?

Se recomienda usar Polars en pipelines repetitivos o con cientos de miles de filas, medir varias repeticiones en lugar de basarse en una sola ejecucion y evaluar lazy execution segun la complejidad real del flujo. En Machine Learning deben compararse varias metricas y no solamente Accuracy. Para una aplicacion operacional se deberian incorporar variables como clima, congestion y retrasos historicos, y validar el modelo con una division temporal.

## Conclusiones

Los experimentos muestran que Polars fue mas rapido que Pandas en las seis operaciones comparadas y alcanzo 13.23x en el pipeline completo. La escalabilidad confirma que la ventaja se mantiene al aumentar el numero de registros. Lazy execution fue ligeramente mas rapido, pero utilizo mas memoria para este flujo corto, lo cual demuestra que su beneficio depende del plan de consulta y no debe asumirse automaticamente. En prediccion, XGBoost obtuvo el mejor balance de Accuracy, F1 y AUC. Por tanto, Polars es recomendable para el procesamiento de este dataset, mientras que la eleccion del modelo debe basarse en evidencia predictiva y en las necesidades de interpretabilidad.
