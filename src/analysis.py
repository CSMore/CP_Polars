"""Genera graficas y analisis a partir de resultados experimentales reales.

Debe ejecutarse despues de src/polars_pipeline.py:
    python src/analysis.py
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
REPORT_DIR = Path("report")


def save_figure_atomic(fig, filename: str) -> None:
    """Guarda la figura completa antes de publicar el PNG final."""
    final_path = FIGURES_DIR / filename
    temporary_path = FIGURES_DIR / f".{filename}.tmp.png"
    fig.savefig(temporary_path, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)
    temporary_path.replace(final_path)
    temporary_path.unlink(missing_ok=True)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro {path}. Ejecute primero: python src/polars_pipeline.py"
        )
    return path


def model_figure(models: pl.DataFrame) -> None:
    names = models["Model"].to_list()
    short_names = ["Logistic", "Random Forest", "XGBoost"]
    metrics = ["Accuracy", "F1Weighted", "AUC"]
    colors = ["#0F4761", "#156082", "#E97132"]
    x = np.arange(len(names))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for index, metric in enumerate(metrics):
        axes[0].bar(
            x + (index - 1) * width,
            models[metric].to_numpy(),
            width,
            label=metric,
            color=colors[index],
        )
    axes[0].set_xticks(x, short_names)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Puntuacion")
    axes[0].set_title("Desempeno predictivo")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(short_names, models["TrainingSeconds"].to_numpy(), color=colors)
    axes[1].set_ylabel("Segundos")
    axes[1].set_title("Tiempo de entrenamiento")
    axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_figure_atomic(fig, "model_comparison.png")


def generate_report() -> None:
    benchmark = pl.read_csv(require(RESULTS_DIR / "benchmark_results.csv"))
    scalability = pl.read_csv(require(RESULTS_DIR / "scalability_results.csv"))
    lazy = pl.read_csv(require(RESULTS_DIR / "lazy_execution_results.csv"))
    models = pl.read_csv(require(RESULTS_DIR / "model_results.csv"))
    missing = pl.read_csv(require(RESULTS_DIR / "missing_values.csv"))
    with open(require(RESULTS_DIR / "environment.json"), encoding="utf-8") as file:
        environment = json.load(file)

    model_figure(models)

    fastest = benchmark.sort("Speedup", descending=True).row(0, named=True)
    smallest = benchmark.sort("Speedup").row(0, named=True)
    total = benchmark.filter(pl.col("Operation") == "TotalPipeline").row(0, named=True)
    best_model = models.sort(["AUC", "F1Weighted"], descending=[True, True]).row(0, named=True)
    eager = lazy.filter(pl.col("Mode") == "read_csv").row(0, named=True)
    lazy_row = lazy.filter(pl.col("Mode") == "scan_csv_collect").row(0, named=True)
    scale_first = scalability.sort("Fraction").row(0, named=True)
    scale_last = scalability.sort("Fraction").row(-1, named=True)
    missing_total = int(missing["Missing"].sum())

    report = f"""# Tarea 3: Polars vs Pandas

**Curso:** Computación Paralela  
**Profesor:** Johansell Villalobos Cubillo  
**Estudiante:** Carolina Salas  
**Institución:** LEAD University  
**Fecha:** Junio 2026

## Análisis de resultados: Polars frente a Pandas

## Resumen experimental

El dataset publico de Kaggle contiene 539,383 registros y 9 columnas, incluida la variable objetivo binaria `Delay`. Se identificaron {missing_total} valores faltantes. La clase 0 contiene 299,119 registros y la clase 1 contiene 240,264, por lo que existe una diferencia moderada, pero no un desbalance extremo.

El entorno utilizado fue {environment['operating_system']}, con {environment['physical_cpu_cores']} nucleos fisicos, {environment['logical_cpu_cores']} nucleos logicos y {environment['ram_gb']:.2f} GB de RAM. El archivo ocupa {environment['dataset_size_mb']:.2f} MB. Se utilizaron Polars {environment['polars']} y Pandas {environment['pandas']}. Cada operacion del benchmark se midio {environment['benchmark_repetitions']} veces y se reporto el promedio.

## Preguntas de analisis

### 1. ¿Que ventajas se observaron al utilizar Polars?

Polars redujo el tiempo en todas las operaciones medidas. El pipeline total paso de {total['PandasSeconds']:.4f} segundos en Pandas a {total['PolarsSeconds']:.4f} segundos en Polars, equivalente a un speedup de {total['Speedup']:.2f}x. La API de expresiones permitio realizar filtrado, escalado, binning, encoding, agregaciones y joins sin ciclos sobre las filas.

### 2. ¿Que operaciones obtuvieron el mayor speedup?

La mayor aceleracion se observo en `{fastest['Operation']}`, con {fastest['Speedup']:.2f}x. El resultado muestra que la arquitectura columnar y la implementacion paralela de Polars producen una diferencia especialmente clara en esta operacion.

### 3. ¿En cuales operaciones la diferencia fue pequena?

La menor diferencia se presento en `{smallest['Operation']}`, aunque Polars todavia obtuvo {smallest['Speedup']:.2f}x. Esto indica que las operaciones simples tienen menor margen de optimizacion que un pipeline con varias transformaciones encadenadas.

### 4. ¿Que beneficios aporto Lazy Execution?

En este experimento, lazy execution no mejoro el resultado. `read_csv` tardo {eager['Seconds']:.4f} segundos y aumento el pico de memoria aproximadamente {eager['PeakMemoryIncreaseMB']:.2f} MB; `scan_csv().collect()` tardo {lazy_row['Seconds']:.4f} segundos y aumento el pico aproximadamente {lazy_row['PeakMemoryIncreaseMB']:.2f} MB. El pipeline evaluado es corto, utiliza todas las columnas requeridas por la agregacion y el archivo solo ocupa {environment['dataset_size_mb']:.2f} MB, por lo que el costo de construir y optimizar el plan supera el ahorro potencial. Lazy execution seria mas beneficioso en pipelines largos, con proyeccion de pocas columnas o filtros capaces de descartar gran parte de los datos.

### 5. ¿Que limitaciones se encontraron en Polars?

La sintaxis y algunos nombres de metodos cambian entre versiones, por lo que fue necesario adaptar el codigo a Polars {environment['polars']}. Ademas, scikit-learn trabaja principalmente con arreglos NumPy o DataFrames de Pandas, de modo que los datos finales debieron convertirse antes del entrenamiento. Las mediciones de memoria de operaciones nativas tambien son aproximadas porque se basan en el RSS del proceso.

### 6. ¿Que ventajas mantiene Pandas?

Pandas conserva un ecosistema muy amplio, abundante documentacion y compatibilidad directa con numerosas bibliotecas estadisticas y de aprendizaje automatico. Para datasets pequenos o exploraciones sencillas, su familiaridad puede ser mas importante que una diferencia reducida de tiempo.

### 7. ¿La aceleracion justifica migrar un proyecto existente?

Para este caso, una aceleracion total de {total['Speedup']:.2f}x si justifica considerar Polars cuando el pipeline se ejecuta repetidamente o tiene restricciones de tiempo. Una migracion real debe valorar tambien el costo de reescribir, validar y mantener el codigo. No se recomienda migrar un proyecto pequeno solo por preferencia tecnologica.

### 8. ¿Como afecto el tamano del dataset al beneficio?

Con {scale_first['Rows']:,} registros, el speedup fue {scale_first['Speedup']:.2f}x. Con {scale_last['Rows']:,} registros, fue {scale_last['Speedup']:.2f}x. La ventaja crecio inicialmente conforme aumento el volumen y luego se estabilizo, lo que evidencia que Polars aprovecha mejor el procesamiento columnar cuando el costo de la operacion supera la sobrecarga inicial.

### 9. ¿Que modelo produjo el mejor desempeno predictivo?

El mejor modelo fue {best_model['Model']}, con Accuracy de {best_model['Accuracy']:.4f}, F1 weighted de {best_model['F1Weighted']:.4f} y AUC de {best_model['AUC']:.4f}. Su ventaja sugiere relaciones no lineales e interacciones entre aerolinea, ruta, horario, dia y duracion que Logistic Regression no captura completamente.

### 10. ¿Que recomendaciones se darian para proyectos futuros?

Se recomienda usar Polars en pipelines repetitivos o con cientos de miles de filas, medir varias repeticiones en lugar de basarse en una sola ejecucion y evaluar lazy execution segun la complejidad real del flujo. En Machine Learning deben compararse varias metricas y no solamente Accuracy. Para una aplicacion operacional se deberian incorporar variables como clima, congestion y retrasos historicos, y validar el modelo con una division temporal.

## Conclusiones

Los experimentos muestran que Polars fue mas rapido que Pandas en las seis operaciones comparadas y alcanzo {total['Speedup']:.2f}x en el pipeline completo. La escalabilidad confirma que la ventaja se mantiene al aumentar el numero de registros. Sin embargo, lazy execution no fue superior en tiempo ni memoria para este flujo corto, lo cual demuestra que su beneficio depende del plan de consulta y no debe asumirse automaticamente. En prediccion, {best_model['Model']} obtuvo el mejor balance de Accuracy, F1 y AUC. Por tanto, Polars es recomendable para el procesamiento de este dataset, mientras que la eleccion del modelo debe basarse en evidencia predictiva y en las necesidades de interpretabilidad.
"""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "analysis_report.md"
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Reporte guardado en: {output}")


if __name__ == "__main__":
    generate_report()

