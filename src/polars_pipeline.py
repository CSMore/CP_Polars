"""Pipeline reproducible para prediccion de retrasos de vuelos.

Dataset: Airlines Dataset to Predict a Delay (Kaggle), 539 383 registros.
Ejecutar desde la raiz del proyecto con:
    python src/polars_pipeline.py
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import psutil
import seaborn as sns
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


DATA_PATH = Path("data/Airlines.csv")
FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("results")
REPORT_DIR = Path("report")
RANDOM_STATE = 42
BENCHMARK_REPETITIONS = 5

CATEGORICAL_COLUMNS = ["Airline", "AirportFrom", "AirportTo"]
NUMERIC_COLUMNS = ["Flight", "DayOfWeek", "Time", "Length", "Delay"]
FEATURE_COLUMNS = [
    "Flight",
    "DayOfWeek",
    "Hour",
    "IsWeekend",
    "LengthScaled",
    "TimeScaled",
    "DurationBin",
    "AirlineEncoded",
    "AirportFromEncoded",
    "AirportToEncoded",
    "AirlineFrequency",
]


def ensure_directories() -> None:
    for directory in (FIGURES_DIR, RESULTS_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_figure_atomic(fig, filename: str) -> None:
    """Guarda una figura completa antes de publicar el PNG final."""
    final_path = FIGURES_DIR / filename
    temporary_path = FIGURES_DIR / f".{filename}.tmp.png"
    fig.savefig(temporary_path, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)
    temporary_path.replace(final_path)
    temporary_path.unlink(missing_ok=True)


def cleanup_temporary_figures() -> None:
    """Elimina temporales que pudiera dejar el sistema de sincronizacion."""
    for temporary_path in FIGURES_DIR.glob(".*.tmp.png"):
        temporary_path.unlink(missing_ok=True)


def build_mappings_polars(df: pl.DataFrame) -> dict[str, dict[str, int]]:
    """Construye mappings deterministas para las variables categoricas."""
    mappings: dict[str, dict[str, int]] = {}
    for column in CATEGORICAL_COLUMNS:
        values = df.get_column(column).drop_nulls().unique().sort().to_list()
        mappings[column] = {value: index for index, value in enumerate(values)}
    return mappings


def clean_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Imputa faltantes y filtra valores fuera de los rangos documentados."""
    expressions: list[pl.Expr] = []
    for column in CATEGORICAL_COLUMNS:
        mode = df.get_column(column).drop_nulls().mode()
        fill_value = mode[0] if len(mode) else "UNKNOWN"
        expressions.append(pl.col(column).fill_null(fill_value))
    for column in ["Flight", "DayOfWeek", "Time", "Length"]:
        median = df.get_column(column).median()
        expressions.append(pl.col(column).fill_null(median))

    return (
        df.with_columns(expressions)
        .filter(
            pl.col("DayOfWeek").is_between(1, 7)
            & pl.col("Time").is_between(0, 1439)
            & (pl.col("Length") > 0)
            & pl.col("Delay").is_in([0, 1])
        )
    )


def feature_engineering_polars(
    df: pl.DataFrame,
    mappings: dict[str, dict[str, int]] | None = None,
) -> pl.DataFrame:
    """Aplica transformaciones vectorizadas, group_by y join en Polars."""
    clean = clean_polars(df)
    mappings = mappings or build_mappings_polars(clean)
    length_mean = clean.get_column("Length").mean()
    length_std = clean.get_column("Length").std() or 1.0
    time_mean = clean.get_column("Time").mean()
    time_std = clean.get_column("Time").std() or 1.0

    airline_frequency = clean.group_by("Airline").agg(
        pl.len().alias("AirlineFrequency")
    )

    transformed = (
        clean.with_columns(
            (pl.col("Time") // 60).cast(pl.Int16).alias("Hour"),
            pl.col("DayOfWeek").is_in([6, 7]).cast(pl.Int8).alias("IsWeekend"),
            pl.when(pl.col("Length") < 120)
            .then(pl.lit(0))
            .when(pl.col("Length") < 240)
            .then(pl.lit(1))
            .otherwise(pl.lit(2))
            .cast(pl.Int8)
            .alias("DurationBin"),
            ((pl.col("Length") - length_mean) / length_std).alias("LengthScaled"),
            ((pl.col("Time") - time_mean) / time_std).alias("TimeScaled"),
            pl.col("Airline")
            .replace_strict(mappings["Airline"], default=-1)
            .cast(pl.Int32)
            .alias("AirlineEncoded"),
            pl.col("AirportFrom")
            .replace_strict(mappings["AirportFrom"], default=-1)
            .cast(pl.Int32)
            .alias("AirportFromEncoded"),
            pl.col("AirportTo")
            .replace_strict(mappings["AirportTo"], default=-1)
            .cast(pl.Int32)
            .alias("AirportToEncoded"),
        )
        .join(airline_frequency, on="Airline", how="left")
    )
    return transformed


def clean_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Replica exactamente la limpieza aplicada en Polars."""
    clean = df.copy()
    for column in CATEGORICAL_COLUMNS:
        mode = clean[column].mode(dropna=True)
        clean[column] = clean[column].fillna(mode.iloc[0] if not mode.empty else "UNKNOWN")
    for column in ["Flight", "DayOfWeek", "Time", "Length"]:
        clean[column] = clean[column].fillna(clean[column].median())
    return clean.loc[
        clean["DayOfWeek"].between(1, 7)
        & clean["Time"].between(0, 1439)
        & clean["Length"].gt(0)
        & clean["Delay"].isin([0, 1])
    ].copy()


def feature_engineering_pandas(
    df: pd.DataFrame,
    mappings: dict[str, dict[str, int]],
) -> pd.DataFrame:
    """Replica la misma logica del pipeline Polars en Pandas."""
    clean = clean_pandas(df)
    airline_frequency = (
        clean.groupby("Airline", as_index=False).size().rename(columns={"size": "AirlineFrequency"})
    )
    clean["Hour"] = (clean["Time"] // 60).astype("int16")
    clean["IsWeekend"] = clean["DayOfWeek"].isin([6, 7]).astype("int8")
    clean["DurationBin"] = np.select(
        [clean["Length"].lt(120), clean["Length"].lt(240)], [0, 1], default=2
    ).astype("int8")
    clean["LengthScaled"] = (clean["Length"] - clean["Length"].mean()) / clean["Length"].std(ddof=1)
    clean["TimeScaled"] = (clean["Time"] - clean["Time"].mean()) / clean["Time"].std(ddof=1)
    for column in CATEGORICAL_COLUMNS:
        clean[f"{column}Encoded"] = clean[column].map(mappings[column]).fillna(-1).astype("int32")
    return clean.merge(airline_frequency, on="Airline", how="left")


def create_eda(df: pl.DataFrame) -> None:
    """Genera estadisticas, faltantes, distribuciones y correlaciones desde Polars."""
    print("\nESTADISTICAS DESCRIPTIVAS")
    print(df.select(NUMERIC_COLUMNS).describe())
    missing = df.null_count().transpose(
        include_header=True, header_name="Variable", column_names=["Missing"]
    )
    missing = missing.with_columns(
        (pl.col("Missing") / df.height * 100).alias("MissingPercent")
    )
    missing.write_csv(RESULTS_DIR / "missing_values.csv")

    plt.figure(figsize=(10, 5))
    plt.bar(
        missing["Variable"].to_list(),
        missing["MissingPercent"].to_list(),
        color="#0F4761",
    )
    plt.xlabel("Variable")
    plt.ylabel("Valores faltantes (%)")
    plt.title("Porcentaje de valores faltantes por variable")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_figure_atomic(plt.gcf(), "missing_values.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].hist(df.get_column("Length").to_numpy(), bins=35, color="#156082")
    axes[0, 0].set(title="Distribucion de duracion", xlabel="Minutos", ylabel="Frecuencia")
    axes[0, 1].hist(df.get_column("Time").to_numpy(), bins=24, color="#E97132")
    axes[0, 1].set(title="Distribucion de hora programada", xlabel="Minutos desde medianoche", ylabel="Frecuencia")
    delay_counts = df.group_by("Delay").len().sort("Delay")
    axes[1, 0].bar(delay_counts["Delay"].cast(pl.String).to_list(), delay_counts["len"].to_list(), color="#0F4761")
    axes[1, 0].set(title="Distribucion de la variable objetivo", xlabel="Delay", ylabel="Vuelos")
    airline_counts = df.group_by("Airline").len().sort("len", descending=True).head(15)
    axes[1, 1].bar(airline_counts["Airline"].to_list(), airline_counts["len"].to_list(), color="#A02B93")
    axes[1, 1].set(title="Quince aerolineas con mas vuelos", xlabel="Aerolinea", ylabel="Vuelos")
    axes[1, 1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    save_figure_atomic(fig, "eda_distributions.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(df.get_column("Flight").to_numpy(), bins=35, color="#156082")
    axes[0, 0].set(
        title="Distribucion del numero de vuelo",
        xlabel="Numero de vuelo",
        ylabel="Frecuencia",
    )
    day_counts = df.group_by("DayOfWeek").len().sort("DayOfWeek")
    axes[0, 1].bar(
        day_counts["DayOfWeek"].cast(pl.String).to_list(),
        day_counts["len"].to_list(),
        color="#E97132",
    )
    axes[0, 1].set(
        title="Vuelos por dia de la semana",
        xlabel="Dia de la semana",
        ylabel="Vuelos",
    )
    origin_counts = (
        df.group_by("AirportFrom").len().sort("len", descending=True).head(15)
    )
    axes[1, 0].bar(
        origin_counts["AirportFrom"].to_list(),
        origin_counts["len"].to_list(),
        color="#A02B93",
    )
    axes[1, 0].set(
        title="Quince aeropuertos de origen con mas vuelos",
        xlabel="Aeropuerto de origen",
        ylabel="Vuelos",
    )
    destination_counts = (
        df.group_by("AirportTo").len().sort("len", descending=True).head(15)
    )
    axes[1, 1].bar(
        destination_counts["AirportTo"].to_list(),
        destination_counts["len"].to_list(),
        color="#70AD47",
    )
    axes[1, 1].set(
        title="Quince aeropuertos de destino con mas vuelos",
        xlabel="Aeropuerto de destino",
        ylabel="Vuelos",
    )
    for axis in (axes[1, 0], axes[1, 1]):
        axis.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    save_figure_atomic(fig, "eda_additional_distributions.png")

    corr = df.select(NUMERIC_COLUMNS).corr()
    corr.write_csv(RESULTS_DIR / "correlations.csv")
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr.to_numpy(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                xticklabels=NUMERIC_COLUMNS, yticklabels=NUMERIC_COLUMNS)
    plt.title("Correlaciones entre variables numericas")
    plt.tight_layout()
    save_figure_atomic(plt.gcf(), "correlations.png")


def elapsed(function, repetitions: int = BENCHMARK_REPETITIONS) -> tuple[float, object]:
    """Devuelve promedio de tiempo y resultado de la ultima repeticion."""
    times = []
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - start)
    return statistics.mean(times), result


def benchmark(filepath: Path) -> pl.DataFrame:
    """Compara operaciones equivalentes y promedia cinco repeticiones."""
    pldf = pl.read_csv(filepath)
    pdf = pd.read_csv(filepath)
    mappings = build_mappings_polars(pldf)
    rows: list[dict[str, float | str]] = []

    operations = {
        "Read": (lambda: pl.read_csv(filepath), lambda: pd.read_csv(filepath)),
        "Filter": (
            lambda: pldf.filter(pl.col("Delay") == 1),
            lambda: pdf.loc[pdf["Delay"] == 1],
        ),
        "Aggregation": (
            lambda: pldf.group_by("Airline").agg(pl.len().alias("Flights")),
            lambda: pdf.groupby("Airline", as_index=False).size(),
        ),
        "Join": (
            lambda: pldf.join(pldf.group_by("Airline").agg(pl.len().alias("AirlineFrequency")), on="Airline", how="left"),
            lambda: pdf.merge(pdf.groupby("Airline", as_index=False).size().rename(columns={"size": "AirlineFrequency"}), on="Airline", how="left"),
        ),
        "FeatureEngineering": (
            lambda: feature_engineering_polars(pldf, mappings),
            lambda: feature_engineering_pandas(pdf, mappings),
        ),
        "TotalPipeline": (
            lambda: feature_engineering_polars(pl.read_csv(filepath), mappings),
            lambda: feature_engineering_pandas(pd.read_csv(filepath), mappings),
        ),
    }
    for operation, (polars_fn, pandas_fn) in operations.items():
        polars_time, _ = elapsed(polars_fn)
        pandas_time, _ = elapsed(pandas_fn)
        rows.append({
            "Operation": operation,
            "PolarsSeconds": polars_time,
            "PandasSeconds": pandas_time,
            "Speedup": pandas_time / polars_time,
        })
    results = pl.DataFrame(rows)
    results.write_csv(RESULTS_DIR / "benchmark_results.csv")
    return results


def scalability(filepath: Path) -> pl.DataFrame:
    """Mide ambos pipelines con 25, 50, 75 y 100 por ciento de los datos."""
    full_pl = pl.read_csv(filepath)
    full_pd = pd.read_csv(filepath)
    mappings = build_mappings_polars(full_pl)
    rows = []
    for fraction in [0.25, 0.50, 0.75, 1.00]:
        n_rows = int(full_pl.height * fraction)
        subset_pl = full_pl.head(n_rows)
        subset_pd = full_pd.head(n_rows)
        polars_time, _ = elapsed(lambda: feature_engineering_polars(subset_pl, mappings), 3)
        pandas_time, _ = elapsed(lambda: feature_engineering_pandas(subset_pd, mappings), 3)
        rows.append({
            "Fraction": int(fraction * 100),
            "Rows": n_rows,
            "PolarsSeconds": polars_time,
            "PandasSeconds": pandas_time,
            "Speedup": pandas_time / polars_time,
        })
    results = pl.DataFrame(rows)
    results.write_csv(RESULTS_DIR / "scalability_results.csv")
    return results


def measure_peak_memory(function) -> tuple[float, float]:
    """Mide tiempo y aumento aproximado del pico RSS del proceso."""
    try:
        process = psutil.Process(os.getpid())
        baseline = process.memory_info().rss
    except psutil.Error:
        process = None
        baseline = psutil.virtual_memory().used
    peak = baseline
    running = True

    def current_memory() -> int:
        if process is not None:
            try:
                return process.memory_info().rss
            except psutil.Error:
                pass
        return psutil.virtual_memory().used

    def sample() -> None:
        nonlocal peak
        while running:
            peak = max(peak, current_memory())
            time.sleep(0.002)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    start = time.perf_counter()
    result = function()
    seconds = time.perf_counter() - start
    running = False
    monitor.join()
    del result
    return seconds, max(0, peak - baseline) / (1024 ** 2)


def lazy_experiment(filepath: Path) -> pl.DataFrame:
    """Compara read_csv eager con scan_csv.collect usando el mismo pipeline."""
    eager_time, eager_memory = measure_peak_memory(
        lambda: pl.read_csv(filepath)
        .filter(pl.col("Length") > 0)
        .group_by("Airline")
        .agg(pl.len().alias("Flights"), pl.col("Length").mean().alias("MeanLength"))
    )
    lazy_time, lazy_memory = measure_peak_memory(
        lambda: pl.scan_csv(filepath)
        .filter(pl.col("Length") > 0)
        .group_by("Airline")
        .agg(pl.len().alias("Flights"), pl.col("Length").mean().alias("MeanLength"))
        .collect()
    )
    result = pl.DataFrame({
        "Mode": ["read_csv", "scan_csv_collect"],
        "Seconds": [eager_time, lazy_time],
        "PeakMemoryIncreaseMB": [eager_memory, lazy_memory],
    })
    result.write_csv(RESULTS_DIR / "lazy_execution_results.csv")
    return result


def train_models(df: pl.DataFrame) -> pl.DataFrame:
    """Entrena tres modelos y guarda metricas y matrices de confusion."""
    model_data = df.select(FEATURE_COLUMNS + ["Delay"]).to_pandas()
    X = model_data[FEATURE_COLUMNS]
    y = model_data["Delay"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                                 random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss"),
    }
    rows = []
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start
        prediction = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, prediction),
            "F1Weighted": f1_score(y_test, prediction, average="weighted"),
            "PrecisionWeighted": precision_score(y_test, prediction, average="weighted", zero_division=0),
            "RecallWeighted": recall_score(y_test, prediction, average="weighted"),
            "AUC": roc_auc_score(y_test, probability),
            "TrainingSeconds": training_time,
        })
        display = ConfusionMatrixDisplay.from_predictions(
            y_test,
            prediction,
            cmap="Blues",
            values_format="d",
        )
        display.ax_.set_title(f"Matriz de confusion - {name}")
        display.figure_.tight_layout()
        save_figure_atomic(
            display.figure_,
            f"confusion_{name.lower().replace(' ', '_')}.png",
        )
    results = pl.DataFrame(rows)
    results.write_csv(RESULTS_DIR / "model_results.csv")
    return results


def create_comparison_figures(benchmark_df: pl.DataFrame, scalability_df: pl.DataFrame) -> None:
    operations = benchmark_df["Operation"].to_list()
    x = np.arange(len(operations))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].bar(
        x - width / 2,
        benchmark_df["PolarsSeconds"].to_numpy(),
        width,
        label="Polars",
        color="#0F4761",
    )
    axes[0].bar(
        x + width / 2,
        benchmark_df["PandasSeconds"].to_numpy(),
        width,
        label="Pandas",
        color="#E97132",
    )
    axes[0].set(title="Tiempo promedio por operacion", ylabel="Segundos", xticks=x, xticklabels=operations)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend()
    axes[1].barh(
        operations,
        benchmark_df["Speedup"].to_numpy(),
        color="#156082",
    )
    axes[1].axvline(1, color="red", linestyle="--", label="Mismo rendimiento")
    axes[1].set(title="Speedup Pandas / Polars", xlabel="Veces")
    axes[1].legend()
    plt.tight_layout()
    save_figure_atomic(fig, "benchmark_comparison.png")

    fig, axis = plt.subplots(figsize=(9, 6))
    rows = scalability_df["Rows"].to_numpy()
    axis.plot(
        rows,
        scalability_df["PolarsSeconds"].to_numpy(),
        marker="o",
        label="Polars",
    )
    axis.plot(
        rows,
        scalability_df["PandasSeconds"].to_numpy(),
        marker="o",
        label="Pandas",
    )
    axis.set_xlabel("Cantidad de registros")
    axis.set_ylabel("Tiempo promedio (segundos)")
    axis.set_title("Escalabilidad del pipeline")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    save_figure_atomic(fig, "scalability.png")


def save_environment(filepath: Path) -> dict[str, object]:
    environment = {
        "operating_system": platform.platform(),
        "processor": platform.processor() or "No reportado por el sistema",
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "dataset_size_mb": round(filepath.stat().st_size / (1024 ** 2), 2),
        "python": platform.python_version(),
        "polars": pl.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "numpy": np.__version__,
        "benchmark_repetitions": BENCHMARK_REPETITIONS,
    }
    with open(RESULTS_DIR / "environment.json", "w", encoding="utf-8") as file:
        json.dump(environment, file, indent=2, ensure_ascii=False)
    return environment


def main() -> None:
    ensure_directories()
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontro el dataset: {DATA_PATH}")

    print("Cargando y validando dataset")
    raw = pl.read_csv(DATA_PATH)
    print(f"Registros: {raw.height:,}; columnas: {raw.width}")
    print(raw.null_count())
    create_eda(raw)

    mappings = build_mappings_polars(raw)
    processed = feature_engineering_polars(raw, mappings)
    processed.write_parquet(RESULTS_DIR / "airlines_processed.parquet")

    print("Ejecutando benchmark")
    benchmark_results = benchmark(DATA_PATH)
    print(benchmark_results)

    print("Ejecutando prueba de escalabilidad")
    scalability_results = scalability(DATA_PATH)
    print(scalability_results)

    print("Comparando eager y lazy execution")
    lazy_results = lazy_experiment(DATA_PATH)
    print(lazy_results)

    print("Entrenando modelos")
    model_results = train_models(processed)
    print(model_results)

    create_comparison_figures(benchmark_results, scalability_results)
    cleanup_temporary_figures()
    environment = save_environment(DATA_PATH)
    print(json.dumps(environment, indent=2, ensure_ascii=False))
    print("Proceso completado. Revise las carpetas figures y results.")


if __name__ == "__main__":
    main()