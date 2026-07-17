import json
import os
import matplotlib.pyplot as plt
import seaborn as sns


def load_metrics(json_path: str) -> dict:
    """Carga y parsea el archivo JSON de métricas de spaCy."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"El archivo no fue encontrado en: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def plot_spacy_performance(metrics: dict, output_image_path: str = None):
    """Genera gráficos de rendimiento para spaCy compatibles con DistilBERT."""
    # 1. Configuración de estilo idéntica para garantizar comparabilidad
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 14})

    # 2. Extracción de datos específicos de spaCy
    train_loss = metrics.get("train_loss", [])
    val_f1 = metrics.get("val_f1", [])
    model_type = metrics.get("model_type", "spaCy").upper()
    timestamp = metrics.get("timestamp", "")

    epochs = range(1, len(train_loss) + 1)

    # 3. Inicializar la estructura de subgráficos idéntica (1x2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Monitoreo de Entrenamiento - {model_type} ({timestamp})",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # --- SUBGRÁFICO 1: Curva de Pérdida ---
    ax1.plot(
        epochs,
        train_loss,
        label="Train Loss",
        color="#1f77b4",
        linewidth=2.5,
        marker="o",
        markersize=2,  # Marcadores más pequeños debido a que spaCy tiene 100 épocas
    )
    ax1.set_title("Curva de Pérdida (Loss Summary)", fontweight="semibold")
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Pérdida Absoluta (Loss)")
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_xlim(1, len(train_loss))

    # --- SUBGRÁFICO 2: Métrica de Validación (F1-Score) ---
    if val_f1:
        ax2.plot(
            epochs,
            val_f1,
            label="Val F1-Score",
            color="#2ca02c",
            linewidth=2.5,
            marker="v",
            markersize=2,
        )

        # Identificar visualmente el mejor F1-Score obtenido en las 100 épocas
        max_f1_idx = val_f1.index(max(val_f1))
        ax2.axvline(
            x=max_f1_idx + 1,
            color="#9467bd",
            linestyle=":",
            linewidth=2,
            label=f"Máx. F1: {val_f1[max_f1_idx]:.4f} (Época {max_f1_idx+1})",
        )

    ax2.set_title("Métrica de Clasificación (F1-Score)", fontweight="semibold")
    ax2.set_xlabel("Épocas")
    ax2.set_ylabel("Score (0.0 - 1.0)")
    ax2.set_ylim(0.5, 1.05)  # Rango idéntico al de DistilBERT para una comparación justa
    ax2.legend(loc="lower right", frameon=True)
    ax2.set_xlim(1, len(train_loss))

    plt.tight_layout()

    # 4. Guardar o Mostrar el gráfico resultante
    if output_image_path:
        plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Gráfico de spaCy guardado en: {output_image_path}")
    else:
        plt.show()


if __name__ == "__main__":
    BASE_DIR = r"C:\Users\HP\Documents\Tesis\Programas\Ner\TesisNer\Version 2.0"
    OUTPUT_SUBDIR = os.path.join(BASE_DIR, "output")

    # Archivo de destino solicitado para spaCy
    FILE_NAME = "metrics_20260525_194453.json"
    IMAGE_NAME = "reporte_performance_spacy.png"

    JSON_FILE = os.path.join(OUTPUT_SUBDIR, FILE_NAME)
    OUTPUT_IMAGE = os.path.join(OUTPUT_SUBDIR, IMAGE_NAME)

    try:
        print(f"[PROCESO] Buscando datos de spaCy en: {JSON_FILE}")
        raw_metrics = load_metrics(JSON_FILE)

        print(f"[PROCESO] Generando gráficos comparables...")
        plot_spacy_performance(raw_metrics, output_image_path=OUTPUT_IMAGE)

    except FileNotFoundError as fnf_error:
        print(f"[ERROR DE RUTA] {fnf_error}")
        print(
            "[CONSEJO] Verifica que el JSON de spaCy esté en la carpeta 'output'."
        )
    except Exception as e:
        print(f"[ERROR] Ocurrió un fallo inesperado en la ejecución: {e}")