import json
import os
import matplotlib.pyplot as plt
import seaborn as sns


def load_metrics(json_path: str) -> dict:
    """Carga y parsea el archivo JSON de métricas.

    Args:
        json_path (str): Ruta al archivo JSON.

    Returns:
        dict: Diccionario con los datos del JSON.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"El archivo no fue encontrado en: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def plot_training_performance(metrics: dict, output_image_path: str = None):
    """Genera y guarda gráficos de rendimiento basados en las métricas del modelo.

    Args:
        metrics (dict): Diccionario que contiene las listas de métricas.
        output_image_path (str, optional): Ruta donde guardar el gráfico generado.
    """
    # 1. Configuración de estilo avanzada con Seaborn
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 14})

    # 2. Extracción y preparación de datos
    train_loss = metrics.get("train_loss", [])
    val_loss = metrics.get("val_loss", [])
    val_f1 = metrics.get("val_f1", [])
    val_precision = metrics.get("val_precision", [])
    val_recall = metrics.get("val_recall", [])
    model_type = metrics.get("model_type", "Modelo").upper()
    timestamp = metrics.get("timestamp", "")

    # Crear el eje X dinámicamente basado en la cantidad de épocas
    epochs = range(1, len(train_loss) + 1)

    # 3. Inicializar la figura con dos subgráficos (subplots) bien distribuidos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Monitoreo de Entrenamiento - {model_type} ({timestamp})",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # --- SUBGRÁFICO 1: Pérdidas (Loss) ---
    ax1.plot(
        epochs,
        train_loss,
        label="Train Loss",
        color="#1f77b4",
        linewidth=2.5,
        marker="o",
        markersize=4,
    )
    if val_loss:
        ax1.plot(
            epochs,
            val_loss,
            label="Val Loss",
            color="#d62728",
            linewidth=2.5,
            marker="s",
            markersize=4,
        )

        # Identificar visualmente el punto de inflexión del Overfitting (mínima pérdida de validación)
        min_val_loss_idx = val_loss.index(min(val_loss))
        ax1.axvline(
            x=min_val_loss_idx + 1,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"Mín. Val Loss (Época {min_val_loss_idx+1})",
        )

    ax1.set_title("Curvas de Pérdida (Loss Summary)", fontweight="semibold")
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Pérdida (Loss)")
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_xlim(1, len(train_loss))

    # --- SUBGRÁFICO 2: Métricas de Validación (F1, Precision, Recall) ---
    if val_f1:
        ax2.plot(epochs, val_f1, label="Val F1-Score", color="#2ca02c", linewidth=2)
    if val_precision:
        ax2.plot(
            epochs,
            val_precision,
            label="Val Precision",
            color="#ff7f0e",
            linewidth=1.5,
            linestyle="--",
        )
    if val_recall:
        ax2.plot(
            epochs,
            val_recall,
            label="Val Recall",
            color="#9467bd",
            linewidth=1.5,
            linestyle="-.",
        )

    ax2.set_title("Métricas de Clasificación en Validación", fontweight="semibold")
    ax2.set_xlabel("Épocas")
    ax2.set_ylabel("Score (0.0 - 1.0)")
    ax2.set_ylim(0.5, 1.05)  # Ajustado al rango de tus datos de validación
    ax2.legend(loc="lower right", frameon=True)
    ax2.set_xlim(1, len(train_loss))

    # Ajustar espaciado entre componentes
    plt.tight_layout()

    # 4. Usabilidad: Guardar o Mostrar el gráfico
    if output_image_path:
        plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Gráfico guardado exitosamente en: {output_image_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # 1. Definir la ruta base del proyecto y la subcarpeta de salida
    BASE_DIR = r"C:\Users\HP\Documents\Tesis\Programas\Ner\TesisNer\Version 2.0"
    OUTPUT_SUBDIR = os.path.join(BASE_DIR, "output")

    # 2. Configurar los nombres de los archivos de entrada y salida
    FILE_NAME = "metrics_20260628_163720.json"
    IMAGE_NAME = "reporte_performance_distilbert.png"

    # 3. Construir las rutas absolutas completas
    JSON_FILE = os.path.join(OUTPUT_SUBDIR, FILE_NAME)
    OUTPUT_IMAGE = os.path.join(OUTPUT_SUBDIR, IMAGE_NAME)

    try:
        print(f"[PROCESO] Buscando datos en: {JSON_FILE}")
        raw_metrics = load_metrics(JSON_FILE)

        print(f"[PROCESO] Generando gráficos...")
        # Guardará la imagen del reporte directamente en la misma carpeta 'output'
        plot_training_performance(raw_metrics, output_image_path=OUTPUT_IMAGE)

    except FileNotFoundError as fnf_error:
        print(f"[ERROR DE RUTA] {fnf_error}")
        print(
            "[CONSEJO] Verifica que el archivo JSON esté realmente dentro de la carpeta 'output'."
        )
    except Exception as e:
        print(f"[ERROR] Ocurrió un fallo inesperado en la ejecución: {e}")