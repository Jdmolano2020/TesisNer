import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS DE MÓDULOS DE PYTHON
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from distilbert_sroie_augmentation import SROIEDistilBERTAugmenter
from spacy_sroie_augmentation import SROIESpacyAugmenter

# Configuración de carpetas relativas a "Version 2.0"
OUTPUT_DIR = os.path.join(PARENT_DIR, "output")
DATA_DIR = os.path.join(PARENT_DIR, "data")  # Necesario para inicializar las clases

SPACY_MODEL_PATH = os.path.join(OUTPUT_DIR, "spacy_model", "best_model_final")
DISTILBERT_MODEL_PATH = os.path.join(OUTPUT_DIR, "distilbert_model")

print("[INICIO] Inicializando script de demostración de complementación...")


# ==========================================
# 2. CARGA DE MODELOS (Inyección Directa y Segura)
# ==========================================
try:
    print("[PROCESO] Instanciando clases base de aumentación...")
    
    # Se eliminó el parámetro data_dir=DATA_DIR que causaba el TypeError
    final_spacy_model = SROIESpacyAugmenter()
    final_distilbert_model = SROIEDistilBERTAugmenter()

    # ---------------------------------------------------------
    # A. INYECCIÓN DE SPACY
    # ---------------------------------------------------------
    print(f"[PROCESO] Cargando motor nativo spaCy desde: {SPACY_MODEL_PATH}")
    import spacy
    
    motor_spacy = spacy.load(SPACY_MODEL_PATH)
    final_spacy_model.nlp = motor_spacy  # Inyección en la variable interna
    print("[INFO] Modelo spaCy inyectado correctamente.")

   # ---------------------------------------------------------
    # B. INYECCIÓN DE DISTILBERT
    # ---------------------------------------------------------
    print(f"[PROCESO] Cargando pesos de DistilBERT desde: {DISTILBERT_MODEL_PATH}")
    from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
    import torch
    
    # 1. Carga inteligente del Tokenizador (Manejo del error de vocabulario faltante)
    try:
        tokenizador_entrenado = DistilBertTokenizerFast.from_pretrained(DISTILBERT_MODEL_PATH)
    except TypeError:
        print("[AVISO] No se halló el 'vocab.txt' en la carpeta local.")
        print("[PROCESO] Descargando tokenizador base ('distilbert-base-uncased') desde HuggingFace...")
        # Nota: Si tu modelo base original fue otro (ej. 'distilbert-base-multilingual-cased' o 'distilbert-base-cased'), 
        # asegúrate de cambiar el string en la siguiente línea por el correcto.
        tokenizador_entrenado = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    # 2. Carga del Modelo (Pesos finamente ajustados en tu Tesis)
    modelo_entrenado = DistilBertForTokenClassification.from_pretrained(DISTILBERT_MODEL_PATH)
    
    # 3. Inyectamos en las variables internas que usa tu clase en predict()
    final_distilbert_model.tokenizer = tokenizador_entrenado
    final_distilbert_model.model = modelo_entrenado
    
    # 4. Enviamos a GPU si está disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_distilbert_model.model.to(device)
    final_distilbert_model.device = device
    
    print("[INFO] Modelo DistilBERT inyectado y enviado a dispositivo correctamente.")
    print("\n[ÉXITO] Ambos modelos fueron cargados en memoria listos para predecir.")
except Exception as e:
    print(f"\n[ERROR CRÍTICO] Falló la inicialización o carga de los modelos: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==========================================
# 3. CARGA DE DATOS DE MUESTRA
# ==========================================
sample_source = None

# Apuntamos directamente al archivo referenciado en tu traza de error anterior
data_checkpoint = os.path.join(OUTPUT_DIR, "spacy_augmented_6_samp100.json")

if os.path.exists(data_checkpoint):
    print(f"\n[PROCESO] Cargando datos de muestra desde: {data_checkpoint}")
    with open(data_checkpoint, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        if isinstance(raw_data, list):
            sample_source = raw_data
else:
    print(f"\n[ALERTA] No se encontró el archivo de evaluación en {data_checkpoint}.")
    sample_source = []

if not sample_source:
    print("[FIN] No hay datos de muestra disponibles para realizar la demostración.")
    sys.exit(0)

sample_size = min(100, len(sample_source))  # Procesaremos hasta 100 textos
sample = sample_source[:sample_size]

texts = []
gold_list = []
for item in sample:
    if isinstance(item, list) or isinstance(item, tuple):
        txt, ann = item[0], item[1]
    else:
        txt, ann = item.get("text", ""), item.get("annotation", {"entities": []})

    texts.append(txt)
    gold_list.append(set([(s, e, lab) for s, e, lab in ann["entities"]]))

print(f"[INFO] Procesando {sample_size} textos de prueba para generar las matrices...")


# ==========================================
# 4. PREDICCIONES DE LOS MODELOS
# ==========================================
print("[PROCESO] Ejecutando inferencia con spaCy...")
spacy_preds_raw = final_spacy_model.predict(texts)
spacy_preds = [
    set([(st, ed, lab) for _, st, ed, lab in doc_ents])
    for doc_ents in spacy_preds_raw
]

print("[PROCESO] Ejecutando inferencia con DistilBERT...")
distil_tags = final_distilbert_model.predict(texts)
distil_preds = []
for txt, tags in zip(texts, distil_tags):
    ents = final_distilbert_model.convert_tags_to_entities(txt, tags)
    ents_set = set([(s, e, lab) for _, s, e, lab in ents])
    distil_preds.append(ents_set)


# ==========================================
# 5. CÁLCULO DE MÉTRICAS
# ==========================================
def compute_metrics(gold_list, pred_list):
    tp = fp = fn = 0
    for gold, pred in zip(gold_list, pred_list):
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    tn = 1000  # Estimación de True Negatives (fondo O) para los gráficos ROC
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tpr": tpr,
        "fpr": fpr,
    }

print("[PROCESO] Calculando métricas de evaluación...")
spacy_metrics = compute_metrics(gold_list, spacy_preds)
distil_metrics = compute_metrics(gold_list, distil_preds)

union_preds = [s | d for s, d in zip(spacy_preds, distil_preds)]
union_metrics = compute_metrics(gold_list, union_preds)


# ==========================================
# 6. GENERACIÓN DE GRÁFICOS (Matriz y ROC)
# ==========================================
sns.set_theme(style="whitegrid")

print("\n[VISUALIZACIÓN] Exportando Matrices de Confusión...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    "Matrices de Confusión de Entidades (NER SROIE)",
    fontsize=16,
    fontweight="bold",
    y=1.05,
)

modelos_metrics = [
    ("spaCy", spacy_metrics),
    ("DistilBERT", distil_metrics),
    ("Unión Combinada", union_metrics),
]

for idx, (nombre, m) in enumerate(modelos_metrics):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    labels = np.array(
        [
            [f"Negativos (O)\n{m['tn']}", f"Falsos Positivos\n{m['fp']}"],
            [f"Falsos Negativos\n{m['fn']}", f"Verdaderos Positivos\n{m['tp']}"],
        ]
    )
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        ax=axes[idx],
        annot_kws={"size": 11, "fontweight": "semibold"},
        linewidths=1,
        linecolor="white",
    )
    axes[idx].set_title(f"Modelo: {nombre}", fontsize=13, fontweight="semibold")
    axes[idx].set_xlabel("Predicción")
    axes[idx].set_ylabel("Valor Real")
    axes[idx].xaxis.set_ticklabels(["No Entidad", "Entidad"])
    axes[idx].yaxis.set_ticklabels(["No Entidad", "Entidad"])

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrices_complementation.png")
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close()

print("[VISUALIZACIÓN] Exportando Curva ROC...")
plt.figure(figsize=(8, 6))

for nombre, m in modelos_metrics:
    plt.plot(
        [0, m["fpr"], 1],
        [0, m["tpr"], 1],
        marker="o",
        label=f"{nombre} (TPR: {m['tpr']:.2f}, FPR: {m['fpr']:.2f})",
    )

plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Línea Aleatoria")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR / Recall)")
plt.title(
    "Comparativa del Espacio Curva ROC (Modelos NER)",
    fontsize=14,
    fontweight="bold",
)
plt.legend(loc="lower right", frameon=True)

roc_path = os.path.join(OUTPUT_DIR, "roc_curve_complementation.png")
plt.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.close()

# ==========================================
# 7. EXPORTACIÓN DEL REPORTE JSON FINAL
# ==========================================
report = {
    "sample_size": sample_size,
    "spacy": spacy_metrics,
    "distilbert": distil_metrics,
    "union": union_metrics,
}

report_path = os.path.join(OUTPUT_DIR, "complementation_report.json")
with open(report_path, "w", encoding="utf-8") as rf:
    json.dump(report, rf, indent=2, ensure_ascii=False)

print(f"\n[ÉXITO] Ejecución completada. Los gráficos se guardaron en: {OUTPUT_DIR}")