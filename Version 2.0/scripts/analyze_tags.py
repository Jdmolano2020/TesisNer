import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import re
import os


def tokenize_text(text):
    """
    Tokenización simple coherente con BIO.
    Divide por espacios y separa puntuación.
    """
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens


def validate_and_fix_tags(text, tags):
    """
    Valida y repara las tags BIO para que coincidan con los tokens generados del texto.
    - Si coinciden → OK
    - Si hay menos tags → se rellenan con "O"
    - Si hay más tags → se recorta
    - Si había desalineación → se regenera
    """
    tokens = tokenize_text(text)

    if len(tokens) == len(tags):
        return tokens, tags

    # Caso 1: tags más cortas → completar con "O"
    if len(tags) < len(tokens):
        fixed_tags = tags + ["O"] * (len(tokens) - len(tags))
        return tokens, fixed_tags

    # Caso 2: tags más largas → recortar
    if len(tags) > len(tokens):
        fixed_tags = tags[:len(tokens)]
        return tokens, fixed_tags

    # fallback
    return tokens, tags

def extraer_entidades(tokens, tags):
    entidades = []
    entidad_actual = []
    etiqueta_actual = None

    for tok, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            # cerrar entidad anterior si existía
            if entidad_actual:
                entidades.append((entidad_actual, etiqueta_actual))

            entidad_actual = [tok]
            etiqueta_actual = tag[2:]  # quitar "B-"

        elif tag.startswith("I-") and entidad_actual:
            entidad_actual.append(tok)

        else:
            # cerrar entidad si se acaba
            if entidad_actual:
                entidades.append((entidad_actual, etiqueta_actual))
            entidad_actual = []
            etiqueta_actual = None

    # cerrar última
    if entidad_actual:
        entidades.append((entidad_actual, etiqueta_actual))

    return entidades

# ------------------------------
# 1. Cargar JSON
# ------------------------------
json_file = os.path.join('output', 'distilbert_augmented_1.json')
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = data["texts"]
tags = data["tags"]

print(f"Total de oraciones: {len(texts)}")


# ------------------------------
# 2. Validar alineación tokens vs etiquetas
# ------------------------------
errores = []
for i, (toks, tg) in enumerate(zip(texts, tags)):
    if len(toks) != len(tg):
        errores.append((i, len(toks), len(tg)))

print("\n=== Validación de alineación ===")
if errores:
    print(f"ADVERTENCIA: Se encontraron {len(errores)} inconsistencias:")
    for e in errores[:5]:
        print(f" - Oración {e[0]}: {e[1]} tokens vs {e[2]} tags")
else:
    print("OK: No hay problemas de alineación tokens–tags")


fixed_texts = []
fixed_tags_list = []

for text, tags in zip(texts, tags):
    tks, tgs = validate_and_fix_tags(text, tags)
    fixed_texts.append(tks)
    fixed_tags_list.append(tgs)

texts = fixed_texts
tags = fixed_tags_list

errores = []
for i, (toks, tg) in enumerate(zip(texts, tags)):
    if len(toks) != len(tg):
        errores.append((i, len(toks), len(tg)))

print("\n=== Validación de alineación despues de corregir ===")
if errores:
    print(f"ADVERTENCIA: Se encontraron {len(errores)} inconsistencias:")
    for e in errores[:5]:
        print(f" - Oración {e[0]}: {e[1]} tokens vs {e[2]} tags")
else:
    print("OK: No hay problemas de alineación tokens–tags")

for i in range(5):
    print(f"\n=== Registro {i} ===")
    print("Tipo texts:", type(texts[i]))
    print("Tipo tags :", type(tags[i]))
    print("Ejemplo texts:", texts[i][:20] if isinstance(texts[i], list) else texts[i][:200])
    print("Ejemplo tags:", tags[i][:20])
    print("Len tokens:", len(texts[i]) if isinstance(texts[i], list) else "es string")
    print("Len tags  :", len(tags[i]))

# ------------------------------
# 3. Distribución de etiquetas BIO
# ------------------------------
all_tags = [tag for seq in tags for tag in seq]
tag_counts = Counter(all_tags)

print("\n=== Distribución de etiquetas BIO ===")
for tag, cnt in tag_counts.items():
    print(f"{tag:5s} : {cnt}")


# ------------------------------
# 4. Conteo de entidades por tipo
# ------------------------------

contador_entidades = Counter()
ejemplos_entidades = defaultdict(list)

for tks, tgs in zip(texts, tags):
    ents = extraer_entidades(tks, tgs)
    for ent_tokens, tipo in ents:
        contador_entidades[tipo] += 1
        if len(ejemplos_entidades[tipo]) < 50:  # guardar solo 50 ejemplos
            ejemplos_entidades[tipo].append(" ".join(ent_tokens))

print("\n=== Entidades detectadas ===")
for tipo, cant in contador_entidades.items():
    print(f"{tipo:10s} : {cant}")

print("\n=== Ejemplos por tipo ===")
for tipo, ej in ejemplos_entidades.items():
    print(f"\n{tipo}:")
    for e in ej:
        print("  -", e)

# ------------------------------
# 4b. Análisis de oraciones incompletas
missing_types = Counter()
missing_examples = []
expected_types = {"company", "address", "date", "total"}

for idx, (tks, tgs) in enumerate(zip(texts, tags)):
    ents = extraer_entidades(tks, tgs)
    found = {tipo for _, tipo in ents}
    missing = expected_types - found
    if missing:
        for m in missing:
            missing_types[m] += 1
        if len(missing_examples) < 50:
            missing_examples.append({
                "index": idx,
                "missing": sorted(missing),
                "tokens": " ".join(tks),
                "entities": [" ".join(ent_tokens) + ":" + tipo for ent_tokens, tipo in ents]
            })

print("\n=== Oraciones incompletas ===")
print(f"Oraciones con entidades faltantes: {len(missing_examples)}")
for tipo, cnt in missing_types.items():
    print(f"{tipo:10s} : {cnt}")

print("\n=== Ejemplos de oraciones con entidades faltantes ===")
for ex in missing_examples:
    print(f"- idx={ex['index']} missing={ex['missing']} entities={ex['entities']}")


# ------------------------------
# 5. Estadísticas de longitudes
# ------------------------------
longitudes = [len(t) for t in texts]

print("\n=== Longitud de textos ===")
print(f"Promedio tokens: {sum(longitudes)/len(longitudes):.2f}")
print(f"Mínimo tokens  : {min(longitudes)}")
print(f"Máximo tokens  : {max(longitudes)}")


# ------------------------------
# 6. Gráfico de distribución BIO (opcional)
# ------------------------------
plt.figure(figsize=(8,4))
plt.bar(tag_counts.keys(), tag_counts.values())
plt.title("Distribución de etiquetas BIO")
plt.ylabel("Frecuencia")
plt.xlabel("Etiqueta")
plt.show()