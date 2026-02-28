"""
Script para separar los primeros 973 registros del archivo distilbert_augmented_2.json
"""

import json
import os

# Rutas
input_file = 'output/distilbert_augmented_2.json'
output_dir = 'output'
first_973_file = os.path.join(output_dir, 'distilbert_augmented_2_first_973.json')
remaining_file = os.path.join(output_dir, 'distilbert_augmented_2_remaining.json')

# Cargar el archivo
print(f"Cargando {input_file}...")
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = data['texts']
tags = data['tags']

# Verificar que tienen la misma longitud
if len(texts) != len(tags):
    print(f"ADVERTENCIA: texts ({len(texts)}) y tags ({len(tags)}) tienen diferente longitud")

total_records = len(texts)
print(f"Total de registros: {total_records}")

# Separar en dos grupos
first_973_texts = texts[:973]
first_973_tags = tags[:973]

remaining_texts = texts[973:]
remaining_tags = tags[973:]

# Guardar los primeros 973 registros
print(f"\nGuardando primeros 973 registros en {first_973_file}...")
with open(first_973_file, 'w', encoding='utf-8') as f:
    json.dump({
        'texts': first_973_texts,
        'tags': first_973_tags
    }, f, ensure_ascii=False, indent=2)
print(f"✓ Guardado: {len(first_973_texts)} textos y {len(first_973_tags)} etiquetas")

# Guardar los registros restantes
print(f"\nGuardando registros restantes ({total_records - 973}) en {remaining_file}...")
with open(remaining_file, 'w', encoding='utf-8') as f:
    json.dump({
        'texts': remaining_texts,
        'tags': remaining_tags
    }, f, ensure_ascii=False, indent=2)
print(f"✓ Guardado: {len(remaining_texts)} textos y {len(remaining_tags)} etiquetas")

print(f"\n✅ Proceso completado:")
print(f"   - Primeros 973 registros: {first_973_file}")
print(f"   - Registros restantes ({total_records - 973}): {remaining_file}")
