#!/usr/bin/env python3
"""
Limpiar solapamientos de entidades
"""
import json
import os
import sys
from typing import List, Tuple

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def load_data():
    """Cargar datos desde spacy_loaded.json"""
    json_file = os.path.join('output', 'spacy_loaded.json')
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_item(item):
    """Parse an item from the loaded data"""
    if isinstance(item, list) and len(item) >= 2:
        text = item[0]
        ents_dict = item[1] if isinstance(item[1], dict) else {}
        entities = ents_dict.get('entities', [])
    elif isinstance(item, dict):
        text = item.get('text', '')
        entities = item.get('entities', [])
    else:
        return None, None
    return text, entities

def remove_overlaps(entities_list: List[List]) -> List[List]:
    """
    Remove overlapping entities, keeping the first one found
    """
    if not entities_list:
        return []
    
    # Normalize to tuples with integers
    normalized = []
    for ent in entities_list:
        if isinstance(ent, (list, tuple)) and len(ent) >= 3:
            normalized.append((int(ent[0]), int(ent[1]), ent[2]))
        elif isinstance(ent, dict):
            s = ent.get('start')
            e = ent.get('end')
            l = ent.get('label')
            if s is not None and e is not None and l is not None:
                normalized.append((int(s), int(e), l))
    
    if not normalized:
        return []
    
    # Sort by start position, then by end position (longer spans last)
    sorted_ents = sorted(normalized, key=lambda x: (x[0], -(x[1]-x[0])))
    
    # Keep only non-overlapping entities
    kept = []
    occupied = set()  # Set of character positions already covered
    
    for start, end, label in sorted_ents:
        # Check if any position is already occupied
        positions = set(range(start, end))
        if not positions.intersection(occupied):
            kept.append([start, end, label])
            occupied.update(positions)
    
    return kept

def clean_overlaps():
    """Remove overlapping entities from the dataset"""
    print("[*] Cargando datos...\n")
    data = load_data()
    
    entities_removed = 0
    docs_modified = 0
    original_entities = 0
    
    cleaned_data = []
    
    print(f"[*] Procesando {len(data)} documentos...\n")
    
    for doc_idx, item in enumerate(data):
        text, entities = parse_item(item)
        
        if text is None or entities is None:
            cleaned_data.append(item)
            continue
        
        original_entities += len(entities)
        cleaned_entities = remove_overlaps(entities)
        
        if len(cleaned_entities) < len(entities):
            entities_removed += len(entities) - len(cleaned_entities)
            docs_modified += 1
        
        # Reconstruct item in the same format
        if isinstance(item, list) and len(item) >= 2:
            cleaned_item = [text, {"entities": cleaned_entities}]
        elif isinstance(item, dict):
            cleaned_item = {"text": text, "entities": cleaned_entities}
        else:
            cleaned_item = item
        
        cleaned_data.append(cleaned_item)
    
    # Save cleaned data
    output_file = os.path.join('output', 'spacy_loaded_cleaned.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print("[RESULTADOS]")
    print(f"Documentos procesados: {len(data)}")
    print(f"Documentos modificados: {docs_modified}")
    print(f"Entidades originales: {original_entities}")
    print(f"Entidades removidas: {entities_removed} ({100*entities_removed/original_entities:.1f}%)")
    print(f"Entidades finales: {original_entities - entities_removed}")
    print(f"\nArchivo guardado: {output_file}")

if __name__ == '__main__':
    clean_overlaps()
