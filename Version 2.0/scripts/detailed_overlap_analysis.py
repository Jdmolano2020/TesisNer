#!/usr/bin/env python3
"""
Análisis detallado de solapamientos de entidades
"""
import json
import os
import sys
from collections import defaultdict

# Configurar UTF-8 para Windows
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

def ents_overlap(ent1, ent2):
    """Check if two entities overlap"""
    start1, end1, label1 = ent1
    start2, end2, label2 = ent2
    
    # They overlap if one starts before the other ends
    return (start1 < end2 and start2 < end1)

def analyze_overlaps_detailed():
    """Analizar solapamientos en detalle"""
    print("[*] Loading data for detailed overlap analysis...\n")
    data = load_data()
    
    overlaps_by_labels = defaultdict(int)  # Qué pares de labels se solapan
    docs_with_overlaps = 0
    total_overlap_pairs = 0
    examples = []  # Guardar ejemplos con contexto
    
    for doc_idx, item in enumerate(data):
        text, entities = parse_item(item)
        
        if text is None or entities is None or not entities:
            continue
        
        # Normalizar entidades
        normalized_ents = []
        for ent in entities:
            if isinstance(ent, (list, tuple)) and len(ent) >= 3:
                normalized_ents.append((int(ent[0]), int(ent[1]), ent[2]))
            elif isinstance(ent, dict):
                s = ent.get('start')
                e = ent.get('end')
                l = ent.get('label')
                if s is not None and e is not None and l is not None:
                    normalized_ents.append((int(s), int(e), l))
        
        if not normalized_ents:
            continue
        
        # Buscar solapamientos
        doc_has_overlaps = False
        for i in range(len(normalized_ents)):
            for j in range(i + 1, len(normalized_ents)):
                ent1 = normalized_ents[i]
                ent2 = normalized_ents[j]
                
                if ents_overlap(ent1, ent2):
                    start1, end1, label1 = ent1
                    start2, end2, label2 = ent2
                    
                    # Crear una clave normalizada para el par de labels
                    pair_key = tuple(sorted([label1, label2]))
                    overlaps_by_labels[pair_key] += 1
                    total_overlap_pairs += 1
                    doc_has_overlaps = True
                    
                    # Guardar ejemplos si no tenemos demasiados
                    if len(examples) < 20:
                        # Calcular la región de solapamiento
                        overlap_start = max(start1, start2)
                        overlap_end = min(end1, end2)
                        overlap_text = text[overlap_start:overlap_end] if overlap_start < overlap_end else ""
                        
                        # Contexto más amplio
                        ctx_start = max(0, min(start1, start2) - 20)
                        ctx_end = min(len(text), max(end1, end2) + 20)
                        context = text[ctx_start:ctx_end].replace('\n', ' ')
                        
                        examples.append({
                            'doc': doc_idx,
                            'pair': pair_key,
                            'ent1': f"{label1}: [{start1}:{end1}]",
                            'ent1_text': text[start1:end1] if start1 >= 0 and end1 <= len(text) else "?",
                            'ent2': f"{label2}: [{start2}:{end2}]",
                            'ent2_text': text[start2:end2] if start2 >= 0 and end2 <= len(text) else "?",
                            'overlap': overlap_text,
                            'context': context
                        })
        
        if doc_has_overlaps:
            docs_with_overlaps += 1
    
    # Imprimir reporte
    print("=" * 80)
    print("ANALISIS DE SOLAPAMIENTOS DE ENTIDADES")
    print("=" * 80)
    print()
    
    print("[ESTADISTICAS GENERALES]")
    print(f"Total de documentos con solapamientos: {docs_with_overlaps}")
    print(f"Total de pares de entidades solapadas: {total_overlap_pairs}")
    print()
    
    # Solapamientos por tipo de label
    print("[SOLAPAMIENTOS POR PARES DE LABELS]")
    print(f"{'Par de Labels':<30} {'Cantidad':>10} {'Porcentaje':>12}")
    print("-" * 52)
    
    sorted_pairs = sorted(overlaps_by_labels.items(), key=lambda x: x[1], reverse=True)
    for pair, count in sorted_pairs:
        percentage = 100 * count / total_overlap_pairs if total_overlap_pairs > 0 else 0
        label1, label2 = pair
        pair_str = f"{label1} <-> {label2}"
        print(f"{pair_str:<30} {count:>10} {percentage:>11.1f}%")
    
    print()
    print("[EJEMPLOS DE SOLAPAMIENTOS]")
    print()
    
    for i, example in enumerate(examples[:10], 1):
        print(f"Ejemplo {i}: Documento {example['doc']}")
        print(f"  Modelo par: {example['pair']}")
        print(f"  Entidad 1: {example['ent1']} -> {repr(example['ent1_text'])}")
        print(f"  Entidad 2: {example['ent2']} -> {repr(example['ent2_text'])}")
        print(f"  Region solapada: {repr(example['overlap'])}")
        print(f"  Contexto: ...{example['context']}...")
        print()
    
    # Análisis de distribución
    print("[DISTRIBUCION DE SOLAPAMIENTOS]")
    overlap_counts = defaultdict(int)  # Cuántos documentos tienen N solapamientos
    
    for doc_idx, item in enumerate(data):
        text, entities = parse_item(item)
        if text is None or entities is None or not entities:
            continue
        
        normalized_ents = []
        for ent in entities:
            if isinstance(ent, (list, tuple)) and len(ent) >= 3:
                normalized_ents.append((int(ent[0]), int(ent[1]), ent[2]))
            elif isinstance(ent, dict):
                s = ent.get('start')
                e = ent.get('end')
                l = ent.get('label')
                if s is not None and e is not None and l is not None:
                    normalized_ents.append((int(s), int(e), l))
        
        if not normalized_ents:
            continue
        
        doc_overlaps = 0
        for i in range(len(normalized_ents)):
            for j in range(i + 1, len(normalized_ents)):
                if ents_overlap(normalized_ents[i], normalized_ents[j]):
                    doc_overlaps += 1
        
        overlap_counts[doc_overlaps] += 1
    
    print(f"{'Pares solapados por doc':<20} {'# de docs':>15}")
    print("-" * 35)
    for count in sorted(overlap_counts.keys()):
        if count > 0:  # Solo mostrar docs con solapamientos
            print(f"{count:<20} {overlap_counts[count]:>15}")
    
    print()
    print("=" * 80)

if __name__ == '__main__':
    analyze_overlaps_detailed()
