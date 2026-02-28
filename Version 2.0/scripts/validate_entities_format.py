"""
Script para validar y diagnosticar la estructura de entities en spacy_augmented_2.json
"""
import json
import os
from collections import defaultdict

def analyze_json_structure():
    """Analiza la estructura del archivo JSON"""
    json_file = os.path.join('output', 'spacy_augmented_2.json')
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Total de ejemplos: {len(data)}")
    
    if len(data) == 0:
        print("Error: JSON está vacío")
        return
    
    first_item = data[0]
    print(f"\n📋 Estructura del primer ejemplo:")
    print(f"   Keys: {first_item.keys()}")
    
    # Analizar entities
    entities = first_item.get('entities', [])
    print(f"\n📍 Tipo de 'entities': {type(entities)}")
    print(f"   Número de entities: {len(entities)}")
    
    if entities:
        first_ent = entities[0]
        print(f"\n🔍 Primera entity (raw):")
        print(f"   Tipo: {type(first_ent)}")
        print(f"   Valor: {first_ent}")
        
        if isinstance(first_ent, dict):
            print(f"   Keys: {first_ent.keys()}")
            print(f"   ¿Tiene 'text'?: {'text' in first_ent}")
            print(f"   ¿Tiene 'start'?: {'start' in first_ent}")
            print(f"   ¿Tiene 'end'?: {'end' in first_ent}")
            print(f"   ¿Tiene 'label'?: {'label' in first_ent}")
        elif isinstance(first_ent, (list, tuple)):
            print(f"   Longitud: {len(first_ent)}")
            if len(first_ent) >= 3:
                print(f"   Elemento [0] (start): {first_ent[0]} (tipo: {type(first_ent[0]).__name__})")
                print(f"   Elemento [1] (end): {first_ent[1]} (tipo: {type(first_ent[1]).__name__})")
                print(f"   Elemento [2] (label): {first_ent[2]} (tipo: {type(first_ent[2]).__name__})")
    
    # Estadísticas generales
    stats = {
        'total_examples': len(data),
        'entity_types': defaultdict(int),
        'examples_with_entities': 0,
        'examples_without_entities': 0,
        'format_issues': []
    }
    
    for idx, item in enumerate(data[:100]):  # Analizar primeros 100
        text = item.get('text', '')
        entities = item.get('entities', [])
        
        if entities:
            stats['examples_with_entities'] += 1
            for ent in entities:
                if isinstance(ent, dict) and 'label' in ent:
                    stats['entity_types'][ent['label']] += 1
                elif isinstance(ent, (list, tuple)) and len(ent) >= 3:
                    stats['entity_types'][ent[2]] += 1
        else:
            stats['examples_without_entities'] += 1
    
    print(f"\n📊 Estadísticas (primeros 100 ejemplos):")
    print(f"   Ejemplos con entities: {stats['examples_with_entities']}")
    print(f"   Ejemplos sin entities: {stats['examples_without_entities']}")
    print(f"   Tipos de entities encontrados:")
    for label, count in sorted(stats['entity_types'].items()):
        print(f"      - {label}: {count}")
    
    # Validar alineaciones
    print(f"\n🔬 Validando alineaciones de entities (primeros 20 ejemplos):")
    alignment_errors = 0
    for idx, item in enumerate(data[:20]):
        text = item.get('text', '')
        entities = item.get('entities', [])
        
        for ent_idx, ent in enumerate(entities):
            try:
                if isinstance(ent, dict):
                    start, end, label = ent.get('start'), ent.get('end'), ent.get('label')
                    ent_text = ent.get('text')
                elif isinstance(ent, (list, tuple)) and len(ent) >= 3:
                    start, end, label = ent[0], ent[1], ent[2]
                    ent_text = None
                else:
                    print(f"   ⚠️  Ejemplo {idx}, entity {ent_idx}: Formato inválido")
                    alignment_errors += 1
                    continue
                
                if isinstance(start, int) and isinstance(end, int):
                    extracted = text[start:end]
                    if ent_text and extracted != ent_text:
                        print(f"   ⚠️  Ejemplo {idx}, entity {ent_idx}:")
                        print(f"       Rango [{start}:{end}] = '{extracted}'")
                        print(f"       Pero 'text' dice: '{ent_text}'")
                        alignment_errors += 1
                else:
                    print(f"   ⚠️  Ejemplo {idx}, entity {ent_idx}: start/end no son int")
                    alignment_errors += 1
                    
            except Exception as e:
                print(f"   ⚠️  Ejemplo {idx}, entity {ent_idx}: Error {e}")
                alignment_errors += 1
    
    if alignment_errors == 0:
        print("   ✓ Todas las alineaciones son correctas")
    else:
        print(f"   ❌ Encontrados {alignment_errors} errores de alineación")

if __name__ == "__main__":
    analyze_json_structure()
