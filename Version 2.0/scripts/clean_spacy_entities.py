"""
Corrector de entidades en spacy_augmented_2.json
Limpia: índices negativos, duplicados, superpuestos y espacios en blanco
"""
import json
import os
from collections import defaultdict

def clean_spacy_json(input_file, output_file, remove_duplicates=True, remove_overlapping=True):
    """
    Limpia el archivo JSON de spaCy removiendo:
    - Entidades con índices inválidos (negativos, fuera de rango)
    - Entidades solo con espacios en blanco
    - Entidades duplicadas (opcional)
    - Entidades superpuestas (opcional)
    
    Args:
        input_file: Ruta del archivo JSON a limpiar
        output_file: Ruta del archivo limpiado
        remove_duplicates: Si True, remover duplicados exactos
        remove_overlapping: Si True, remover entidades superpuestas
    """
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cleaned_data = []
    stats = {
        'original_docs': len(data),
        'cleaned_docs': 0,
        'total_original_entities': 0,
        'total_cleaned_entities': 0,
        'removed_negative_indices': 0,
        'removed_out_of_bounds': 0,
        'removed_whitespace': 0,
        'removed_duplicates': 0,
        'removed_overlapping': 0,
        'docs_with_issues': 0
    }
    
    for doc_idx, item in enumerate(data):
        text = item.get('text', '')
        entities_raw = item.get('entities', [])
        
        if not entities_raw:
            cleaned_data.append({'text': text, 'entities': []})
            continue
        
        stats['total_original_entities'] += len(entities_raw)
        
        # Convertir a tuplas normalizadas
        normalized_ents = []
        for ent in entities_raw:
            if isinstance(ent, (list, tuple)) and len(ent) >= 3:
                normalized_ents.append((ent[0], ent[1], ent[2]))
            elif isinstance(ent, dict):
                s = ent.get('start')
                e = ent.get('end')
                l = ent.get('label')
                if s is not None and e is not None and l is not None:
                    normalized_ents.append((s, e, l))
        
        # Fase 1: Filtrar entidades inválidas
        valid_ents = []
        doc_has_issues = False
        
        for start, end, label in normalized_ents:
            # Validar índices
            if start < 0 or end < 0:
                stats['removed_negative_indices'] += 1
                doc_has_issues = True
                continue
            
            if start >= end:
                stats['removed_negative_indices'] += 1
                doc_has_issues = True
                continue
            
            if end > len(text):
                stats['removed_out_of_bounds'] += 1
                doc_has_issues = True
                continue
            
            # Validar contenido
            try:
                span_text = text[start:end]
            except Exception:
                stats['removed_out_of_bounds'] += 1
                doc_has_issues = True
                continue
            
            if not span_text or span_text.isspace():
                stats['removed_whitespace'] += 1
                doc_has_issues = True
                continue
            
            valid_ents.append((start, end, label))
        
        # Fase 2: Remover duplicados exactos (opcional)
        if remove_duplicates:
            unique_ents = {}
            for ent in valid_ents:
                if ent not in unique_ents:
                    unique_ents[ent] = ent
                else:
                    stats['removed_duplicates'] += 1
                    doc_has_issues = True
            valid_ents = list(unique_ents.values())
        
        # Fase 3: Remover entidades superpuestas (opcional)
        # Mantener solo una de cada grupo de entidades superpuestas
        if remove_overlapping:
            # Ordenar por start, luego por length (preferir más largas)
            sorted_ents = sorted(valid_ents, key=lambda x: (x[0], -(x[1] - x[0])))
            
            non_overlapping = []
            for start, end, label in sorted_ents:
                # Verificar si se superpone con alguna ya añadida
                overlaps = False
                for existing_start, existing_end, _ in non_overlapping:
                    # Verificar solapamiento
                    if not (end <= existing_start or start >= existing_end):
                        overlaps = True
                        break
                
                if not overlaps:
                    non_overlapping.append((start, end, label))
                else:
                    stats['removed_overlapping'] += 1
                    doc_has_issues = True
            
            valid_ents = non_overlapping
        
        # Ordenar entidades válidas por posición
        valid_ents.sort(key=lambda x: (x[0], x[1]))
        
        # Construir item limpio
        cleaned_item = {'text': text, 'entities': valid_ents}
        cleaned_data.append(cleaned_item)
        
        if valid_ents:
            stats['cleaned_docs'] += 1
            stats['total_cleaned_entities'] += len(valid_ents)
        
        if doc_has_issues:
            stats['docs_with_issues'] += 1
    
    # Guardar archivos limpios
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    # Imprimir estadísticas
    print("=" * 70)
    print("📊 ESTADÍSTICAS DE LIMPIEZA")
    print("=" * 70)
    print(f"\nDocumentos:")
    print(f"  Total original: {stats['original_docs']}")
    print(f"  Total limpiados: {stats['cleaned_docs']}")
    print(f"  Con problemas corregidos: {stats['docs_with_issues']}")
    
    print(f"\nEntidades:")
    print(f"  Total original: {stats['total_original_entities']}")
    print(f"  Total limpiadas: {stats['total_cleaned_entities']}")
    print(f"  Removidas - Índices negativos: {stats['removed_negative_indices']}")
    print(f"  Removidas - Fuera de rango: {stats['removed_out_of_bounds']}")
    print(f"  Removidas - Solo espacios: {stats['removed_whitespace']}")
    print(f"  Removidas - Duplicadas: {stats['removed_duplicates']}")
    print(f"  Removidas - Superpuestas: {stats['removed_overlapping']}")
    
    total_removed = (stats['removed_negative_indices'] + 
                     stats['removed_out_of_bounds'] + 
                     stats['removed_whitespace'] + 
                     stats['removed_duplicates'] + 
                     stats['removed_overlapping'])
    
    print(f"\n  Total removidas: {total_removed} ({total_removed / stats['total_original_entities'] * 100:.1f}%)")
    print(f"\nArchivo limpiado guardado en: {output_file}")
    print("=" * 70)
    
    return cleaned_data, stats

if __name__ == "__main__":
    input_file = os.path.join('output', 'spacy_augmented_2.json')
    output_file = os.path.join('output', 'spacy_augmented_2_cleaned.json')
    
    if not os.path.exists(input_file):
        print(f"❌ Archivo no encontrado: {input_file}")
    else:
        print("🔧 Limpiando archivo de entidades...\n")
        cleaned_data, stats = clean_spacy_json(
            input_file, 
            output_file,
            remove_duplicates=True,
            remove_overlapping=True
        )
