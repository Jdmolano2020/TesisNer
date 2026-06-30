"""
Análisis detallado de problemas en el armado de entities
"""
import json
import os
import sys
from collections import defaultdict

# Configurar UTF-8 para la salida en Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def detailed_entity_analysis():
    """Análisis profundo de los problemas de entities"""
    json_file = os.path.join('output', 'spacy_augmented_6_samp100.json')
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    #spacy_augmented_6_samp100
    issues = {
        'overlapping': [],          # Entities que se solapan
        'out_of_bounds': [],        # Entities fuera de los límites del texto
        'negative_range': [],       # start >= end
        'empty_entities': [],       # Entities que apuntan a strings vacíos
        'whitespace_only': [],      # Entities que son solo espacios
        'duplicates': [],           # Entities duplicadas
        'invalid_labels': set(),    # Labels inválidos
        'span_mismatches': []       # La alineación no coincide con lo esperado
    }
    
    valid_labels = {'company', 'address', 'date', 'total'}
    
    print("[*] Análisis detallado de problemas...\n")
    stats = {
        'total_examples': len(data),
        'entity_types': defaultdict(int),
        'examples_with_entities': 0,
        'examples_without_entities': 0,
        'format_issues': []
    }
    for doc_idx, item in enumerate(data):
        # item es [texto, {"entities": [...]}] en spacy_loaded.json
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            text = item[0]
            entities = item[1].get('entities', []) if isinstance(item[1], dict) else []
        elif isinstance(item, dict):
            # Formato alternativo: {"text": "...", "entities": [...]}
            text = item.get('text', '')
            entities = item.get('entities', [])
        else:
            continue
        
        if not entities:
            continue
        print(f"Doc {doc_idx}: '{text[:50]}...' con {len(entities)} entities")
        if entities:
            stats['examples_with_entities'] += 1
            for ent in entities:
                if isinstance(ent, dict) and 'label' in ent:
                    stats['entity_types'][ent['label']] += 1
                elif isinstance(ent, (list, tuple)) and len(ent) >= 3:
                    stats['entity_types'][ent[2]] += 1
        else:
            stats['examples_without_entities'] += 1
        # Convertir a formato normalizado
        normalized_ents = []
        for ent in entities:
            if isinstance(ent, (list, tuple)) and len(ent) >= 3:
                normalized_ents.append((ent[0], ent[1], ent[2]))
            elif isinstance(ent, dict):
                normalized_ents.append((ent.get('start'), ent.get('end'), ent.get('label')))
        
        # Revisar cada entity
        for ent_idx, (start, end, label) in enumerate(normalized_ents):
            # 1. Check label validity
            if label not in valid_labels:
                issues['invalid_labels'].add(label)
                issues['span_mismatches'].append({
                    'doc': doc_idx,
                    'ent': ent_idx,
                    'issue': f'Label inválido: {label}',
                    'range': f'[{start}:{end}]'
                })
            
            # 2. Check negative/invalid range
            if start is None or end is None or not isinstance(start, int) or not isinstance(end, int):
                issues['span_mismatches'].append({
                    'doc': doc_idx,
                    'ent': ent_idx,
                    'issue': f'start/end no son int: start={start}, end={end}',
                    'range': f'[{start}:{end}]'
                })
                continue
            
            if start >= end:
                issues['negative_range'].append({
                    'doc': doc_idx,
                    'ent': ent_idx,
                    'range': f'[{start}:{end}]',
                    'label': label
                })
                continue
            
            # 3. Check out of bounds
            if start < 0 or end > len(text):
                issues['out_of_bounds'].append({
                    'doc': doc_idx,
                    'ent': ent_idx,
                    'range': f'[{start}:{end}]',
                    'text_len': len(text),
                    'label': label
                })
                continue
            
            # 4. Extract span and validate
            try:
                span_text = text[start:end]
            except Exception as e:
                issues['span_mismatches'].append({
                    'doc': doc_idx,
                    'ent': ent_idx,
                    'issue': f'Error extrayendo span: {e}',
                    'range': f'[{start}:{end}]'
                })
                continue
            
            # 5. Check empty entities
            if not span_text:
                issues['empty_entities'].append({
                    'doc': doc_idx,
                    'ent': ent_idx,
                    'range': f'[{start}:{end}]',
                    'label': label
                })
            
            # 6. Check whitespace only
            elif span_text.strip() == '':
                issues['whitespace_only'].append({
                    'doc': doc_idx,
                    'ent': ent_idx,
                    'range': f'[{start}:{end}]',
                    'label': label,
                    'content': repr(span_text)
                })
        
        # 7. Check overlapping entities in this document
        sorted_ents = sorted([(start, end, label) for start, end, label in normalized_ents if start is not None and end is not None])
        for i in range(len(sorted_ents) - 1):
            s1, e1, l1 = sorted_ents[i]
            s2, e2, l2 = sorted_ents[i + 1]
            if s2 < e1:  # Overlapping
                issues['overlapping'].append({
                    'doc': doc_idx,
                    'ent1': f'[{s1}:{e1}]({l1})',
                    'ent2': f'[{s2}:{e2}]({l2})',
                    'overlap': f'[{s2}:{e1}]'
                })
        
        # 8. Check for duplicates in this document
        ent_tuples = [tuple(e) for e in normalized_ents]
        seen = {}
        for ent in ent_tuples:
            if ent in seen:
                issues['duplicates'].append({
                    'doc': doc_idx,
                    'entity': f'{ent}',
                    'count': seen[ent] + 1
                })
                seen[ent] += 1
            else:
                seen[ent] = 1
    
    # Print results
    print("=" * 70)
    print("[PROBLEMAS ENCONTRADOS]")
    print("=" * 70)
    print(f"\n📊 Estadísticas :")
    print(f"   Total de ejemplos: {stats['total_examples']}")
    print(f"   Ejemplos con entities: {stats['examples_with_entities']}")
    print(f"   Ejemplos sin entities: {stats['examples_without_entities']}")
    print(f"   Tipos de entities encontrados:")
    for label, count in sorted(stats['entity_types'].items()):
        print(f"      - {label}: {count}")
    
    if issues['invalid_labels']:
        print(f"\n[ERROR] Labels inválidos encontrados: {issues['invalid_labels']}")
    
    if issues['negative_range']:
        print(f"\n[ERROR] Entities con start >= end: {len(issues['negative_range'])}")
        for item in issues['negative_range'][:5]:
            print(f"   Doc {item['doc']}, Entity {item['ent']}: {item['range']} ({item['label']})")
        if len(issues['negative_range']) > 5:
            print(f"   ... y {len(issues['negative_range']) - 5} más")
    
    if issues['out_of_bounds']:
        print(f"\n[ERROR] Entities fuera de límites: {len(issues['out_of_bounds'])}")
        for item in issues['out_of_bounds'][:5]:
            print(f"   Doc {item['doc']}, Entity {item['ent']}: {item['range']} (texto len={item['text_len']})")
        if len(issues['out_of_bounds']) > 5:
            print(f"   ... y {len(issues['out_of_bounds']) - 5} más")
    
    if issues['empty_entities']:
        print(f"\n[AVISO] Entities vacías: {len(issues['empty_entities'])}")
        for item in issues['empty_entities'][:5]:
            print(f"   Doc {item['doc']}, Entity {item['ent']}: {item['range']} ({item['label']})")
        if len(issues['empty_entities']) > 5:
            print(f"   ... y {len(issues['empty_entities']) - 5} más")
    
    if issues['whitespace_only']:
        print(f"\n[AVISO] Entities solo con espacios: {len(issues['whitespace_only'])}")
        for item in issues['whitespace_only'][:5]:
            print(f"   Doc {item['doc']}, Entity {item['ent']}: {item['range']} ({item['label']}) = {item['content']}")
        if len(issues['whitespace_only']) > 5:
            print(f"   ... y {len(issues['whitespace_only']) - 5} más")
    
    if issues['overlapping']:
        print(f"\n[AVISO] Entities superpuestas: {len(issues['overlapping'])}")
        for item in issues['overlapping'][:5]:
            print(f"   Doc {item['doc']}: {item['ent1']} sobrelapsa con {item['ent2']} en {item['overlap']}")
        if len(issues['overlapping']) > 5:
            print(f"   ... y {len(issues['overlapping']) - 5} más")
    
    if issues['duplicates']:
        dup_count = len(issues['duplicates'])
        print(f"\n[AVISO] Entities duplicadas: {dup_count} duplicados")
        # Group by entity
        dup_by_entity = {}
        for item in issues['duplicates']:
            dup_by_entity.setdefault(item['entity'], []).append(item['doc'])
        for ent, docs in sorted(dup_by_entity.items())[:5]:
            print(f"   {ent} aparece en {len(docs)} documentos")
        if len(dup_by_entity) > 5:
            print(f"   ... y {len(dup_by_entity) - 5} más")
    
    if issues['span_mismatches']:
        print(f"\n❌ Problemas de span: {len(issues['span_mismatches'])}")
        for item in issues['span_mismatches'][:5]:
            print(f"   Doc {item['doc']}, Entity {item['ent']}: {item.get('issue')} {item.get('range', '')}")
        if len(issues['span_mismatches']) > 5:
            print(f"   ... y {len(issues['span_mismatches']) - 5} más")
    
    # Summary
    total_issues = sum(len(v) for k, v in issues.items() if k != 'invalid_labels')
    print(f"\n{'='*70}")
    if total_issues == 0:
        print("✓ ¡No se encontraron problemas! Las entities están bien formateadas.")
    else:
        print(f"❌ Total de problemas: {total_issues}")
    print("=" * 70)

if __name__ == "__main__":
    detailed_entity_analysis()
