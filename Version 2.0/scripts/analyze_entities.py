"""
Análisis detallado de problemas en entity data
"""
import json
import os
import sys

# Configurar UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def load_data():
    """Cargar datos desde spacy_loaded.json"""
    json_file = os.path.join('output', 'spacy_augmented_2.json')
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_item(item):
    """
    Parse an item from the loaded data.
    Handles two formats:
    1. [text, {"entities": [[start, end, label], ...]}]
    2. {"text": "...", "entities": [[start, end, label], ...]}
    """
    if isinstance(item, list) and len(item) >= 2:
        # Format 1: [text, {...}]
        text = item[0]
        ents_dict = item[1] if isinstance(item[1], dict) else {}
        entities = ents_dict.get('entities', [])
    elif isinstance(item, dict):
        # Format 2: {...}
        text = item.get('text', '')
        entities = item.get('entities', [])
    else:
        return None, None
    
    return text, entities

def analyze_entities():
    """Analyze entity quality in the dataset"""
    print("[*] Loading data for analysis...\n")
    data = load_data()
    
    issues = {
        'overlapping': [],
        'out_of_bounds': [],
        'negative_range': [],
        'empty_entities': [],
        'whitespace_only': [],
        'duplicates': [],
        'invalid_labels': set(),
        'span_mismatches': []
    }
    
    valid_labels = {'company', 'address', 'date', 'total'}
    docs_with_issues = 0
    total_docs = 0
    total_entities = 0
    
    for doc_idx, item in enumerate(data):
        text, entities = parse_item(item)
        
        if text is None or entities is None:
            continue
        
        total_docs += 1
        
        if not entities:
            continue
        
        total_entities += len(entities)
        doc_has_issues = False
        
        # Normalize entities to tuples
        normalized_ents = []
        for ent in entities:
            if isinstance(ent, (list, tuple)) and len(ent) >= 3:
                normalized_ents.append((ent[0], ent[1], ent[2]))
            elif isinstance(ent, dict):
                normalized_ents.append((ent.get('start'), ent.get('end'), ent.get('label')))
        
        # Check each entity
        for ent_idx, (start, end, label) in enumerate(normalized_ents):
            # Check label validity
            if label not in valid_labels:
                issues['invalid_labels'].add(label)
                issues['span_mismatches'].append({
                    'doc': doc_idx,
                    'ent': ent_idx,
                    'issue': f'Invalid label: {label}',
                    'range': f'[{start}:{end}]'
                })
                doc_has_issues = True
            
            # Check if start/end are valid
            if start is None or end is None or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                issues['span_mismatches'].append({
                    'doc': doc_idx,
                    'ent': ent_idx,
                    'issue': f'Invalid start/end types: start={type(start).__name__}, end={type(end).__name__}',
                    'range': f'[{start}:{end}]'
                })
                doc_has_issues = True
                continue
            
            start, end = int(start), int(end)
            
            # Check negative range
            if start >= end:
                issues['negative_range'].append({
                    'doc': doc_idx,
                    'ent': ent_idx,
                    'range': f'[{start}:{end}]',
                    'label': label
                })
                doc_has_issues = True
                continue
            
            # Check out of bounds
            if start < 0 or end > len(text):
                issues['out_of_bounds'].append({
                    'doc': doc_idx,
                    'ent': ent_idx,
                    'range': f'[{start}:{end}]',
                    'text_len': len(text),
                    'label': label
                })
                doc_has_issues = True
                continue
            
            # Extract span
            try:
                span_text = text[start:end]
                
                # Check for empty entities
                if not span_text or span_text.isspace():
                    if not span_text:
                        issues['empty_entities'].append({
                            'doc': doc_idx,
                            'ent': ent_idx,
                            'range': f'[{start}:{end}]',
                            'label': label
                        })
                    else:
                        issues['whitespace_only'].append({
                            'doc': doc_idx,
                            'ent': ent_idx,
                            'range': f'[{start}:{end}]',
                            'label': label,
                            'span_repr': repr(span_text)
                        })
                    doc_has_issues = True
            except Exception as e:
                issues['span_mismatches'].append({
                    'doc': doc_idx,
                    'ent': ent_idx,
                    'issue': f'Error extracting span: {str(e)}',
                    'range': f'[{start}:{end}]'
                })
                doc_has_issues = True
        
        # Check for duplicates
        seen = set()
        for start, end, label in normalized_ents:
            key = (start, end, label)
            if key in seen:
                issues['duplicates'].append({
                    'doc': doc_idx,
                    'range': f'[{start}:{end}]',
                    'label': label
                })
                doc_has_issues = True
            seen.add(key)
        
        # Check for overlaps
        sorted_ents = sorted(normalized_ents, key=lambda x: (x[0], x[1]))
        for i in range(len(sorted_ents) - 1):
            _, end1, _ = sorted_ents[i]
            start2, _, _ = sorted_ents[i + 1]
            if end1 > start2:
                issues['overlapping'].append({
                    'doc': doc_idx,
                    'ent1': f'[{sorted_ents[i][0]}:{end1}]',
                    'ent2': f'[{start2}:{sorted_ents[i + 1][1]}]'
                })
                doc_has_issues = True
        
        if doc_has_issues:
            docs_with_issues += 1
    
    # Print summary
    print(f"[SUMMARY]")
    print(f"Total documents analyzed: {total_docs}")
    print(f"Total entities analyzed: {total_entities}")
    print(f"Documents with issues: {docs_with_issues} ({100*docs_with_issues/total_docs:.1f}%)")
    print()
    
    print(f"[ISSUES FOUND]")
    print(f"Overlapping entities: {len(issues['overlapping'])}")
    print(f"Out of bounds: {len(issues['out_of_bounds'])}")
    print(f"Negative range (start >= end): {len(issues['negative_range'])}")
    print(f"Empty entities: {len(issues['empty_entities'])}")
    print(f"Whitespace-only entities: {len(issues['whitespace_only'])}")
    print(f"Duplicate entities: {len(issues['duplicates'])}")
    print(f"Invalid labels: {len(issues['invalid_labels'])} unique - {issues['invalid_labels']}")
    print(f"Other span mismatches: {len(issues['span_mismatches'])}")
    print()
    
    # Print detailed issues if any
    if issues['overlapping']:
        print(f"[ERROR] First 5 overlapping entities:")
        for issue in issues['overlapping'][:5]:
            print(f"  Doc {issue['doc']}: {issue['ent1']} overlaps {issue['ent2']}")
    
    if issues['out_of_bounds']:
        print(f"[ERROR] First 5 out of bounds:")
        for issue in issues['out_of_bounds'][:5]:
            print(f"  Doc {issue['doc']}: [{issue['range']}] exceeds text length {issue['text_len']}")
    
    if issues['negative_range']:
        print(f"[ERROR] First 5 negative ranges:")
        for issue in issues['negative_range'][:5]:
            print(f"  Doc {issue['doc']}: {issue['range']} ({issue['label']})")

if __name__ == '__main__':
    analyze_entities()
