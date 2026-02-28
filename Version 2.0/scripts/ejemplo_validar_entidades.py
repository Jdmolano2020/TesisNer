"""
Script de ejemplo: Validar datos de entrenamiento spaCy

Este script demuestra cómo usar las nuevas funciones de validación
para identificar y arreglar problemas de alineamiento de entidades.
"""

import json
from spacy_sroie_augmentation import SROIESpacyAugmenter
from logging_config import get_logger

logger = get_logger(__name__)


def ejemplo_validar_muestra_individual():
    """Ejemplo 1: Validar una muestra individual"""
    print("\n" + "="*60)
    print("EJEMPLO 1: Validar una muestra individual")
    print("="*60)
    
    augmenter = SROIESpacyAugmenter(use_gpu=False)
    augmenter.initialize_spacy()
    
    # Muestra 1: Correctamente alineada
    text1 = "RESTORAN WAN NO.2, JALAN TEMENGGUNG"
    entities1 = [(0, 9, 'COMPANY'), (10, 35, 'ADDRESS')]
    
    print(f"\nMuestra 1 (debe estar bien alineada):")
    print(f"Text: {text1}")
    print(f"Entities: {entities1}")
    
    is_valid, issues = augmenter.validate_entity_alignment(text1, entities1)
    print(f"¿Válida? {is_valid}")
    if issues:
        for issue in issues:
            print(f"  → {issue}")
    
    # Muestra 2: Incorrectamente alineada
    text2 = "RESTORAN WAN NO.2, JALAN TEMENGGUNG"
    entities2 = [(0, 9, 'COMPANY'), (15, 50, 'ADDRESS')]  # Rango incorrecto
    
    print(f"\nMuestra 2 (debe estar mal alineada):")
    print(f"Text: {text2}")
    print(f"Entities: {entities2}")
    
    is_valid, issues = augmenter.validate_entity_alignment(text2, entities2)
    print(f"¿Válida? {is_valid}")
    if issues:
        for issue in issues:
            print(f"  ✗ {issue}")


def ejemplo_reparar_entidades():
    """Ejemplo 2: Reparar entidades desalineadas"""
    print("\n" + "="*60)
    print("EJEMPLO 2: Reparar entidades desalineadas")
    print("="*60)
    
    augmenter = SROIESpacyAugmenter(use_gpu=False)
    augmenter.initialize_spacy()
    
    # Texto donde el span está en otro lugar
    text = "RESTORAN WAN NO.2, JALAN TEMENGGUNG 19/9"
    
    # Entidades con offsets incorrectos
    bad_entities = [
        (0, 9, 'COMPANY'),  # "RESTORAN" ← Correcto
        (50, 70, 'ADDRESS')  # ← Incorrecto, está fuera del texto
    ]
    
    print(f"Texto: {text}")
    print(f"Longitud: {len(text)} caracteres")
    print(f"\nEntidades defectuosas:")
    for start, end, label in bad_entities:
        print(f"  [{start}:{end}] {label} → fuera de rango")
    
    # Intentar reparar
    fixed = augmenter.fix_misaligned_entities(text, bad_entities, strict=False)
    
    print(f"\nEntidades reparadas:")
    if fixed:
        for start, end, label in fixed:
            span = text[start:end]
            print(f"  [{start}:{end}] {label} → '{span}'")
    else:
        print("  (No se pudo reparar)")


def ejemplo_validar_dataset_completo():
    """Ejemplo 3: Validar un dataset completo"""
    print("\n" + "="*60)
    print("EJEMPLO 3: Validar un dataset completo")
    print("="*60)
    
    augmenter = SROIESpacyAugmenter(use_gpu=False)
    augmenter.initialize_spacy()
    
    # Crear datos de ejemplo (algunos correctos, algunos incorrectos)
    spacy_data = [
        ("RESTORAN WAN NO.2, JALAN TEMENGGUNG", 
         {'entities': [(0, 9, 'COMPANY'), (10, 35, 'ADDRESS')]}),
        
        ("Fecha: 10/03/2023 Monto: $150.00",
         {'entities': [(7, 17, 'DATE'), (25, 32, 'TOTAL')]}),
        
        ("EMPRESA ABC DEL 15/12/2022",
         {'entities': [(0, 11, 'COMPANY'), (16, 26, 'DATE'), (100, 150, 'BAD')]}),  # BAD entity
    ]
    
    print(f"Dataset con {len(spacy_data)} muestras")
    print(f"(2 datos correctos, 1 con entidad fuera de rango)")
    
    # Validar y reparar
    repaired_data, stats = augmenter.validate_and_repair_training_data(
        spacy_data, 
        remove_invalid=True
    )
    
    print(f"\nResultados:")
    print(f"  Total: {stats['total_samples']}")
    print(f"  Válidos sin cambios: {stats['valid_without_changes']}")
    print(f"  Reparados: {stats['repaired']}")
    print(f"  Eliminados: {stats['removed_invalid']}")
    print(f"\nDatos finales: {len(repaired_data)} muestras")
    
    if stats['sample_issues']:
        print(f"\nProblemas encontrados:")
        for issue in stats['sample_issues']:
            print(f"  Muestra {issue['index']}: {issue['issues']}")


def ejemplo_cargar_y_validar_archivo():
    """Ejemplo 4: Cargar y validar un archivo JSON existente"""
    print("\n" + "="*60)
    print("EJEMPLO 4: Cargar y validar archivo JSON")
    print("="*60)
    
    json_file = './output/spacy_augmented_2.json'
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Archivo: {json_file}")
        print(f"Muestras cargadas: {len(data)}")
        
        # Convertir a formato interno
        spacy_data = []
        for item in data[:10]:  # Usar solo las primeras 10 para este ejemplo
            text = item.get('text', '')
            entities = item.get('entities', [])
            
            # Convertir dicts a tuplas
            if entities and isinstance(entities[0], dict):
                entities = [(e.get('start'), e.get('end'), e.get('label')) for e in entities]
            
            spacy_data.append((text, {'entities': entities}))
        
        # Validar
        augmenter = SROIESpacyAugmenter(use_gpu=False)
        augmenter.initialize_spacy()
        
        repaired, stats = augmenter.validate_and_repair_training_data(spacy_data, remove_invalid=True)
        
        print(f"\nValidación de primeras 10 muestras:")
        print(f"  Válidas: {stats['valid_without_changes']}")
        print(f"  Reparadas: {stats['repaired']}")
        print(f"  Eliminadas: {stats['removed_invalid']}")
        
        if stats['sample_issues']:
            print(f"\nProblemas encontrados:")
            for issue in stats['sample_issues']:
                print(f"  - Muestra {issue['index']}: {issue['issues']}")
    
    except FileNotFoundError:
        print(f"⚠️  Archivo no encontrado: {json_file}")
        print("   Primero ejecuta: python sroie_main.py Data/sroie/ --model_type spacy")


def main():
    """Ejecutar todos los ejemplos"""
    try:
        ejemplo_validar_muestra_individual()
        ejemplo_reparar_entidades()
        ejemplo_validar_dataset_completo()
        ejemplo_cargar_y_validar_archivo()
        
        print("\n" + "="*60)
        print("✓ Todos los ejemplos completados")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.exception("Error ejecutando ejemplos: %s", e)
        raise


if __name__ == '__main__':
    main()
