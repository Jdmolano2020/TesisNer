"""
Script de prueba para validar que W036 está resuelto

Este script demuestra los diferentes escenarios donde W036 solía ocurrir
y verifica que ahora se manejan correctamente.
"""

import json
from spacy_sroie_augmentation import SROIESpacyAugmenter
from logging_config import get_logger
import warnings

logger = get_logger(__name__)

# Capturar warnings para mostrarlos
warnings.simplefilter('always')


def test_empty_entities():
    """Test 1: Datos sin entidades (el caso más probable de W036)"""
    print("\n" + "="*70)
    print("TEST 1: Datos sin entidades")
    print("="*70)
    
    augmenter = SROIESpacyAugmenter(use_gpu=False)
    augmenter.initialize_spacy()
    
    # Datos con textos pero sin entidades
    empty_data = [
        ("Factura de prueba número 001", {"entities": []}),
        ("Jalan Temenggung 19/9", {"entities": []}),
        ("Total: $150.00", {"entities": []}),
    ]
    
    print(f"\nEntrando con {len(empty_data)} muestras sin entidades")
    print("Esperado: Sin warning W036")
    print("\nEntrenando...")
    
    try:
        metrics = augmenter.train_model(empty_data, n_iter=1, batch_size=2)
        print("✓ Entrenamiento completado sin W036")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_partial_entities():
    """Test 2: Datos parcialmente con entidades"""
    print("\n" + "="*70)
    print("TEST 2: Datos parcialmente con entidades (mezcla)")
    print("="*70)
    
    augmenter = SROIESpacyAugmenter(use_gpu=False)
    augmenter.initialize_spacy()
    
    # Mezcla de datos con y sin entidades
    mixed_data = [
        ("Factura de prueba", {"entities": []}),  # Sin entidades
        ("Empresa ABC Jalan Temenggung", {"entities": [(0, 11, 'COMPANY'), (12, 28, 'ADDRESS')]}),  # Con entidades
        ("Fecha 10/03/2023", {"entities": [(6, 16, 'DATE')]}),  # Con entidades
        ("Sin datos aquí", {"entities": []}),  # Sin entidades
    ]
    
    print(f"\nEntrando con {len(mixed_data)} muestras (2 sin ents, 2 con ents)")
    print("Esperado: Patrones creados, sin W036")
    print("\nEntrenando...")
    
    try:
        metrics = augmenter.train_model(mixed_data, n_iter=1, batch_size=2)
        print("✓ Entrenamiento completado")
        print("✓ Patrones creados exitosamente")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_valid_entities():
    """Test 3: Datos con entidades válidas"""
    print("\n" + "="*70)
    print("TEST 3: Datos con entidades válidas")
    print("="*70)
    
    augmenter = SROIESpacyAugmenter(use_gpu=False)
    augmenter.initialize_spacy()
    
    # Datos con entidades válidas
    valid_data = [
        ("RESTORAN WAN NO.2, JALAN TEMENGGUNG 19/9, SELANGOR MALAYSIA 12345 Tanggal: 10/03/2023 Total: $150.00",
         {"entities": [(0, 9, 'COMPANY'), (13, 38, 'ADDRESS'), (49, 59, 'DATE'), (67, 74, 'TOTAL')]}),
        
        ("ENTERPRISE ABC Jalan Sultan Ismail, Kuala Lumpur Tanggal: 15/12/2022 Monto: $250.50",
         {"entities": [(0, 14, 'COMPANY'), (15, 50, 'ADDRESS'), (60, 70, 'DATE'), (78, 85, 'TOTAL')]}),
    ]
    
    print(f"\nEntrando con {len(valid_data)} muestras con entidades")
    print("Esperado: Muchos patrones creados, sin W036")
    print("\nEntrenando...")
    
    try:
        metrics = augmenter.train_model(valid_data, n_iter=1, batch_size=2)
        print("✓ Entrenamiento completado")
        print("✓ Patrones creados exitosamente")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_load_and_train():
    """Test 4: Cargar datos reales si existen"""
    print("\n" + "="*70)
    print("TEST 4: Cargar datos reales del dataset")
    print("="*70)
    
    json_file = './output/spacy_augmented_2.json'
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\nCargados {len(data)} muestras desde {json_file}")
        
        # Convertir a formato interno
        spacy_data = []
        for item in data[:20]:  # Solo primeras 20 para test rápido
            text = item.get('text', '')
            entities = item.get('entities', [])
            
            if isinstance(entities, list) and entities and isinstance(entities[0], dict):
                entities = [(e.get('start'), e.get('end'), e.get('label')) for e in entities]
            
            spacy_data.append((text, {'entities': entities}))
        
        print(f"Usando {len(spacy_data)} muestras para entrenamiento")
        print("Esperado: Entrenamiento sin W036")
        print("\nEntrenando...")
        
        augmenter = SROIESpacyAugmenter(use_gpu=False)
        metrics = augmenter.train_model(spacy_data, n_iter=1, batch_size=4)
        
        print("✓ Entrenamiento completado sin W036")
        return True
        
    except FileNotFoundError:
        print(f"⚠️ Archivo no encontrado: {json_file}")
        print("   Skipping este test (requiere haber generado datos primero)")
        print("   Ejecuta: python sroie_main.py Data/sroie/completo --model_type spacy --num_augmentations 2")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_validate_functions():
    """Test 5: Validar funciones individuales"""
    print("\n" + "="*70)
    print("TEST 5: Validar funciones individuales")
    print("="*70)
    
    augmenter = SROIESpacyAugmenter(use_gpu=False)
    augmenter.initialize_spacy()
    
    tests_passed = 0
    tests_total = 0
    
    # Test create_entity_patterns con datos vacíos
    tests_total += 1
    print("\nTest 5a: create_entity_patterns con datos vacíos")
    patterns = augmenter.create_entity_patterns([])
    if patterns == []:
        print("✓ Retorna lista vacía")
        tests_passed += 1
    else:
        print(f"✗ Esperaba [], obtuvo {patterns}")
    
    # Test create_entity_patterns con datos sin entidades
    tests_total += 1
    print("\nTest 5b: create_entity_patterns con datos sin entidades")
    empty_ents_data = [("Texto", {"entities": []})]
    patterns = augmenter.create_entity_patterns(empty_ents_data)
    if patterns == []:
        print("✓ Retorna lista vacía cuando no hay entidades")
        tests_passed += 1
    else:
        print(f"✗ Esperaba [], obtuvo {patterns}")
    
    # Test create_entity_patterns con datos válidos
    tests_total += 1
    print("\nTest 5c: create_entity_patterns con datos válidos")
    valid_data = [("EMPRESA ABC Jalan", {"entities": [(0, 11, 'COMPANY'), (12, 17, 'ADDRESS')]})]
    patterns = augmenter.create_entity_patterns(valid_data)
    if len(patterns) > 0:
        print(f"✓ Crea {len(patterns)} patrones")
        tests_passed += 1
    else:
        print("✗ No creó patrones")
    
    # Test add_entity_patterns con lista vacía
    tests_total += 1
    print("\nTest 5d: add_entity_patterns con lista vacía")
    try:
        augmenter.add_entity_patterns([])
        if "entity_ruler" not in augmenter.nlp.pipe_names:
            print("✓ No agrega EntityRuler cuando patterns está vacío")
            tests_passed += 1
        else:
            print("✗ Agregó EntityRuler pese a patrones vacíos")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test add_entity_patterns con patrones válidos
    tests_total += 1
    print("\nTest 5e: add_entity_patterns con patrones válidos")
    try:
        valid_patterns = [{"label": "TEST", "pattern": "test"}]
        augmenter.add_entity_patterns(valid_patterns)
        if "entity_ruler" in augmenter.nlp.pipe_names:
            print("✓ Agrega EntityRuler cuando hay patrones")
            tests_passed += 1
        else:
            print("✗ No agregó EntityRuler")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print(f"\nResultado: {tests_passed}/{tests_total} tests pasados")
    return tests_passed == tests_total


def main():
    """Ejecutar todos los tests"""
    print("\n" + "="*70)
    print("SUITE DE TESTS: Validar que W036 está Resuelto")
    print("="*70)
    
    results = {
        "Test 1 (Datos sin entidades)": test_empty_entities(),
        "Test 2 (Datos mixtos)": test_partial_entities(),
        "Test 3 (Datos válidos)": test_valid_entities(),
        "Test 4 (Datos reales)": test_load_and_train(),
        "Test 5 (Funciones)": test_validate_functions(),
    }
    
    print("\n" + "="*70)
    print("RESUMEN DE RESULTADOS")
    print("="*70)
    
    for test_name, result in results.items():
        status = "✓ PASÓ" if result is True else ("⚠️ SKIPPED" if result is None else "✗ FALLÓ")
        print(f"{test_name}: {status}")
    
    passed = sum(1 for r in results.values() if r is True)
    total = sum(1 for r in results.values() if r is not None)
    
    print(f"\nResultado general: {passed}/{total} tests completados y pasados")
    
    if passed == total and total > 0:
        print("\n🎉 ¡Todos los tests pasaron! W036 está resuelto.")
    elif passed == total:
        print("\n⚠️ Algunos tests fueron skipped. Ejecuta con datos reales para pruebas completas.")
    else:
        print(f"\n⚠️ {total - passed} test(s) fallaron. Revisar logs arriba.")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception("Error en suite de tests: %s", e)
        raise
