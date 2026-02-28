#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demostración visual de cómo funciona la reparación de entidades.
"""

import sys
sys.path.insert(0, '.')

from spacy_sroie_augmentation import SROIESpacyAugmenter
import spacy

def visualize_alignment(text, entities, title=""):
    """Visualiza la alineación de entidades con el texto y los tokens."""
    print("\n" + "="*80)
    print(f"📊 {title}")
    print("="*80)
    
    # Crear documento spaCy
    nlp = spacy.blank('es')
    doc = nlp.make_doc(text)
    
    print(f"\n📝 Texto ({len(text)} caracteres):")
    print(f"   {text}")
    
    print(f"\n🔤 Tokens ({len(doc)} tokens):")
    for i, token in enumerate(doc):
        print(f"   Token {i:2d}: [{token.idx:3d}:{token.idx + len(token.text):3d}] '{token.text}'")
    
    print(f"\n🏷️  Entidades ({len(entities)} entidades):")
    for i, (start, end, label) in enumerate(entities):
        span_text = text[start:end]
        print(f"   [{i}] [{start:3d}:{end:3d}] '{span_text}' (label={label})")
        
        # Intentar alinear con char_span
        try:
            span = doc.char_span(start, end, alignment_mode="contract")
            if span is None:
                span = doc.char_span(start, end, alignment_mode="expand")
            
            if span is not None:
                print(f"       ✅ Alineada a: [{span.start_char:3d}:{span.end_char:3d}] '{span.text}'")
            else:
                print(f"       ❌ No se puede alinear")
        except Exception as e:
            print(f"       ❌ Error: {e}")

def demo_caso_problematico():
    """Demuestra cómo se reparan las entidades problemáticas."""
    print("\n\n" + "█"*80)
    print("█ DEMOSTRACIÓN: REPARACIÓN DE ENTIDADES DESALINEADAS")
    print("█"*80)
    
    augmenter = SROIESpacyAugmenter()
    augmenter.initialize_spacy()
    
    # Caso 1: Entidades del log original
    text = "TAN WOON YANN MR D.I.Y. (JOHOR) SDN BHD (CO.REG : 933109-X) NO.89&91, JALAN UTAMA, TAMAN MUTIA RINI, 81300 SKUDAI, JOHOR."
    entities = [(0, 25, 'company'), (28, 85, 'address')]  # Simplificado
    
    visualize_alignment(text, entities, "ANTES de reparación")
    
    # Validar
    is_valid, issues = augmenter.validate_entity_alignment(text, entities)
    print(f"\n🔍 Validación: {'✅ VÁLIDO' if is_valid else '❌ INVÁLIDO'}")
    
    if not is_valid:
        print("   Problemas:")
        for issue in issues:
            print(f"   - {issue}")
        
        # Reparar
        fixed = augmenter.fix_misaligned_entities(text, entities, strict=False)
        
        print(f"\n🔧 Reparación:")
        print(f"   Originales: {entities}")
        print(f"   Reparadas:  {fixed}")
        
        # Validar nuevamente
        is_valid_fixed, issues_fixed = augmenter.validate_entity_alignment(text, fixed)
        print(f"\n🔍 Validación después: {'✅ VÁLIDO' if is_valid_fixed else '❌ INVÁLIDO'}")
        
        visualize_alignment(text, fixed, "DESPUÉS de reparación")

def demo_multiples_estrategias():
    """Demuestra las 3 estrategias de reparación."""
    print("\n\n" + "█"*80)
    print("█ DEMOSTRACIÓN: ESTRATEGIAS DE REPARACIÓN")
    print("█"*80)
    
    augmenter = SROIESpacyAugmenter()
    augmenter.initialize_spacy()
    
    cases = [
        {
            "name": "Estrategia 1: Alineación directa con char_span",
            "text": "Company Name Address City 2023-01-15 Total: $100",
            "entities": [(0, 12, 'ORG'), (14, 21, 'LOC')],
        },
        {
            "name": "Estrategia 2: Búsqueda de texto exacto desplazado",
            "text": "The quick brown fox jumps over lazy dog",
            "entities": [(4, 9, 'ADJ'), (16, 20, 'ADJ')],
        },
        {
            "name": "Estrategia 3: Normalización espacios",
            "text": "Item1    Item2    Item3",  # Espacios múltiples
            "entities": [(0, 5, 'ITEM'), (9, 14, 'ITEM')],
        },
    ]
    
    for case in cases:
        print(f"\n\n{'─'*80}")
        print(f"📌 {case['name']}")
        print(f"{'─'*80}")
        
        text = case['text']
        entities = case['entities']
        
        visualize_alignment(text, entities, "Original")
        
        # Validar y reparar
        is_valid, _ = augmenter.validate_entity_alignment(text, entities)
        if not is_valid:
            fixed = augmenter.fix_misaligned_entities(text, entities, strict=False)
            print(f"\n✅ Reparada a: {fixed}")

def demo_comparacion_metodos():
    """Compara el método antiguo vs nuevo."""
    print("\n\n" + "█"*80)
    print("█ COMPARACIÓN: ANTIGUO vs NUEVO MÉTODO")
    print("█"*80)
    
    from spacy.training import offsets_to_biluo_tags
    import spacy
    
    text = "COMPANY ADDRESS TOTAL"
    entities = [(0, 7, 'ORG'), (8, 15, 'LOC'), (16, 21, 'NUM')]
    
    nlp = spacy.blank('es')
    doc = nlp.make_doc(text)
    
    print(f"\n📝 Texto: {text}")
    print(f"🏷️  Entidades: {entities}")
    
    # Método antiguo: offsets_to_biluo_tags
    print(f"\n🔴 MÉTODO ANTIGUO (offsets_to_biluo_tags):")
    try:
        biluo_tags = offsets_to_biluo_tags(doc, entities)
        has_misaligned = '-' in biluo_tags
        print(f"   BILUO tags: {biluo_tags}")
        print(f"   ¿Desalineadas?: {'❌ SÍ' if has_misaligned else '✅ NO'}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Método nuevo: char_span con alignment_mode
    print(f"\n🟢 MÉTODO NUEVO (char_span + alignment_mode):")
    augmenter = SROIESpacyAugmenter()
    augmenter.initialize_spacy()
    
    is_valid, issues = augmenter.validate_entity_alignment(text, entities)
    print(f"   ¿Válidas?: {'✅ SÍ' if is_valid else '❌ NO'}")
    if issues:
        for issue in issues:
            print(f"   Issue: {issue}")

def main():
    """Ejecuta todas las demostraciones."""
    print("\n\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  🔧 DEMOSTRACIÓN: REPARACIÓN AUTOMÁTICA DE ENTIDADES DESALINEADAS".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    # Demo 1: Caso problemático
    demo_caso_problematico()
    
    # Demo 2: Múltiples estrategias
    demo_multiples_estrategias()
    
    # Demo 3: Comparación de métodos
    demo_comparacion_metodos()
    
    print("\n\n" + "="*80)
    print("✅ DEMOSTRACIONES COMPLETADAS")
    print("="*80)
    print("\nResumen:")
    print("  1. Las nuevas funciones alinean automáticamente a límites de tokens")
    print("  2. Usa múltiples estrategias para maximizar el éxito de reparación")
    print("  3. Compatible con el pipeline de spaCy para entrenamiento")
    print("\n")

if __name__ == "__main__":
    main()
