#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test para validar que las funciones mejoradas de alineamiento funcionan.
Usa los casos reales de los logs.
"""

import sys
import json
sys.path.insert(0, '.')

from spacy_sroie_augmentation import SROIESpacyAugmenter

def test_case_1():
    """Primer caso del log: entidades problemáticas"""
    augmenter = SROIESpacyAugmenter()
    augmenter.initialize_spacy()
    
    text = """TAN WOON YANN MR D.I.Y. (JOHOR) SDN BHD (CO.REG : 933109-X) NO.89&91, JALAN UTAMA, TAMAN MUTIA RINI, 81300 SKUDAI, JOHOR. -INVOICE- CHOPPING BOARD 35.5X25.5CM 803M# EZ10HD05 - 24 8970669 1 X 19.00 19.00 AIR PRESSURE SPRAYER SX-575-1 1.5L HC03-7 - 15 9066468 1 X 8.02 8.02 WAXCO WINDSHILED CLEANER 120ML WA14-3A - 48 9557031100236 1 X 3.02 3.02 BOPP TAPE 48MM*100M CLEAR FZ-04 - 36 6935818350846 1 X 3.88 3.88 ITEM(S) : 4 QTY(S) : 4 TOTAL RM 33.92 ROUNDING ADJUSTMENT -RM 0.02 TOTAL ROUNDED RM 36.04 CASH RM 50.00 CHANGE RM 16.10 12-01-19 21:13 SH01 ZK09 T4 R000027830 OPERATOR TRAINEE CASHIER EXCHANGE ARE ALLOWED WITHIN 7 DIMILIKI OLEH : DOVE HOLDINGS SDN BHDLY NO CASH REFUND."""
    
    # Entidades originales que estaban desalineadas
    entities = [(623, 660, 'company'), (60, 121, 'address'), (493, 498, 'total'), (529, 537, 'date')]
    
    print("=" * 80)
    print("TEST CASE 1")
    print("=" * 80)
    print(f"Texto: {text[:100]}...")
    print(f"Entidades originales: {entities}")
    
    # Validar
    is_valid, issues = augmenter.validate_entity_alignment(text, entities)
    print(f"¿Válidas?: {is_valid}")
    if issues:
        print("Problemas encontrados:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Reparar
    if not is_valid:
        fixed = augmenter.fix_misaligned_entities(text, entities, strict=False)
        print(f"\nEntidades reparadas: {fixed}")
        
        # Validar de nuevo
        is_valid_fixed, issues_fixed = augmenter.validate_entity_alignment(text, fixed)
        print(f"¿Reparadas son válidas?: {is_valid_fixed}")
        if not is_valid_fixed:
            print("Problemas en reparadas:")
            for issue in issues_fixed:
                print(f"  - {issue}")
    
    print()

def test_case_2():
    """Segundo caso del log: entidades desalineadas"""
    augmenter = SROIESpacyAugmenter()
    augmenter.initialize_spacy()
    
    text = """THAN WOON YANN YONGFATT ENTERPRISE (JM0517726) __NO 122.124. JALAN DEDAP 13 81100 JOHOR BAHRUL 07-3523888 GST ID: 000849813504 IMPUESTO SIMPLIFICADO EN EFECTO DEL IMPUESTO DOC NO CS00031663 FECHA25/12/2018 TIEMPO DEL USUARIO EN CASO 12 31 00 SALESPERSON REF. Partida Cuantía del IMPUESTO S/PRICE E8318 180.901 80.91 SR. SCHTR BAG 15 TOTAL DE TIEMPO 1 80.91 Total de ventas (excluido GST): 80.91 Desglose: 0,00 GST TOTAL: 0,00 RENDIMIENTO -0.01 TOTAL DE SALAS (INCLUSIVE DE GST) 80."""
    
    # Entidades originales
    entities = [(15, 34, 'company'), (49, 93, 'address'), (195, 205, 'date'), (303, 308, 'total')]
    
    print("=" * 80)
    print("TEST CASE 2")
    print("=" * 80)
    print(f"Texto: {text[:100]}...")
    print(f"Entidades originales: {entities}")
    
    # Validar
    is_valid, issues = augmenter.validate_entity_alignment(text, entities)
    print(f"¿Válidas?: {is_valid}")
    if issues:
        print("Problemas encontrados:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Reparar
    if not is_valid:
        fixed = augmenter.fix_misaligned_entities(text, entities, strict=False)
        print(f"\nEntidades reparadas: {fixed}")
        
        # Validar de nuevo
        is_valid_fixed, issues_fixed = augmenter.validate_entity_alignment(text, fixed)
        print(f"¿Reparadas son válidas?: {is_valid_fixed}")
        if not is_valid_fixed:
            print("Problemas en reparadas:")
            for issue in issues_fixed:
                print(f"  - {issue}")
    
    print()

def test_case_3():
    """Test simple: entidades bien alineadas"""
    augmenter = SROIESpacyAugmenter()
    augmenter.initialize_spacy()
    
    text = "Company Inc. at 123 Main Street, Date: 2023-01-15, Total: $100.00"
    entities = [(0, 12, 'company'), (20, 36, 'address'), (44, 54, 'date'), (59, 65, 'total')]
    
    print("=" * 80)
    print("TEST CASE 3 (Control - entidades simples)")
    print("=" * 80)
    print(f"Texto: {text}")
    print(f"Entidades: {entities}")
    
    is_valid, issues = augmenter.validate_entity_alignment(text, entities)
    print(f"¿Válidas?: {is_valid}")
    if issues:
        print("Problemas:")
        for issue in issues:
            print(f"  - {issue}")
    
    print()

def test_case_4():
    """Cuarto caso del log: entidades desalineadas"""
    augmenter = SROIESpacyAugmenter()
    augmenter.initialize_spacy()
    spacy_aug_file = './output/spacy_augmented_2_samp100_reparado.json'
    print('Cargando datos aumentados spaCy desde checkpoint: %s', spacy_aug_file)
    with open(spacy_aug_file, 'r', encoding='utf-8') as sf:
        serial_aug = json.load(sf)
    augmented_data = []
    for item in serial_aug:
        text = item.get('text', '')
        ents = item.get('entities', [])
        parsed = []
        for ent in ents:
            if isinstance(ent, dict):
                s = ent.get('start')
                e = ent.get('end')
                lab = ent.get('label')
            elif isinstance(ent, (list, tuple)) and len(ent) >= 3:
                s, e, lab = ent[0], ent[1], ent[2]
            else:
                s, e, lab = None, None, None
            parsed.append((s, e, lab))
        augmented_data.append((text, {'entities': parsed}))
    # text = """THAN WOON YANN YONGFATT ENTERPRISE (JM0517726) __NO 122.124. JALAN DEDAP 13 81100 JOHOR BAHRUL 07-3523888 GST ID: 000849813504 IMPUESTO SIMPLIFICADO EN EFECTO DEL IMPUESTO DOC NO CS00031663 FECHA25/12/2018 TIEMPO DEL USUARIO EN CASO 12 31 00 SALESPERSON REF. Partida Cuantía del IMPUESTO S/PRICE E8318 180.901 80.91 SR. SCHTR BAG 15 TOTAL DE TIEMPO 1 80.91 Total de ventas (excluido GST): 80.91 Desglose: 0,00 GST TOTAL: 0,00 RENDIMIENTO -0.01 TOTAL DE SALAS (INCLUSIVE DE GST) 80."""
    
    # Entidades originales
    # entities = [(15, 34, 'company'), (49, 93, 'address'), (195, 205, 'date'), (303, 308, 'total')]
    
    print("=" * 80)
    print("TEST CASE 4")
    print("=" * 80)
    # total_textos = sum(len(item[1]['text']) for item in augmented_data)
    # print(f"Cantidad Texto: {total_textos}")
    # total_entities = sum(len(item[1]['entities']) for item in augmented_data)
    # print(f"Cantidad Entidades originales: {total_entities}")
    
    # Validar
    serial_aug = []
    for idx, (text, annotations) in enumerate(augmented_data):
        entities = annotations.get('entities', [])
        is_valid, issues = augmenter.validate_entity_alignment(text, entities)
        print(f"¿Válidas?: {is_valid}")
        if issues:
            print("Problemas encontrados:")
            for issue in issues:
                print(f"  - {issue}")
    
        # Reparar
        if not is_valid:
            fixed = augmenter.fix_misaligned_entities(text, entities, strict=False)
            print(f"\nEntidades reparadas: {fixed}")
            
            # Validar de nuevo
            is_valid_fixed, issues_fixed = augmenter.validate_entity_alignment(text, fixed)
            print(f"¿Reparadas son válidas?: {is_valid_fixed}")
            if not is_valid_fixed:
                print("Problemas en reparadas:")
                for issue in issues_fixed:
                    print(f"  - {issue}")
            else:
                serial_aug.append({'text': text, 'entities': fixed})
        else:
            serial_aug.append({'text': text, 'entities': entities})
        
    spacy_aug_file = './output/spacy_augmented_2_samp100_reparado.json'
    with open(spacy_aug_file, 'w', encoding='utf-8') as sf:
        json.dump(serial_aug, sf, ensure_ascii=False)
    print()

if __name__ == "__main__":
    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    print("\n✅ Todos los tests completados")
