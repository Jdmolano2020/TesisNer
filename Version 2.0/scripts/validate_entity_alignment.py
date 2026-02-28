"""
Script para validar y corregir alineamiento de entidades en datos SROIE

Este script proporciona herramientas para:
1. Validar que las entidades están correctamente alineadas con el texto
2. Identificar y corregir desalineamientos
3. Generar un reporte detallado de problemas
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any
import spacy
from spacy.training import offsets_to_biluo_tags

# Agregar rutas para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from spacy_sroie_augmentation import SROIESpacyAugmenter
from logging_config import get_logger

logger = get_logger(__name__)


def validate_single_sample(text: str, entities: List[Tuple[int, int, str]], 
                          nlp=None) -> Dict[str, Any]:
    """
    Valida una muestra individual de texto y entidades.
    
    Args:
        text: Texto a validar.
        entities: Lista de tuplas (start, end, label).
        nlp: Objeto spaCy Language (opcional, crea uno en blanco si es None).
        
    Returns:
        Diccionario con detalles de validación.
    """
    if nlp is None:
        nlp = spacy.blank('es')
    
    result = {
        'text_length': len(text),
        'num_entities': len(entities),
        'valid': True,
        'issues': [],
        'entity_details': []
    }
    
    # Crear documento para tokenización
    try:
        doc = nlp.make_doc(text)
    except Exception as e:
        result['valid'] = False
        result['issues'].append(f"Error al crear documento spaCy: {str(e)}")
        return result
    
    # Validar cada entidad individualmente
    for i, (start, end, label) in enumerate(entities):
        entity_detail = {
            'index': i,
            'label': label,
            'start': start,
            'end': end,
            'valid': True,
            'text': '',
            'problems': []
        }
        
        # Validaciones básicas
        if start is None or end is None or label is None:
            entity_detail['valid'] = False
            entity_detail['problems'].append('Índice o etiqueta None')
            result['valid'] = False
        elif not isinstance(start, int) or not isinstance(end, int):
            entity_detail['valid'] = False
            entity_detail['problems'].append(f'Índices no son int: {type(start)}, {type(end)}')
            result['valid'] = False
        elif start < 0 or end < 0:
            entity_detail['valid'] = False
            entity_detail['problems'].append(f'Índices negativos: [{start}:{end}]')
            result['valid'] = False
        elif start >= end:
            entity_detail['valid'] = False
            entity_detail['problems'].append(f'start >= end: [{start}:{end}]')
            result['valid'] = False
        elif start > len(text) or end > len(text):
            entity_detail['valid'] = False
            entity_detail['problems'].append(f'Índices fuera de rango: [{start}:{end}] vs text_len={len(text)}')
            result['valid'] = False
        else:
            try:
                span_text = text[start:end]
                entity_detail['text'] = span_text
                
                if not span_text:
                    entity_detail['valid'] = False
                    entity_detail['problems'].append('Span vacío')
                    result['valid'] = False
                elif span_text.isspace():
                    entity_detail['valid'] = False
                    entity_detail['problems'].append('Span contiene solo espacios')
                    result['valid'] = False
                    
            except Exception as e:
                entity_detail['valid'] = False
                entity_detail['problems'].append(f'Error extrayendo span: {str(e)}')
                result['valid'] = False
        
        result['entity_details'].append(entity_detail)
    
    # Validar alineamiento usando offsets_to_biluo_tags
    if entities and result['valid']:
        try:
            biluo_tags = offsets_to_biluo_tags(doc, entities)
            misaligned_count = sum(1 for tag in biluo_tags if tag == '-')
            
            if misaligned_count > 0:
                result['valid'] = False
                result['issues'].append(f'CRÍTICO: {misaligned_count} entidades desalineadas (tags "-" encontrados)')
                logger.warning(f"Desalineamiento detectado en '{text[:50]}...': {misaligned_count} de {len(entities)} entidades")
            else:
                result['issues'].append('✓ Todas las entidades están correctamente alineadas')
                
        except Exception as e:
            result['valid'] = False
            result['issues'].append(f'Error en validación BILUO: {str(e)}')
    
    return result


def validate_spacy_data_file(json_file_path: str, sample_size: int = None) -> Dict[str, Any]:
    """
    Valida un archivoJSON con datos en formato spaCy.
    
    Args:
        json_file_path: Ruta al archivo JSON.
        sample_size: Número de muestras a validar (None = todas).
        
    Returns:
        Diccionario con resultados de validación.
    """
    results = {
        'file': json_file_path,
        'file_exists': os.path.exists(json_file_path),
        'total_samples': 0,
        'valid_samples': 0,
        'invalid_samples': 0,
        'samples_checked': 0,
        'detailed_results': [],
        'summary': {}
    }
    
    if not results['file_exists']:
        logger.error(f"Archivo no encontrado: {json_file_path}")
        return results
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Soportar tanto lista como dict
        if isinstance(data, dict):
            samples = data.get('data', []) if 'data' in data else [data]
        else:
            samples = data
        
        results['total_samples'] = len(samples)
        nlp = spacy.blank('es')
        
        # Determinar cuántas muestras validar
        check_count = sample_size if sample_size else len(samples)
        check_count = min(check_count, len(samples))
        
        logger.info(f"Validando {check_count}/{len(samples)} muestras de {json_file_path}...")
        
        for idx in range(check_count):
            sample = samples[idx]
            
            # Parsear formato
            if isinstance(sample, dict):
                text = sample.get('text', '')
                entities = sample.get('entities', [])
                
                # Convertir dict a tuplas si es necesario
                if entities and isinstance(entities[0], dict):
                    entities = [(e.get('start'), e.get('end'), e.get('label')) for e in entities]
            elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
                text = sample[0]
                ents = sample[1].get('entities', []) if isinstance(sample[1], dict) else sample[1]
                
                if isinstance(ents[0], dict) if ents else False:
                    entities = [(e.get('start'), e.get('end'), e.get('label')) for e in ents]
                else:
                    entities = ents
            else:
                continue
            
            # Validar
            validity = validate_single_sample(text, entities, nlp)
            
            if validity['valid']:
                results['valid_samples'] += 1
            else:
                results['invalid_samples'] += 1
                # Guardar solo los primeros 5 inválidos para no sobrecargar
                if len([x for x in results['detailed_results'] if not x['valid']]) < 5:
                    results['detailed_results'].append({
                        'index': idx,
                        'text_sample': text[:100],
                        'validity': validity
                    })
            
            results['samples_checked'] += 1
        
        # Generar resumen
        results['summary'] = {
            'validation_rate': f"{(results['valid_samples']/results['samples_checked']*100):.1f}%" if results['samples_checked'] > 0 else "N/A",
            'total_checked': results['samples_checked'],
            'issues_found': results['invalid_samples'] > 0,
            'recommendation': 'Los datos están listos para entrenamiento' if results['invalid_samples'] == 0 else 'Se recomienda revisar y reparar los datos antes del entrenamiento'
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Error decodificando JSON: {str(e)}")
        results['error'] = str(e)
    except Exception as e:
        logger.error(f"Error validando archivo: {str(e)}")
        results['error'] = str(e)
    
    return results


def repair_spacy_data_file(json_file_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    Repara un archivo JSON con datos spaCy usando el validador del augmenter.
    
    Args:
        json_file_path: Ruta al archivo JSON con datos.
        output_path: Ruta para guardar datos reparados (default = original con _repaired)
        
    Returns:
        Diccionario con estadísticas de reparación.
    """
    if output_path is None:
        base = json_file_path.replace('.json', '')
        output_path = f"{base}_repaired.json"
    
    logger.info(f"Reparando {json_file_path}...")
    
    # Cargar datos
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error cargando archivo: {str(e)}")
        return {'error': str(e)}
    
    # Convertir a formato de tuplas para el augmenter
    if isinstance(data, dict):
        samples = data.get('data', []) if 'data' in data else [data]
    else:
        samples = data
    
    spacy_data = []
    for sample in samples:
        if isinstance(sample, dict):
            text = sample.get('text', '')
            entities = sample.get('entities', [])
            
            if isinstance(entities, list) and entities and isinstance(entities[0], dict):
                entities = [(e.get('start'), e.get('end'), e.get('label')) for e in entities]
        else:
            continue
        
        spacy_data.append((text, {'entities': entities}))
    
    # Reparar usando el augmenter
    augmenter = SROIESpacyAugmenter(use_gpu=False)
    augmenter.initialize_spacy()
    
    repaired_data, stats = augmenter.validate_and_repair_training_data(spacy_data, remove_invalid=True)
    
    # Convertir de vuelta a formato dict
    repaired_output = []
    for text, annotations in repaired_data:
        entities = annotations.get('entities', [])
        # Convertir tuplas a dicts para JSON
        ent_dicts = [{'start': s, 'end': e, 'label': l} for s, e, l in entities]
        repaired_output.append({'text': text, 'entities': ent_dicts})
    
    # Guardar
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(repaired_output, f, ensure_ascii=False, indent=2)
        logger.info(f"Datos reparados guardados en: {output_path}")
    except Exception as e:
        logger.error(f"Error guardando archivo reparado: {str(e)}")
        return {'error': str(e), 'stats': stats}
    
    return {
        'stats': stats,
        'output_file': output_path,
        'original_file': json_file_path
    }


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(description='Validar y reparar alineamiento de entidades spaCy')
    parser.add_argument('action', choices=['validate', 'repair'], help='Acción a ejecutar')
    parser.add_argument('data_file', help='Archivo JSON con datos spaCy')
    parser.add_argument('--output', help='Ruta de salida para datos reparados (para action=repair)')
    parser.add_argument('--sample', type=int, help='Número de muestras a validar (default=todas)')
    
    args = parser.parse_args()
    
    if args.action == 'validate':
        logger.info(f"Validando {args.data_file}...")
        results = validate_spacy_data_file(args.data_file, sample_size=args.sample)
        
        print("\n" + "="*60)
        print("REPORTE DE VALIDACIÓN")
        print("="*60)
        print(f"Archivo: {results['file']}")
        print(f"Total muestras: {results['total_samples']}")
        print(f"Muestras validadas: {results['samples_checked']}")
        print(f"Válidas: {results['valid_samples']}")
        print(f"Inválidas: {results['invalid_samples']}")
        if 'summary' in results:
            print(f"\nTasa de validación: {results['summary'].get('validation_rate', 'N/A')}")
            print(f"Recomendación: {results['summary'].get('recommendation', 'N/A')}")
        
        if results['detailed_results']:
            print(f"\nPrimeros problemas encontrados:")
            for detail in results['detailed_results']:
                print(f"\n  Índice {detail['index']}: {detail['text_sample']}...")
                for issue in detail['validity'].get('issues', []):
                    print(f"    - {issue}")
        
        print("="*60 + "\n")
        
    elif args.action == 'repair':
        logger.info(f"Reparando {args.data_file}...")
        results = repair_spacy_data_file(args.data_file, output_path=args.output)
        
        if 'error' in results:
            logger.error(f"Error durante reparación: {results['error']}")
        else:
            stats = results.get('stats', {})
            print("\n" + "="*60)
            print("REPORTE DE REPARACIÓN")
            print("="*60)
            print(f"Archivo original: {results['original_file']}")
            print(f"Archivo reparado: {results['output_file']}")
            print(f"Total muestras: {stats.get('total_samples', 0)}")
            print(f"Válidas sin cambios: {stats.get('valid_without_changes', 0)}")
            print(f"Reparadas: {stats.get('repaired', 0)}")
            print(f"Eliminadas (inválidas): {stats.get('removed_invalid', 0)}")
            print("="*60 + "\n")


if __name__ == '__main__':
    main()
