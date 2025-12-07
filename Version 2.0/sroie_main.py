"""
Script Principal para Aumentación de Datos en SROIE

Este script proporciona una interfaz sencilla para aplicar técnicas de aumentación
de datos al dataset SROIE y mejorar la métrica F1 en el reconocimiento de entidades.
"""

import os
import argparse
import torch
import json
import time
from logging_config import get_logger

logger = get_logger(__name__)
from typing import List, Dict, Tuple, Any, Optional

# Importar los módulos de aumentación de datos
from sroie_data_augmentation import SROIEDataAugmenter
from distilbert_sroie_augmentation import SROIEDistilBERTAugmenter
from spacy_sroie_augmentation import SROIESpacyAugmenter
from sroie_evaluation import SROIEEvaluator

def parse_args():
    """
    Parsea los argumentos de línea de comandos.
    
    Returns:
        Argumentos parseados.
    """
    parser = argparse.ArgumentParser(description='Aumentación de Datos para SROIE')
    
    parser.add_argument('data_dir', type=str,
                       help='Directorio con los datos de SROIE')
    
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Directorio para guardar los resultados')
    
    parser.add_argument('--model_type', type=str, choices=['distilbert', 'spacy', 'both'],
                       default='both', help='Tipo de modelo a entrenar')
    
    parser.add_argument('--technique', type=str, 
                       choices=['back_translation', 'ter', 'cwr', 
                              'back_translation+ter', 'back_translation+cwr', 'combined'],
                       default='combined', help='Técnica de aumentación a utilizar')
    
    parser.add_argument('--num_augmentations', type=int, default=2,
                       help='Número de aumentaciones a generar por ejemplo')
    
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Tamaño del lote para entrenamiento (mayor aprovecha más GPU/paralelismo)')
    
    parser.add_argument('--n_iter', type=int, default=50,
                       help='Número de iteraciones/épocas de entrenamiento')
    
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluar diferentes técnicas de aumentación')
    
    parser.add_argument('--use_gpu', action='store_true',
                       help='Usar GPU para el entrenamiento')
    
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Número de workers para paralelismo en carga de datos (0=no paralelo). Si no se especifica, se ajusta automáticamente según memoria/plataforma')
    
    parser.add_argument('--reset_state', action='store_true',
                       help='Borrar estado de ejecución y comenzar desde el paso 1')
    
    return parser.parse_args()

def main():
    """
    Función principal.
    """
    # Parsear argumentos
    args = parse_args()
    
    # Ajuste automático de num_workers cuando no se especifica
    if args.num_workers is None:
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / (1024**3)
        except Exception:
            psutil = None
            avail_gb = None
        if os.name == 'nt':
            # En Windows el método 'spawn' tiene overhead mayor; ser conservador
            if avail_gb is not None and avail_gb < 4:
                chosen_workers = 0
            else:
                chosen_workers = min(2, max(0, (os.cpu_count() or 1) - 1))
        else:
            # Unix-like: permitir más workers si hay RAM suficiente
            if avail_gb is not None:
                if avail_gb < 2:
                    chosen_workers = 0
                elif avail_gb < 8:
                    chosen_workers = min(2, max(0, (os.cpu_count() or 1) - 1))
                else:
                    chosen_workers = min(8, max(0, (os.cpu_count() or 1) - 1))
            else:
                chosen_workers = min(4, max(0, (os.cpu_count() or 1) - 1))
        args.num_workers = int(chosen_workers)
        logger.info("num_workers no especificado: ajustado automáticamente a %d (avail_gb=%s)", args.num_workers, str(avail_gb))
    else:
        logger.info("num_workers especificado por usuario: %d", args.num_workers)
    
    # Crear directorios de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Verificar disponibilidad de GPU
    use_gpu = args.use_gpu and torch.cuda.is_available()
    if args.use_gpu and not torch.cuda.is_available():
        logger.info("ADVERTENCIA: GPU solicitada pero no disponible. Usando CPU.")
    
    # Guardar configuración
    config = vars(args)
    config['use_gpu'] = use_gpu
    
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Estado de ejecución (para reanudar pasos)
    state_file = os.path.join(args.output_dir, '.run_state.json')
    def load_state():
        if os.path.exists(state_file) and not args.reset_state:
            try:
                with open(state_file, 'r', encoding='utf-8') as sf:
                    return json.load(sf)
            except Exception:
                return {}
        return {}

    def save_state(state: dict):
        try:
            with open(state_file, 'w', encoding='utf-8') as sf:
                json.dump(state, sf, indent=2)
        except Exception:
            logger.exception('No se pudo guardar el estado de ejecución')

    state = load_state()
    if args.reset_state and os.path.exists(state_file):
        try:
            os.remove(state_file)
            state = {}
        except Exception:
            pass
    
    # Modo de evaluación
    if args.evaluate:
        logger.info("Modo de evaluación: comparando técnicas de aumentación de datos")
        
        # Crear evaluador
        evaluator = SROIEEvaluator(args.data_dir, use_gpu=use_gpu)
        
        # Cargar datos (paso 1)
        if not state.get('load_data'):
            evaluator.load_data()
            state['load_data'] = True
            save_state(state)
        else:
            logger.info('Estado detectado: datos ya cargados, saltando carga de datos')
        
        
        # Evaluar técnicas según el modelo seleccionado
        if args.model_type in ['distilbert', 'both']:
            logger.info("Evaluando técnicas para DistilBERT...")
            distilbert_results = evaluator.evaluate_augmentation_techniques(
                model_type="distilbert",
                techniques=["original", "back_translation", "ter", "cwr", 
                           "back_translation+ter", "back_translation+cwr", "combined"],
                num_augmentations=[1, 2, 3],
                n_iter=args.n_iter,
                batch_size=args.batch_size
            )
            
            # Visualizar resultados
            best_distilbert_config = evaluator.visualize_results(distilbert_results, "distilbert")
            
            # Entrenar modelo final con la mejor configuración
            if not state.get('distilbert_trained'):
                final_distilbert_model = evaluator.train_best_model(
                model_type="distilbert",
                technique=best_distilbert_config["technique"],
                num_augmentations=best_distilbert_config["num_augmentations"],
                n_iter=args.n_iter * 2,  # Más iteraciones para el modelo final
                batch_size=args.batch_size
                )
                state['distilbert_trained'] = True
                save_state(state)
            else:
                logger.info('Estado detectado: DistilBERT ya entrenado, saltando entrenamiento final')
                # cargar referencia al modelo final si es necesario
                final_distilbert_model = evaluator.train_best_model(model_type="distilbert",
                                                                   technique=best_distilbert_config["technique"],
                                                                   num_augmentations=best_distilbert_config["num_augmentations"],
                                                                   n_iter=0, batch_size=args.batch_size)

            logger.info("Modelo DistilBERT final guardado en: %s", os.path.join(evaluator.results_dir, 'final_distilbert_model'))
            logger.info("Modelo DistilBERT final guardado")
        if args.model_type in ['spacy', 'both']:
            logger.info("Evaluando técnicas para spaCy...")

            spacy_results = evaluator.evaluate_augmentation_techniques(
                model_type="spacy",
                techniques=["original", "back_translation", "ter", "cwr", 
                           "back_translation+ter", "back_translation+cwr", "combined"],
                num_augmentations=[1, 2, 3],
                n_iter=args.n_iter,
                batch_size=args.batch_size
            )
            
            # Visualizar resultados
            best_spacy_config = evaluator.visualize_results(spacy_results, "spacy")
            
            # Entrenar modelo final con la mejor configuración
            if not state.get('spacy_trained'):
                final_spacy_model = evaluator.train_best_model(
                model_type="spacy",
                technique=best_spacy_config["technique"],
                num_augmentations=best_spacy_config["num_augmentations"],
                n_iter=args.n_iter * 2,  # Más iteraciones para el modelo final
                batch_size=args.batch_size
                )
                state['spacy_trained'] = True
                save_state(state)
            else:
                logger.info('Estado detectado: spaCy ya entrenado, saltando entrenamiento final')
                final_spacy_model = evaluator.train_best_model(model_type="spacy",
                                                             technique=best_spacy_config["technique"],
                                                             num_augmentations=best_spacy_config["num_augmentations"],
                                                             n_iter=0, batch_size=args.batch_size)

            logger.info("Modelo spaCy final guardado en: %s", os.path.join(evaluator.results_dir, 'final_spacy_model'))
            logger.info("Modelo spaCy final guardado...")
    
    # Modo de entrenamiento directo
    else:
        logger.info("Entrenando modelo(s) con técnica: %s, aumentaciones: %d", args.technique, args.num_augmentations)
 
        # Entrenar según el modelo seleccionado
        if args.model_type in ['distilbert', 'both']:
            logger.info("Entrenando modelo DistilBERT...")
            
            # Crear aumentador
            distilbert_augmenter = SROIEDistilBERTAugmenter(use_gpu=use_gpu)
            
            # Cargar datos (con checkpoint)
            distilbert_data_file = os.path.join(args.output_dir, 'distilbert_loaded.json')
            if not state.get('distilbert_data_loaded') or not os.path.exists(distilbert_data_file):
                train_texts, train_tags = distilbert_augmenter.load_data(args.data_dir)
                try:
                    with open(distilbert_data_file, 'w', encoding='utf-8') as df:
                        json.dump({'texts': train_texts, 'tags': train_tags}, df, ensure_ascii=False)
                    state['distilbert_data_loaded'] = True
                    save_state(state)
                    logger.info('Datos DistilBERT cargados y guardados en checkpoint: %s', distilbert_data_file)
                except Exception:
                    logger.exception('No se pudo guardar checkpoint de datos DistilBERT')
            else:
                logger.info('Cargando datos DistilBERT desde checkpoint: %s', distilbert_data_file)
                with open(distilbert_data_file, 'r', encoding='utf-8') as df:
                    dd = json.load(df)
                train_texts = dd.get('texts', [])
                train_tags = dd.get('tags', [])

            # Aumentar datos (con checkpoint por número de aumentaciones)
            distilbert_aug_file = os.path.join(args.output_dir, f'distilbert_augmented_{args.num_augmentations}.json')
            aug_state_key = f'distilbert_augmented_{args.num_augmentations}'
            if not state.get(aug_state_key) or not os.path.exists(distilbert_aug_file):
                augmented_texts, augmented_tags = distilbert_augmenter.augment_data(
                    train_texts, train_tags, num_augmentations=args.num_augmentations
                )
                try:
                    with open(distilbert_aug_file, 'w', encoding='utf-8') as af:
                        json.dump({'texts': augmented_texts, 'tags': augmented_tags}, af, ensure_ascii=False)
                    state[aug_state_key] = True
                    save_state(state)
                    logger.info('Datos aumentados DistilBERT guardados en checkpoint: %s', distilbert_aug_file)
                except Exception:
                    logger.exception('No se pudo guardar checkpoint de datos aumentados DistilBERT')
            else:
                logger.info('Cargando datos aumentados DistilBERT desde checkpoint: %s', distilbert_aug_file)
                with open(distilbert_aug_file, 'r', encoding='utf-8') as af:
                    ad = json.load(af)
                augmented_texts = ad.get('texts', [])
                augmented_tags = ad.get('tags', [])
            
            # Entrenar modelo (paso: distilbert train)
            if not state.get('distilbert_trained'):
                distilbert_metrics = distilbert_augmenter.train_model(
                    augmented_texts, augmented_tags,
                    batch_size=args.batch_size,
                    num_epochs=args.n_iter,
                    model_dir=os.path.join(args.output_dir, "distilbert_model")
                )
                state['distilbert_trained'] = True
                save_state(state)
            else:
                logger.info('Estado detectado: DistilBERT ya entrenado, saltando entrenamiento')
                distilbert_metrics = {}
            
            # Guardar métricas del entrenamiento
            if distilbert_metrics:
                if not state.get('distilbert_metrics_saved'):
                    distilbert_augmenter.save_metrics(distilbert_metrics, args.output_dir)
                    state['distilbert_metrics_saved'] = True
                    save_state(state)
                else:
                    logger.info('Estado detectado: métricas DistilBERT ya guardadas, saltando save_metrics')

                if not state.get('distilbert_plotted'):
                    distilbert_augmenter.plot_metrics(distilbert_metrics, args.output_dir)
                    state['distilbert_plotted'] = True
                    save_state(state)
                else:
                    logger.info('Estado detectado: métricas DistilBERT ya graficadas, saltando plot_metrics')
            
            logger.info("Modelo DistilBERT guardado en: %s", os.path.join(args.output_dir, 'distilbert_model'))
            logger.info("Modelo DistilBERT guardado")
            # Exponer el modelo para la demostración de complementación
            final_distilbert_model = distilbert_augmenter
            # Guardar modelo DistilBERT si procede
            if not state.get('distilbert_model_saved'):
                try:
                    # El trainer ya guarda best_model.pt en model_dir; marcar como guardado
                    state['distilbert_model_saved'] = True
                    save_state(state)
                except Exception:
                    pass
        if args.model_type in ['spacy', 'both']:
            logger.info("Entrenando modelo spaCy...")

            # Crear aumentador
            spacy_augmenter = SROIESpacyAugmenter(use_gpu=use_gpu)
            
            # Inicializar spaCy
            spacy_augmenter.initialize_spacy()
            
            # Cargar datos (con checkpoint)
            spacy_data_file = os.path.join(args.output_dir, 'spacy_loaded.json')
            if not state.get('spacy_data_loaded') or not os.path.exists(spacy_data_file):
                spacy_data = spacy_augmenter.load_data(args.data_dir)
                # serializar a formato JSON-friendly
                serial = []
                for text, ann in spacy_data:
                    ents = ann.get('entities', []) if isinstance(ann, dict) else []
                    serial.append({'text': text, 'entities': [[e[0], e[1], e[2]] for e in ents]})
                try:
                    with open(spacy_data_file, 'w', encoding='utf-8') as sf:
                        json.dump(serial, sf, ensure_ascii=False)
                    state['spacy_data_loaded'] = True
                    save_state(state)
                    logger.info('Datos spaCy cargados y guardados en checkpoint: %s', spacy_data_file)
                except Exception:
                    logger.exception('No se pudo guardar checkpoint de datos spaCy')
            else:
                logger.info('Cargando datos spaCy desde checkpoint: %s', spacy_data_file)
                with open(spacy_data_file, 'r', encoding='utf-8') as sf:
                    serial = json.load(sf)
                spacy_data = []
                for item in serial:
                    text = item.get('text', '')
                    ents = item.get('entities', [])
                    spacy_data.append((text, {'entities': [tuple(e) for e in ents]}))

            # Aumentar datos (con checkpoint por número de aumentaciones)
            spacy_aug_file = os.path.join(args.output_dir, f'spacy_augmented_{args.num_augmentations}.json')
            spacy_aug_key = f'spacy_augmented_{args.num_augmentations}'
            if not state.get(spacy_aug_key) or not os.path.exists(spacy_aug_file):
                augmented_data = spacy_augmenter.augment_data(
                    spacy_data, num_augmentations=args.num_augmentations
                )
                # serializar augmented_data
                serial_aug = []
                for text, ann in augmented_data:
                    ents = ann.get('entities', []) if isinstance(ann, dict) else []
                    serial_aug.append({'text': text, 'entities': [[e[0], e[1], e[2]] for e in ents]})
                try:
                    with open(spacy_aug_file, 'w', encoding='utf-8') as sf:
                        json.dump(serial_aug, sf, ensure_ascii=False)
                    state[spacy_aug_key] = True
                    save_state(state)
                    logger.info('Datos aumentados spaCy guardados en checkpoint: %s', spacy_aug_file)
                except Exception:
                    logger.exception('No se pudo guardar checkpoint de datos aumentados spaCy')
            else:
                logger.info('Cargando datos aumentados spaCy desde checkpoint: %s', spacy_aug_file)
                with open(spacy_aug_file, 'r', encoding='utf-8') as sf:
                    serial_aug = json.load(sf)
                augmented_data = []
                for item in serial_aug:
                    text = item.get('text', '')
                    ents = item.get('entities', [])
                    augmented_data.append((text, {'entities': [tuple(e) for e in ents]}))
            
            # Entrenar modelo (paso: spacy train)
            if not state.get('spacy_trained'):
                metrics = spacy_augmenter.train_model(
                    augmented_data,
                    n_iter=args.n_iter,
                    batch_size=args.batch_size,
                    model_dir=os.path.join(args.output_dir, "spacy_model")
                )
                state['spacy_trained'] = True
                save_state(state)
            else:
                logger.info('Estado detectado: spaCy ya entrenado, saltando entrenamiento')
                metrics = {}
            
            # Guardar métricas del entrenamiento
            if metrics:
                if not state.get('spacy_metrics_saved'):
                    spacy_augmenter.save_metrics(metrics, args.output_dir)
                    state['spacy_metrics_saved'] = True
                    save_state(state)
                else:
                    logger.info('Estado detectado: métricas spaCy ya guardadas, saltando save_metrics')

                if not state.get('spacy_plotted'):
                    spacy_augmenter.plot_metrics(metrics, args.output_dir)
                    state['spacy_plotted'] = True
                    save_state(state)
                else:
                    logger.info('Estado detectado: métricas spaCy ya graficadas, saltando plot_metrics')
            
            # Agregar post-procesamiento
            if not state.get('spacy_postproc_added'):
                spacy_augmenter.add_post_processing()
                state['spacy_postproc_added'] = True
                save_state(state)
            else:
                logger.info('Estado detectado: post-procesamiento spaCy ya añadido, saltando add_post_processing')
            
            # Guardar modelo final
            if not state.get('spacy_model_saved'):
                spacy_augmenter.nlp.to_disk(os.path.join(args.output_dir, "spacy_model"))
                state['spacy_model_saved'] = True
                save_state(state)
                logger.info("Modelo spaCy guardado en: %s", os.path.join(args.output_dir, 'spacy_model'))
                logger.info("Modelo spaCy guardado")
            else:
                logger.info('Estado detectado: spaCy ya guardado en disco, saltando to_disk')
            # Exponer el modelo para la demostración de complementación
            final_spacy_model = spacy_augmenter
    logger.info("Proceso completado.")

    # Si se entrenaron ambos modelos, realizar demostración de complementación
    try:
        if args.model_type == 'both' and 'final_distilbert_model' in locals() and 'final_spacy_model' in locals():
            logger.info("Realizando demostración de complementación entre DistilBERT y spaCy...")

            # Obtener la fuente de muestra (evaluator.spacy_data o spacy_data)
            sample_source = None
            if 'evaluator' in locals() and hasattr(evaluator, 'spacy_data'):
                sample_source = evaluator.spacy_data
            elif 'spacy_data' in locals():
                sample_source = spacy_data
            else:
                try:
                    tmp_eval = SROIEEvaluator(args.data_dir, use_gpu=use_gpu)
                    tmp_eval.load_data()
                    sample_source = tmp_eval.spacy_data
                except Exception:
                    sample_source = []

            if not sample_source:
                logger.warning("No hay datos de muestra disponibles para la demostración de complementación.")
                sample = []
                texts = []
            else:
                # Tomar una muestra pequeña de evaluación
                sample_size = min(10, len(sample_source))
                sample = sample_source[:sample_size]
                texts = [t for t, _ in sample]

            if texts:
                # Gold entities
                gold_list = [set([(s, e, lab) for s, e, lab in ann['entities']]) for _, ann in sample]

                # Predicciones spaCy
                spacy_preds_raw = final_spacy_model.predict(texts)
                spacy_preds = [set([(st, ed, lab) for _, st, ed, lab in doc_ents]) for doc_ents in spacy_preds_raw]

                # Predicciones DistilBERT
                distil_tags = final_distilbert_model.predict(texts)
                distil_preds = []
                for txt, tags in zip(texts, distil_tags):
                    ents = final_distilbert_model.convert_tags_to_entities(txt, tags)
                    # convert to (start,end,label)
                    ents_set = set([(s, e, lab) for _, s, e, lab in ents])
                    distil_preds.append(ents_set)

                # Métricas: función auxiliar
                def compute_metrics(gold_list, pred_list):
                    tp = fp = fn = 0
                    for gold, pred in zip(gold_list, pred_list):
                        tp += len(gold & pred)
                        fp += len(pred - gold)
                        fn += len(gold - pred)
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}

                spacy_metrics = compute_metrics(gold_list, spacy_preds)
                distil_metrics = compute_metrics(gold_list, distil_preds)

                # Unión simple de predicciones (complementación)
                union_preds = [s | d for s, d in zip(spacy_preds, distil_preds)]
                union_metrics = compute_metrics(gold_list, union_preds)

                report = {
                    'sample_size': sample_size,
                    'spacy': spacy_metrics,
                    'distilbert': distil_metrics,
                    'union': union_metrics
                }

                # Guardar informe
                report_path = os.path.join(args.output_dir, 'complementation_report.json')
                with open(report_path, 'w', encoding='utf-8') as rf:
                    json.dump(report, rf, indent=2, ensure_ascii=False)

                logger.info("Informe de complementación guardado en: %s", report_path)
    except Exception as e:
        logger.exception("Error durante la demostración de complementación: %s", e)
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Error al ejecutar sroie_main: %s", e)
        raise

