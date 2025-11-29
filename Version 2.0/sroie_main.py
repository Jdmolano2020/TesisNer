"""
Script Principal para Aumentación de Datos en SROIE

Este script proporciona una interfaz sencilla para aplicar técnicas de aumentación
de datos al dataset SROIE y mejorar la métrica F1 en el reconocimiento de entidades.
"""

import os
import argparse
import torch
import json
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
    
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Tamaño del lote para entrenamiento')
    
    parser.add_argument('--n_iter', type=int, default=50,
                       help='Número de iteraciones/épocas de entrenamiento')
    
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluar diferentes técnicas de aumentación')
    
    parser.add_argument('--use_gpu', action='store_true',
                       help='Usar GPU para el entrenamiento')
    
    return parser.parse_args()

def main():
    """
    Función principal.
    """
    # Parsear argumentos
    args = parse_args()
    
    # Crear directorios de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Verificar disponibilidad de GPU
    use_gpu = args.use_gpu and torch.cuda.is_available()
    if args.use_gpu and not torch.cuda.is_available():
        logger.info("ADVERTENCIA: GPU solicitada pero no disponible. Usando CPU.")
        print("ADVERTENCIA: GPU solicitada pero no disponible. Usando CPU.")
    
    # Guardar configuración
    config = vars(args)
    config['use_gpu'] = use_gpu
    
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Modo de evaluación
    if args.evaluate:
        logger.info("Modo de evaluación: comparando técnicas de aumentación de datos")
        print("Modo de evaluación: comparando técnicas de aumentación de datos")
        
        # Crear evaluador
        evaluator = SROIEEvaluator(args.data_dir, use_gpu=use_gpu)
        
        # Cargar datos
        evaluator.load_data()
        
        # Evaluar técnicas según el modelo seleccionado
        if args.model_type in ['distilbert', 'both']:
            print("\nEvaluando técnicas para DistilBERT...")
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
            final_distilbert_model = evaluator.train_best_model(
                model_type="distilbert",
                technique=best_distilbert_config["technique"],
                num_augmentations=best_distilbert_config["num_augmentations"],
                n_iter=args.n_iter * 2,  # Más iteraciones para el modelo final
                batch_size=args.batch_size
            )
            
            print(f"\nModelo DistilBERT final guardado en: {os.path.join(evaluator.results_dir, 'final_distilbert_model')}")
            logger.info("Modelo DistilBERT final guardado")
        if args.model_type in ['spacy', 'both']:
            print("\nEvaluando técnicas para spaCy...")
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
            final_spacy_model = evaluator.train_best_model(
                model_type="spacy",
                technique=best_spacy_config["technique"],
                num_augmentations=best_spacy_config["num_augmentations"],
                n_iter=args.n_iter * 2,  # Más iteraciones para el modelo final
                batch_size=args.batch_size
            )
            
            print(f"\nModelo spaCy final guardado en: {os.path.join(evaluator.results_dir, 'final_spacy_model')}")
            logger.info("Modelo spaCy final guardado...")
    
    # Modo de entrenamiento directo
    else:
        print(f"Entrenando modelo(s) con técnica: {args.technique}, aumentaciones: {args.num_augmentations}")
        logger.info("Entrenando modelo(s) con técnica: %s, aumentaciones: %d", args.technique, args.num_augmentations)
        # Entrenar según el modelo seleccionado
        if args.model_type in ['distilbert', 'both']:
            print("\nEntrenando modelo DistilBERT...")
            logger.info("Entrenando modelo DistilBERT...")
            
            # Crear aumentador
            distilbert_augmenter = SROIEDistilBERTAugmenter(use_gpu=use_gpu)
            
            # Cargar datos
            train_texts, train_tags = distilbert_augmenter.load_data(args.data_dir)
            
            # Aumentar datos
            augmented_texts, augmented_tags = distilbert_augmenter.augment_data(
                train_texts, train_tags, num_augmentations=args.num_augmentations
            )
            
            # Entrenar modelo
            distilbert_augmenter.train_model(
                augmented_texts, augmented_tags,
                batch_size=args.batch_size,
                num_epochs=args.n_iter,
                model_dir=os.path.join(args.output_dir, "distilbert_model")
            )
            
            print(f"\nModelo DistilBERT guardado en: {os.path.join(args.output_dir, 'distilbert_model')}")
            logger.info("Modelo DistilBERT guardado")
        if args.model_type in ['spacy', 'both']:
            print("\nEntrenando modelo spaCy...")
            logger.info("Entrenando modelo spaCy...")

            # Crear aumentador
            spacy_augmenter = SROIESpacyAugmenter(use_gpu=use_gpu)
            
            # Inicializar spaCy
            spacy_augmenter.initialize_spacy()
            
            # Cargar datos
            spacy_data = spacy_augmenter.load_data(args.data_dir)
            
            # Aumentar datos
            augmented_data = spacy_augmenter.augment_data(
                spacy_data, num_augmentations=args.num_augmentations
            )
            
            # Entrenar modelo
            spacy_augmenter.train_model(
                augmented_data,
                n_iter=args.n_iter,
                batch_size=args.batch_size,
                model_dir=os.path.join(args.output_dir, "spacy_model")
            )
            
            # Agregar post-procesamiento
            spacy_augmenter.add_post_processing()
            
            # Guardar modelo final
            spacy_augmenter.nlp.to_disk(os.path.join(args.output_dir, "spacy_model"))
            
            print(f"\nModelo spaCy guardado en: {os.path.join(args.output_dir, 'spacy_model')}")
            logger.info("Modelo spaCy guardado")
    print("\nProceso completado.")
    logger.info("Proceso completado.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Error al ejecutar sroie_main: %s", e)
        raise

