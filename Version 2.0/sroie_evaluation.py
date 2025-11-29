"""
Script de Evaluación y Ajuste para Técnicas de Aumentación de Datos en SROIE

Este script permite evaluar y comparar diferentes técnicas de aumentación de datos
para mejorar la métrica F1 en el reconocimiento de entidades en facturas SROIE.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Optional
import json
import time
import pickle
from logging_config import get_logger

logger = get_logger(__name__)

# Importar los módulos de aumentación de datos
from sroie_data_augmentation import SROIEDataAugmenter, Entity, Entities
from distilbert_sroie_augmentation import SROIEDistilBERTAugmenter
from spacy_sroie_augmentation import SROIESpacyAugmenter

# Configuración
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class SROIEEvaluator:
    """Clase para evaluar y ajustar técnicas de aumentación de datos para SROIE."""
    
    def __init__(self, data_dir: str, use_gpu: bool = True):
        """
        Inicializa el evaluador.
        
        Args:
            data_dir: Directorio con los datos de SROIE.
            use_gpu: Si se debe usar GPU para el entrenamiento.
        """
        self.data_dir = data_dir
        self.use_gpu = use_gpu
        self.results_dir = os.path.join(os.getcwd(), "evaluation_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Inicializar aumentadores
        self.data_augmenter = SROIEDataAugmenter(use_gpu=use_gpu)
        self.distilbert_augmenter = SROIEDistilBERTAugmenter(use_gpu=use_gpu)
        self.spacy_augmenter = SROIESpacyAugmenter(use_gpu=use_gpu)
    
    def load_data(self):
        """
        Carga los datos de SROIE para ambos modelos.
        """
        logger.info("Cargando datos para DistilBERT...")
        self.distilbert_texts, self.distilbert_tags = self.distilbert_augmenter.load_data(self.data_dir)
        
        logger.info("Cargando datos para spaCy...")
        self.spacy_data = self.spacy_augmenter.load_data(self.data_dir)
    
    def evaluate_augmentation_techniques(self, model_type: str = "distilbert",
                                        techniques: List[str] = None,
                                        num_augmentations: List[int] = None,
                                        n_iter: int = 30,
                                        batch_size: int = 8) -> Dict[str, Any]:
        """
        Evalúa diferentes técnicas de aumentación de datos.
        
        Args:
            model_type: Tipo de modelo a evaluar ("distilbert" o "spacy").
            techniques: Lista de técnicas a evaluar.
            num_augmentations: Lista de números de aumentaciones a probar.
            n_iter: Número de iteraciones de entrenamiento.
            batch_size: Tamaño del lote para entrenamiento.
            
        Returns:
            Diccionario con resultados de evaluación.
        """
        if techniques is None:
            techniques = ["original", "back_translation", "ter", "cwr", 
                         "back_translation+ter", "back_translation+cwr", "combined"]
        
        if num_augmentations is None:
            num_augmentations = [1, 2, 3]
        
        results = {}
        
        for technique in techniques:
            technique_results = {}
            
            for num_aug in num_augmentations:
                if technique == "original" and num_aug > 1:
                    continue  # Solo una versión para datos originales
                
                logger.info("Evaluando %s con técnica: %s, aumentaciones: %d", model_type, technique, num_aug)
                
                # Preparar datos según el modelo
                if model_type == "distilbert":
                    # Usar datos originales sin aumentación
                    if technique == "original":
                        train_texts = self.distilbert_texts
                        train_tags = self.distilbert_tags
                    else:
                        # Convertir etiquetas a entidades
                        all_entities = [
                            self.distilbert_augmenter.convert_tags_to_entities(text, tags)
                            for text, tags in zip(self.distilbert_texts, self.distilbert_tags)
                        ]
                        
                        # Generar datos sintéticos según la técnica
                        if technique == "combined":
                            # Usar todas las técnicas
                            synthetic_texts, synthetic_entities = self.data_augmenter.generate_synthetic_data(
                                self.distilbert_texts, all_entities, 
                                techniques=["back_translation", "ter", "cwr", 
                                          "back_translation+ter", "back_translation+cwr"],
                                num_augmentations=num_aug
                            )
                        else:
                            # Usar técnica específica
                            synthetic_texts, synthetic_entities = self.data_augmenter.generate_synthetic_data(
                                self.distilbert_texts, all_entities, 
                                techniques=[technique],
                                num_augmentations=num_aug
                            )
                        
                        # Filtrar datos sintéticos de baja calidad
                        filtered_texts, filtered_entities = self.data_augmenter.filter_synthetic_data(
                            self.distilbert_texts, all_entities, synthetic_texts, synthetic_entities
                        )
                        
                        # Convertir entidades a etiquetas
                        synthetic_tags = [
                            self.distilbert_augmenter.convert_entities_to_tags(text, entities)
                            for text, entities in zip(filtered_texts, filtered_entities)
                        ]
                        
                        # Combinar datos originales y sintéticos
                        train_texts = self.distilbert_texts + filtered_texts
                        train_tags = self.distilbert_tags + synthetic_tags
                    
                    # Entrenar y evaluar modelo DistilBERT
                    start_time = time.time()
                    
                    # Crear nueva instancia para cada evaluación
                    evaluator = SROIEDistilBERTAugmenter(use_gpu=self.use_gpu)
                    evaluator.load_tokenizer()
                    
                    # Dividir datos para validación
                    from sklearn.model_selection import train_test_split
                    train_texts, val_texts, train_tags, val_tags = train_test_split(
                        train_texts, train_tags, test_size=0.2, random_state=42
                    )
                    
                    # Entrenar modelo
                    metrics = evaluator.train_model(
                        train_texts, train_tags,
                        val_texts, val_tags,
                        batch_size=batch_size,
                        num_epochs=n_iter,
                        model_dir=os.path.join(self.results_dir, f"distilbert_{technique}_{num_aug}")
                    )
                    
                    training_time = time.time() - start_time
                    
                    # Guardar resultados
                    technique_results[num_aug] = {
                        'f1': max(metrics['val_f1']),
                        'precision': metrics['val_precision'][metrics['val_f1'].index(max(metrics['val_f1']))],
                        'recall': metrics['val_recall'][metrics['val_f1'].index(max(metrics['val_f1']))],
                        'training_time': training_time,
                        'num_examples': len(train_texts)
                    }
                
                elif model_type == "spacy":
                    # Inicializar spaCy
                    self.spacy_augmenter.initialize_spacy()
                    
                    # Usar datos originales sin aumentación
                    if technique == "original":
                        train_data = self.spacy_data
                    else:
                        # Convertir datos de spaCy a formato de entidades
                        texts, all_entities = self.spacy_augmenter.convert_spacy_to_entities(self.spacy_data)
                        
                        # Generar datos sintéticos según la técnica
                        if technique == "combined":
                            # Usar todas las técnicas
                            synthetic_texts, synthetic_entities = self.data_augmenter.generate_synthetic_data(
                                texts, all_entities, 
                                techniques=["back_translation", "ter", "cwr", 
                                          "back_translation+ter", "back_translation+cwr"],
                                num_augmentations=num_aug
                            )
                        else:
                            # Usar técnica específica
                            synthetic_texts, synthetic_entities = self.data_augmenter.generate_synthetic_data(
                                texts, all_entities, 
                                techniques=[technique],
                                num_augmentations=num_aug
                            )
                        
                        # Filtrar datos sintéticos de baja calidad
                        filtered_texts, filtered_entities = self.data_augmenter.filter_synthetic_data(
                            texts, all_entities, synthetic_texts, synthetic_entities
                        )
                        
                        # Convertir datos sintéticos a formato spaCy
                        synthetic_spacy_data = self.spacy_augmenter.convert_entities_to_spacy(
                            filtered_texts, filtered_entities
                        )
                        
                        # Combinar datos originales y sintéticos
                        train_data = self.spacy_data + synthetic_spacy_data
                    
                    # Entrenar y evaluar modelo spaCy
                    start_time = time.time()
                    
                    # Crear nueva instancia para cada evaluación
                    evaluator = SROIESpacyAugmenter(use_gpu=self.use_gpu)
                    evaluator.initialize_spacy()
                    
                    # Dividir datos para validación
                    from sklearn.model_selection import train_test_split
                    train_indices = list(range(len(train_data)))
                    train_idx, val_idx = train_test_split(train_indices, test_size=0.2, random_state=42)
                    
                    train_subset = [train_data[i] for i in train_idx]
                    val_subset = [train_data[i] for i in val_idx]
                    
                    # Entrenar modelo
                    metrics = evaluator.train_model(
                        train_subset,
                        n_iter=n_iter,
                        batch_size=batch_size,
                        use_cross_validation=False,
                        model_dir=os.path.join(self.results_dir, f"spacy_{technique}_{num_aug}")
                    )
                    
                    # Evaluar en conjunto de validación
                    val_metrics = evaluator.evaluate_model(evaluator.nlp, val_subset)
                    
                    training_time = time.time() - start_time
                    
                    # Guardar resultados
                    technique_results[num_aug] = {
                        'f1': val_metrics['f1'],
                        'precision': val_metrics['precision'],
                        'recall': val_metrics['recall'],
                        'training_time': training_time,
                        'num_examples': len(train_subset)
                    }
            
            results[technique] = technique_results
        
        # Guardar resultados
        results_file = os.path.join(self.results_dir, f"{model_type}_results.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        return results
    
    def visualize_results(self, results: Dict[str, Any], model_type: str):
        """
        Visualiza los resultados de la evaluación.
        
        Args:
            results: Diccionario con resultados de evaluación.
            model_type: Tipo de modelo evaluado.
        """
        # Preparar datos para visualización
        techniques = []
        num_augs = []
        f1_scores = []
        precisions = []
        recalls = []
        training_times = []
        num_examples = []
        
        for technique, technique_results in results.items():
            for num_aug, metrics in technique_results.items():
                techniques.append(technique)
                num_augs.append(num_aug)
                f1_scores.append(metrics['f1'])
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                training_times.append(metrics['training_time'])
                num_examples.append(metrics['num_examples'])
        
        # Crear DataFrame
        df = pd.DataFrame({
            'Técnica': techniques,
            'Aumentaciones': num_augs,
            'F1': f1_scores,
            'Precisión': precisions,
            'Recall': recalls,
            'Tiempo (s)': training_times,
            'Ejemplos': num_examples
        })
        
        # Guardar resultados en CSV
        csv_file = os.path.join(self.results_dir, f"{model_type}_results.csv")
        df.to_csv(csv_file, index=False)
        
        # Visualizar F1 por técnica y número de aumentaciones
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Técnica', y='F1', hue='Aumentaciones', data=df)
        plt.title(f'F1 por Técnica de Aumentación ({model_type})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"{model_type}_f1_by_technique.png"))
        
        # Visualizar precisión y recall
        plt.figure(figsize=(12, 8))
        df_melted = pd.melt(df, id_vars=['Técnica', 'Aumentaciones'], 
                           value_vars=['Precisión', 'Recall'],
                           var_name='Métrica', value_name='Valor')
        sns.barplot(x='Técnica', y='Valor', hue='Métrica', data=df_melted)
        plt.title(f'Precisión y Recall por Técnica de Aumentación ({model_type})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"{model_type}_precision_recall.png"))
        
        # Visualizar tiempo de entrenamiento
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Técnica', y='Tiempo (s)', hue='Aumentaciones', data=df)
        plt.title(f'Tiempo de Entrenamiento por Técnica de Aumentación ({model_type})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"{model_type}_training_time.png"))
        
        # Visualizar relación entre número de ejemplos y F1
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Ejemplos', y='F1', hue='Técnica', size='Aumentaciones', data=df, sizes=(50, 200))
        plt.title(f'Relación entre Número de Ejemplos y F1 ({model_type})')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"{model_type}_examples_vs_f1.png"))
        
        # Encontrar la mejor configuración
        best_idx = df['F1'].idxmax()
        best_technique = df.loc[best_idx, 'Técnica']
        best_num_aug = df.loc[best_idx, 'Aumentaciones']
        best_f1 = df.loc[best_idx, 'F1']
        
        logger.info("Mejor configuración para %s:", model_type)
        logger.info("Técnica: %s", best_technique)
        logger.info("Número de aumentaciones: %s", best_num_aug)
        logger.info("F1: %.4f", best_f1)
        
        # Guardar mejor configuración
        best_config = {
            'model_type': model_type,
            'technique': best_technique,
            'num_augmentations': best_num_aug,
            'f1': best_f1
        }
        
        with open(os.path.join(self.results_dir, f"{model_type}_best_config.json"), 'w') as f:
            json.dump(best_config, f, indent=4)
        
        return best_config
    
    def train_best_model(self, model_type: str, technique: str, num_augmentations: int,
                        n_iter: int = 100, batch_size: int = 16) -> Any:
        """
        Entrena el mejor modelo con la configuración óptima.
        
        Args:
            model_type: Tipo de modelo a entrenar.
            technique: Técnica de aumentación a utilizar.
            num_augmentations: Número de aumentaciones a generar.
            n_iter: Número de iteraciones de entrenamiento.
            batch_size: Tamaño del lote para entrenamiento.
            
        Returns:
            Modelo entrenado.
        """
        logger.info("Entrenando modelo final %s con técnica: %s, aumentaciones: %s", model_type, technique, num_augmentations)
        
        # Preparar datos según el modelo
        if model_type == "distilbert":
            # Usar datos originales sin aumentación
            if technique == "original":
                train_texts = self.distilbert_texts
                train_tags = self.distilbert_tags
            else:
                # Convertir etiquetas a entidades
                all_entities = [
                    self.distilbert_augmenter.convert_tags_to_entities(text, tags)
                    for text, tags in zip(self.distilbert_texts, self.distilbert_tags)
                ]
                
                # Generar datos sintéticos según la técnica
                if technique == "combined":
                    # Usar todas las técnicas
                    synthetic_texts, synthetic_entities = self.data_augmenter.generate_synthetic_data(
                        self.distilbert_texts, all_entities, 
                        techniques=["back_translation", "ter", "cwr", 
                                  "back_translation+ter", "back_translation+cwr"],
                        num_augmentations=num_augmentations
                    )
                else:
                    # Usar técnica específica
                    synthetic_texts, synthetic_entities = self.data_augmenter.generate_synthetic_data(
                        self.distilbert_texts, all_entities, 
                        techniques=[technique],
                        num_augmentations=num_augmentations
                    )
                
                # Filtrar datos sintéticos de baja calidad
                filtered_texts, filtered_entities = self.data_augmenter.filter_synthetic_data(
                    self.distilbert_texts, all_entities, synthetic_texts, synthetic_entities
                )
                
                # Convertir entidades a etiquetas
                synthetic_tags = [
                    self.distilbert_augmenter.convert_entities_to_tags(text, entities)
                    for text, entities in zip(filtered_texts, filtered_entities)
                ]
                
                # Combinar datos originales y sintéticos
                train_texts = self.distilbert_texts + filtered_texts
                train_tags = self.distilbert_tags + synthetic_tags
            
            # Crear nueva instancia para el modelo final
            final_model = SROIEDistilBERTAugmenter(use_gpu=self.use_gpu)
            final_model.load_tokenizer()
            
            # Entrenar modelo final
            metrics = final_model.train_model(
                train_texts, train_tags,
                batch_size=batch_size,
                num_epochs=n_iter,
                model_dir=os.path.join(self.results_dir, "final_distilbert_model")
            )
            
            return final_model
        
        elif model_type == "spacy":
            # Inicializar spaCy
            final_model = SROIESpacyAugmenter(use_gpu=self.use_gpu)
            final_model.initialize_spacy()
            
            # Usar datos originales sin aumentación
            if technique == "original":
                train_data = self.spacy_data
            else:
                # Convertir datos de spaCy a formato de entidades
                texts, all_entities = self.spacy_augmenter.convert_spacy_to_entities(self.spacy_data)
                
                # Generar datos sintéticos según la técnica
                if technique == "combined":
                    # Usar todas las técnicas
                    synthetic_texts, synthetic_entities = self.data_augmenter.generate_synthetic_data(
                        texts, all_entities, 
                        techniques=["back_translation", "ter", "cwr", 
                                  "back_translation+ter", "back_translation+cwr"],
                        num_augmentations=num_augmentations
                    )
                else:
                    # Usar técnica específica
                    synthetic_texts, synthetic_entities = self.data_augmenter.generate_synthetic_data(
                        texts, all_entities, 
                        techniques=[technique],
                        num_augmentations=num_augmentations
                    )
                
                # Filtrar datos sintéticos de baja calidad
                filtered_texts, filtered_entities = self.data_augmenter.filter_synthetic_data(
                    texts, all_entities, synthetic_texts, synthetic_entities
                )
                
                # Convertir datos sintéticos a formato spaCy
                synthetic_spacy_data = self.spacy_augmenter.convert_entities_to_spacy(
                    filtered_texts, filtered_entities
                )
                
                # Combinar datos originales y sintéticos
                train_data = self.spacy_data + synthetic_spacy_data
            
            # Entrenar modelo final
            metrics = final_model.train_model(
                train_data,
                n_iter=n_iter,
                batch_size=batch_size,
                use_cross_validation=True,
                model_dir=os.path.join(self.results_dir, "final_spacy_model")
            )
            
            # Agregar post-procesamiento
            final_model.add_post_processing()
            
            return final_model


# Ejemplo de uso
if __name__ == "__main__":
    try:
        # Directorio con los datos de SROIE
        data_dir = "./data/sroie"
        
        # Crear evaluador
        evaluator = SROIEEvaluator(data_dir, use_gpu=torch.cuda.is_available())
        
        # Cargar datos
        evaluator.load_data()
        
        # Evaluar técnicas para DistilBERT
        distilbert_results = evaluator.evaluate_augmentation_techniques(
            model_type="distilbert",
            techniques=["original", "back_translation", "ter", "cwr", "combined"],
            num_augmentations=[1, 2],
            n_iter=20,
            batch_size=8
        )
        
        # Visualizar resultados para DistilBERT
        best_distilbert_config = evaluator.visualize_results(distilbert_results, "distilbert")
        
        # Evaluar técnicas para spaCy
        spacy_results = evaluator.evaluate_augmentation_techniques(
            model_type="spacy",
            techniques=["original", "back_translation", "ter", "cwr", "combined"],
            num_augmentations=[1, 2],
            n_iter=30,
            batch_size=16
        )
        
        # Visualizar resultados para spaCy
        best_spacy_config = evaluator.visualize_results(spacy_results, "spacy")
        
        # Entrenar modelos finales con las mejores configuraciones
        final_distilbert_model = evaluator.train_best_model(
            model_type="distilbert",
            technique=best_distilbert_config["technique"],
            num_augmentations=best_distilbert_config["num_augmentations"],
            n_iter=100,
            batch_size=16
        )
        
        final_spacy_model = evaluator.train_best_model(
            model_type="spacy",
            technique=best_spacy_config["technique"],
            num_augmentations=best_spacy_config["num_augmentations"],
            n_iter=100,
            batch_size=16
        )
        
        logger.info("Entrenamiento completado. Los modelos finales están disponibles en:")
        logger.info("DistilBERT: %s", os.path.join(evaluator.results_dir, 'final_distilbert_model'))
        logger.info("spaCy: %s", os.path.join(evaluator.results_dir, 'final_spacy_model'))
    except Exception as e:
        logger.exception("Error al ejecutar sroie_evaluation: %s", e)
        raise

