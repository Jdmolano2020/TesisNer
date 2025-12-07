"""
Integración de Técnicas de Aumentación de Datos en la Solución spaCy para SROIE

Este script implementa las modificaciones necesarias para integrar técnicas de
aumentación de datos en la solución basada en spaCy para el dataset SROIE.
""" 

import os
import random
import numpy as np
import pandas as pd
import spacy
from spacy.training import Example
from spacy.language import Language
from spacy.util import minibatch, compounding
from spacy.pipeline import EntityRuler
from sklearn.model_selection import KFold
from typing import List, Dict, Tuple, Any, Optional
import json
import re
import unicodedata
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from datetime import datetime

# Importar el aumentador de datos
from sroie_data_augmentation import SROIEDataAugmenter, Entity, Entities
from logging_config import get_logger

logger = get_logger(__name__)


@Language.component("sroie_post_process")
def sroie_post_process(doc):
    """
    Componente spaCy registrado para post-procesamiento de entidades.
    Aplica reglas simples para corregir fechas y totales.
    """
    new_ents = []

    for ent in doc.ents:
        # Regla 1: Corregir fechas mal formateadas
        if ent.label_ == "DATE":
            if re.match(r'\d{2}/\d{2}/\d{4}', ent.text) or re.match(r'\d{2}-\d{2}-\d{4}', ent.text):
                new_ents.append(ent)
            else:
                context = doc.text[max(0, ent.start_char - 20):min(len(doc.text), ent.end_char + 20)]
                date_match = re.search(r'\d{2}[/-]\d{2}[/-]\d{4}', context)
                if date_match:
                    start = doc.text.find(date_match.group(0))
                    end = start + len(date_match.group(0))
                    span = doc.char_span(start, end, label="DATE")
                    if span:
                        new_ents.append(span)
                else:
                    new_ents.append(ent)

        # Regla 2: Verificar totales con formato incorrecto
        elif ent.label_ == "TOTAL":
            if re.match(r'\$?\d+\.\d+', ent.text):
                new_ents.append(ent)
            else:
                context = doc.text[max(0, ent.start_char - 20):min(len(doc.text), ent.end_char + 20)]
                total_match = re.search(r'\$?\d+\.\d+', context)
                if total_match:
                    start = doc.text.find(total_match.group(0))
                    end = start + len(total_match.group(0))
                    span = doc.char_span(start, end, label="TOTAL")
                    if span:
                        new_ents.append(span)
                else:
                    new_ents.append(ent)

        else:
            new_ents.append(ent)

    doc.ents = new_ents
    return doc

# Configuración
random.seed(42)
np.random.seed(42)

def parse(line):
    fields = line.strip().split(",")
    if len(fields) == 9:
        return fields
    else:
        return fields[:8] + [",".join(fields[8:])]


def build_text(data):
    text = " ".join(data.text)
    text = text.replace("  "," ")
    return text

def normalize_text(text: str) -> str:
    """
    Normaliza texto: NFKC unicode, espacios múltiples, caracteres especiales.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def try_fix_entity_alignment(nlp, text: str, start: int, end: int, label: str) -> Optional[Tuple[int, int]]:
    """
    Intenta encontrar y alinear una entidad usando múltiples estrategias.
    Devuelve (new_start, new_end) si logra alinear, None si no puede.
    """
    if start < 0 or end > len(text) or start >= end:
        return None

    ent_text = text[start:end].strip()
    if not ent_text:
        return None

    cleaned_text = normalize_text(text)
    try:
        doc = nlp.make_doc(cleaned_text)
    except Exception:
        return None

    # Estrategia 1: char_span con alignment_mode contract (ajusta a tokens)
    for mode in ("contract", "expand"):
        try:
            span = doc.char_span(start, end, label=label, alignment_mode=mode)
            if span is not None:
                return span.start_char, span.end_char
        except Exception:
            pass

    # Estrategia 2: búsqueda exacta en texto normalizado
    pos = cleaned_text.find(ent_text)
    if pos != -1:
        return pos, pos + len(ent_text)

    # Estrategia 3: búsqueda de versión normalizada de la entidad
    ent_norm = normalize_text(ent_text)
    pos = cleaned_text.find(ent_norm)
    if pos != -1:
        return pos, pos + len(ent_norm)

    # Estrategia 4: búsqueda fuzzy (similitud con tokens)
    try:
        tokens = [t.text for t in doc]
        best = None
        for i in range(len(tokens)):
            for j in range(i, min(i + 10, len(tokens))):
                span = doc[i:j+1]
                joined_text = span.text
                similarity = SequenceMatcher(None, ent_text, joined_text).ratio()
                if best is None or similarity > best[0]:
                    best = (similarity, span.start_char, span.end_char)
        if best and best[0] > 0.7:
            return best[1], best[2]
    except Exception:
        pass

    return None

class SROIESpacyAugmenter:
    """Clase para integrar aumentación de datos en la solución spaCy para SROIE."""
    
    def __init__(self, use_gpu: bool = False):
        """
        Inicializa el aumentador para spaCy.
        
        Args:
            use_gpu: Si se debe usar GPU para el entrenamiento.
        """
        self.use_gpu = use_gpu
        self.data_augmenter = SROIEDataAugmenter(use_gpu=use_gpu)
        self.nlp = None
        self.entity_ruler = None
    
    def initialize_spacy(self, lang: str = "es"):
        """
        Inicializa el modelo spaCy.
        
        Args:
            lang: Código del idioma para el modelo base.
        """
        # Crear modelo base
        self.nlp = spacy.blank(lang)
        # No añadimos EntityRuler aquí para evitar crear el componente sin
        # patrones (que puede producir warnings). Lo añadiremos en
        # `train_model` cuando tengamos los `patterns` generados.
        self.entity_ruler = None
        self.ner = None
    
    def add_entity_patterns(self, patterns: List[Dict]):
        """
        Agrega patrones al EntityRuler para mejorar el reconocimiento.
        
        Args:
            patterns: Lista de patrones para el EntityRuler.
        """
        # Crear el EntityRuler si no existe (agregar por nombre y obtener el pipe)
        if self.entity_ruler is None:
            if "entity_ruler" not in self.nlp.pipe_names:
                # Añadimos el componente por su factory name
                self.nlp.add_pipe("entity_ruler")
            self.entity_ruler = self.nlp.get_pipe("entity_ruler")

        # Añadir patrones
        self.entity_ruler.add_patterns(patterns)
    
    def load_data(self, data_dir: str) -> List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]]:
        """
        Carga los datos del dataset SROIE en formato para spaCy.
        Valida y limpia alineaciones de entidades.
        
        Args:
            data_dir: Directorio con los archivos del dataset.
            
        Returns:
            Lista de tuplas (texto, anotaciones) en formato spaCy.
        """
        spacy_data = []
        # Carga de datos
        data_dir_texto = data_dir+"\\box"
        data_dir_tag = data_dir+"\\entities"
        text_files = [f for f in os.listdir(data_dir_texto) if f.endswith('.txt')]
        #text_files = text_files[:5] #para realizar pruebas con pocos archivos

        for text_file in text_files:
            # Cargar texto
            with open(os.path.join(data_dir_texto, text_file), 'r', encoding='utf-8') as f:
                text = f.readlines()
            data = pd.DataFrame(list(map(parse, text)), columns=[*(f"coor{i}" for i in range(8)), "text"])
            data = data.dropna()
            text = build_text(data)
            
            # Cargar etiquetas correspondientes
            tag_file = text_file
            if os.path.exists(os.path.join(data_dir_tag, tag_file)):
                with open(os.path.join(data_dir_tag, tag_file), 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                
                # Convertir anotaciones al formato de spaCy
                entities = []
                for entity_type, values in annotations.items():
                    for value in values:
                        # Buscar la posición de la entidad en el texto
                        start = text.find(value)
                        if start != -1:
                            end = start + len(value)
                            entities.append((start, end, entity_type))
                
                # Validar y fijar alineación de entidades antes de agregar
                cleaned_text, valid_entities = self._validate_and_fix_alignment(text, entities)
                
                if valid_entities:
                    spacy_data.append((cleaned_text, {"entities": valid_entities}))
                else:
                    logger.warning("Archivo %s no tiene entidades válidas tras limpieza", text_file)
        
        return spacy_data
    
    def convert_spacy_to_entities(self, spacy_data: List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]]) -> Tuple[List[str], List[Entities]]:
        """
        Convierte datos en formato spaCy a formato de entidades para el aumentador.
        
        Args:
            spacy_data: Lista de tuplas (texto, anotaciones) en formato spaCy.
            
        Returns:
            Tuple con listas de textos y entidades.
        """
        texts = []
        all_entities = []
        
        for text, annotations in spacy_data:
            entities = []
            for start, end, label in annotations["entities"]:
                entity_text = text[start:end]
                entities.append((entity_text, start, end, label))
            
            texts.append(text)
            all_entities.append(entities)
        
        return texts, all_entities
    
    def convert_entities_to_spacy(self, texts: List[str], all_entities: List[Entities]) -> List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]]:
        """
        Convierte datos en formato de entidades a formato spaCy.
        
        Args:
            texts: Lista de textos.
            all_entities: Lista de listas de entidades.
            
        Returns:
            Lista de tuplas (texto, anotaciones) en formato spaCy.
        """
        spacy_data = []
        
        for text, entities in zip(texts, all_entities):
            spacy_entities = []
            for entity_text, start, end, label in entities:
                spacy_entities.append((start, end, label))
            
            spacy_data.append((text, {"entities": spacy_entities}))
        
        return spacy_data
    
    def augment_data(self, spacy_data: List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]],
                    num_augmentations: int = 2) -> List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]]:
        """
        Aumenta los datos aplicando técnicas de aumentación.
        
        Args:
            spacy_data: Lista de tuplas (texto, anotaciones) en formato spaCy.
            num_augmentations: Número de versiones aumentadas a generar por texto.
            
        Returns:
            Lista aumentada de tuplas (texto, anotaciones) en formato spaCy.
        """
        # Convertir datos de spaCy a formato de entidades
        texts, all_entities = self.convert_spacy_to_entities(spacy_data)
        
        # Generar datos sintéticos
        synthetic_texts, synthetic_entities = self.data_augmenter.generate_synthetic_data(
            texts, all_entities, num_augmentations=num_augmentations,
            use_parallel=True,use_threads=True, num_workers=6
        )
        
        # Filtrar datos sintéticos de baja calidad
        filtered_texts, filtered_entities = self.data_augmenter.filter_synthetic_data(
            texts, all_entities, synthetic_texts, synthetic_entities
        )
        
        # Convertir datos sintéticos a formato spaCy
        synthetic_spacy_data = self.convert_entities_to_spacy(filtered_texts, filtered_entities)
        
        # Combinar datos originales y sintéticos
        augmented_spacy_data = spacy_data + synthetic_spacy_data
        
        return augmented_spacy_data
    
    def create_entity_patterns(self, spacy_data: List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]]) -> List[Dict]:
        """
        Crea patrones para el EntityRuler basados en los datos de entrenamiento.
        
        Args:
            spacy_data: Lista de tuplas (texto, anotaciones) en formato spaCy.
            
        Returns:
            Lista de patrones para el EntityRuler.
        """
        patterns = []
        entity_examples = {}
        
        # Recopilar ejemplos de entidades
        for text, annotations in spacy_data:
            for start, end, label in annotations["entities"]:
                entity_text = text[start:end]
                if label not in entity_examples:
                    entity_examples[label] = set()
                entity_examples[label].add(entity_text)
        
        # Crear patrones para cada tipo de entidad
        for label, examples in entity_examples.items():
            for example in examples:
                # Patrón exacto
                patterns.append({"label": label, "pattern": example})
                
                # Para fechas, agregar patrones de formato
                if label == "DATE":
                    # Detectar formato de fecha
                    if re.match(r'\d{2}/\d{2}/\d{4}', example):
                        patterns.append({"label": label, "pattern": [{"SHAPE": "dd/dd/dddd"}]})
                    elif re.match(r'\d{2}-\d{2}-\d{4}', example):
                        patterns.append({"label": label, "pattern": [{"SHAPE": "dd-dd-dddd"}]})
                
                # Para totales, agregar patrones de formato
                elif label == "TOTAL":
                    if re.match(r'\$\d+\.\d+', example):
                        patterns.append({"label": label, "pattern": [{"SHAPE": "$d+.d+"}]})
                    elif re.match(r'\d+\.\d+', example):
                        patterns.append({"label": label, "pattern": [{"SHAPE": "d+.d+"}]})
        
        return patterns
    
    def train_model(self, spacy_data: List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]],
                   n_iter: int = 100, batch_size: int = 16,
                   dropout: float = 0.2, use_cross_validation: bool = True,
                   model_dir: str = './models') -> Dict[str, Any]:
        """
        Entrena el modelo spaCy con los datos aumentados.
        
        Args:
            spacy_data: Lista de tuplas (texto, anotaciones) en formato spaCy.
            n_iter: Número de iteraciones de entrenamiento.
            batch_size: Tamaño del lote para entrenamiento.
            dropout: Tasa de dropout para regularización.
            use_cross_validation: Si se debe usar validación cruzada.
            model_dir: Directorio para guardar el modelo.
            
        Returns:
            Diccionario con métricas de entrenamiento.
        """
        if self.nlp is None:
            self.initialize_spacy()
        
        # Crear directorio para modelos si no existe
        os.makedirs(model_dir, exist_ok=True)
        
        # Crear patrones para el EntityRuler
        patterns = self.create_entity_patterns(spacy_data)
        # Añadir EntityRuler y sus patrones (se crea si falta)
        self.add_entity_patterns(patterns)

        # Asegurarnos de que el componente NER exista después del EntityRuler
        if "ner" not in self.nlp.pipe_names:
            self.ner = self.nlp.add_pipe("ner", after="entity_ruler")
        else:
            self.ner = self.nlp.get_pipe("ner")

        # Agregar etiquetas al componente NER
        for _, annotations in spacy_data:
            for _, _, label in annotations["entities"]:
                try:
                    self.ner.add_label(label)
                except Exception:
                    pass
        
        # Métricas de entrenamiento
        metrics = {
            'train_loss': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }
        
        if use_cross_validation:
            # Implementar validación cruzada
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_results = []
            
            # Convertir datos a formato de lista para KFold
            data_indices = list(range(len(spacy_data)))
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(data_indices)):
                logger.info("Entrenando fold %d/5...", fold+1)
                
                # Obtener datos de entrenamiento y validación para este fold
                fold_train_data = [spacy_data[i] for i in train_idx]
                fold_val_data = [spacy_data[i] for i in val_idx]
                
                # Reiniciar el modelo para este fold
                fold_nlp = spacy.blank(self.nlp.lang)
                # Añadir EntityRuler por factory name y obtener el pipe
                if "entity_ruler" not in fold_nlp.pipe_names:
                    fold_nlp.add_pipe("entity_ruler")
                fold_entity_ruler = fold_nlp.get_pipe("entity_ruler")
                fold_entity_ruler.add_patterns(patterns)
                # Añadir NER después del EntityRuler
                if "ner" not in fold_nlp.pipe_names:
                    fold_ner = fold_nlp.add_pipe("ner", after="entity_ruler")
                else:
                    fold_ner = fold_nlp.get_pipe("ner")
                
                # Agregar etiquetas
                for _, annotations in fold_train_data:
                    for _, _, label in annotations["entities"]:
                        fold_ner.add_label(label)
                
                # Entrenar
                fold_metrics = self._train_fold(fold_nlp, fold_train_data, fold_val_data, n_iter, batch_size, dropout)
                cv_results.append(fold_metrics['val_f1'][-1])
                
                # Guardar modelo de este fold
                fold_nlp.to_disk(os.path.join(model_dir, f"model_fold_{fold+1}"))
            
            # Calcular F1 promedio de validación cruzada
            avg_f1 = sum(cv_results) / len(cv_results)
            logger.info("F1 promedio en validación cruzada: %.4f", avg_f1)
            
            # Actualizar métricas
            metrics['cv_f1'] = avg_f1
        
        # Entrenar modelo final con datos divididos en entrenamiento y evaluación
        logger.info("Entrenando modelo final con todos los datos...")
        
        # Dividir datos en 80% entrenamiento y 20% evaluación
        split_idx = int(len(spacy_data) * 0.8)
        final_train_data = spacy_data[:split_idx]
        final_eval_data = spacy_data[split_idx:]
        
        logger.info("Datos de entrenamiento final: %d, Datos de evaluación: %d", len(final_train_data), len(final_eval_data))
        
        final_metrics = self._train_fold(self.nlp, final_train_data, final_eval_data, n_iter, batch_size, dropout)
        
        # Actualizar métricas
        metrics.update(final_metrics)
        
        # Guardar modelo final
        self.nlp.to_disk(os.path.join(model_dir, "final_model"))
        
        return metrics

    def _clean_entities(self, entities: List[Tuple[int, int, str]], text_len: Optional[int] = None) -> List[Tuple[int, int, str]]:
        """
        Limpia la lista de entidades eliminando duplicados exactos y resolviendo
        solapamientos. Se prefiere spans más largos cuando hay solapamiento.

        Args:
            entities: Lista de tuplas (start, end, label).
            text_len: Longitud del texto para validar límites (opcional).

        Returns:
            Lista filtrada de tuplas (start, end, label) sin solapamientos.
        """
        if not entities:
            return []

        # Filtrar spans inválidos y normalizar
        cleaned = []
        for start, end, label in entities:
            if start is None or end is None:
                continue
            if start < 0 or end <= start:
                continue
            if text_len is not None and end > text_len:
                continue
            cleaned.append((start, end, label))

        # Eliminar duplicados exactos (mismo start,end,label)
        unique = list(dict.fromkeys(cleaned))

        # Ordenar por start asc y length desc para preferir spans más largos
        unique.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        result = []
        occupied = []  # lista de (start,end) ya ocupados
        for start, end, label in unique:
            overlap = False
            for ostart, oend in occupied:
                # comprobar solapamiento
                if not (end <= ostart or start >= oend):
                    overlap = True
                    break
            if not overlap:
                result.append((start, end, label))
                occupied.append((start, end))

        if len(result) != len(entities):
            logger.debug("_clean_entities: reducidas %d -> %d entidades por duplicados/solapamientos", len(entities), len(result))

        return result

    def _validate_and_fix_alignment(self, text: str, entities: List[Tuple[int, int, str]]) -> Tuple[str, List[Tuple[int, int, str]]]:
        """
        Valida y corrige la alineación de entidades usando múltiples estrategias.
        Normaliza espacios, intenta realinear con char_span, búsqueda exacta y fuzzy.

        Args:
            text: Texto original.
            entities: Lista de tuplas (start, end, label).

        Returns:
            Tuple con (texto_limpiado, entidades_validadas).
        """
        # Normalizar espacios y caracteres especiales
        cleaned_text = normalize_text(text)

        valid_entities = []
        removed_count = 0
        recovered_count = 0

        for start, end, label in entities:
            if start is None or end is None or label is None:
                removed_count += 1
                continue

            # Caso 1: Índices inválidos (negativos o fuera de rango)
            if start < 0 or end > len(text) or start >= end:
                logger.warning("Entidad inválida (fuera de rango): start=%d, end=%d, len(text)=%d, label=%s. Intentando recuperar por búsqueda...", start, end, len(text), label)
                
                # Intentar recuperar la entidad por búsqueda de patrones
                # (asumiendo que 'start' y 'end' pueden ser erróneos pero 'label' es correcto)
                # Buscar una entidad de ese tipo que tenga sentido en el contexto
                
                # Si tenemos entity_text en la tupla original (raro), usarla
                # Si no, intentar patrones comunes basados en el label
                pattern_found = False
                
                # Patrones por tipo de entidad
                patterns = {
                    'address': r'[\w\s.,#\-\d]{5,}',  # Dirección: al menos 5 caracteres
                    'company': r'[A-Z][\w\s\.\&\-]{3,}',  # Empresa: comienza con mayúscula
                    'date': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # Fecha: formato xx/xx/xxxx
                    'total': r'\$?\d+[\.,]\d{2}',  # Total: número con decimales
                }
                
                pattern = patterns.get(label.lower())
                if pattern:
                    import re
                    matches = list(re.finditer(pattern, text))
                    if matches:
                        # Usar el primer match encontrado
                        match = matches[0]
                        new_start, new_end = match.start(), match.end()
                        entity_text = text[new_start:new_end]
                        valid_entities.append((new_start, new_end, label))
                        recovered_count += 1
                        pattern_found = True
                        logger.debug("Recuperada entidad '%s' (label=%s) por patrón: (%d,%d)", entity_text[:30], label, new_start, new_end)
                
                if not pattern_found:
                    logger.debug("No se pudo recuperar entidad con label=%s por búsqueda de patrones", label)
                    removed_count += 1
                continue

            try:
                entity_text = text[start:end].strip()

                if not entity_text:
                    logger.debug("Omitiendo entidad vacía: label=%s", label)
                    removed_count += 1
                    continue

                # Primero intentar alineación avanzada
                fixed_pos = try_fix_entity_alignment(self.nlp or spacy.blank('es'), text, start, end, label)

                if fixed_pos:
                    new_start, new_end = fixed_pos
                    valid_entities.append((new_start, new_end, label))
                    if (new_start, new_end) != (start, end):
                        recovered_count += 1
                        logger.debug("Recuperada entidad '%s' (label=%s): (%d,%d) -> (%d,%d)",
                                    entity_text[:30], label, start, end, new_start, new_end)
                    continue

                # Si alineación avanzada falla, intentar búsqueda simple
                found_pos = cleaned_text.find(entity_text)
                if found_pos == -1:
                    entity_normalized = re.sub(r'\s+', ' ', entity_text)
                    found_pos = cleaned_text.find(entity_normalized)

                if found_pos != -1:
                    new_start = found_pos
                    new_end = found_pos + len(entity_text)
                    if new_start >= 0 and new_end <= len(cleaned_text):
                        valid_entities.append((new_start, new_end, label))
                        recovered_count += 1
                        logger.debug("Realineada entidad '%s' (label=%s): búsqueda", entity_text[:30], label)
                        continue

                # No se pudo alinear
                logger.debug("No se pudo alinear entidad '%s' (label=%s)", entity_text[:50], label)
                removed_count += 1

            except Exception as e:
                logger.debug("Error al procesar entidad: %s", e)
                removed_count += 1

        if removed_count > 0 or recovered_count > 0:
            logger.info("_validate_and_fix_alignment: %d recuperadas, %d removidas (total: %d)",
                       recovered_count, removed_count, len(entities))


        return cleaned_text, valid_entities
    
    def _train_fold(self, nlp, train_data, val_data, n_iter, batch_size, dropout):
        """
        Entrena un fold del modelo.
        
        Args:
            nlp: Modelo spaCy a entrenar.
            train_data: Datos de entrenamiento.
            val_data: Datos de validación.
            n_iter: Número de iteraciones.
            batch_size: Tamaño del lote.
            dropout: Tasa de dropout.
            
        Returns:
            Diccionario con métricas de entrenamiento.
        """
        # Métricas de entrenamiento
        fold_metrics = {
            'train_loss': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }
        
        # Configurar optimizador
        optimizer = nlp.begin_training()
        
        # Variables para early stopping y timing
        best_val_f1 = 0
        patience = 3
        patience_counter = 0
        import time
        epoch_start_time = time.time()
        
        # Entrenar
        for epoch in range(n_iter):
            # Mezclar datos
            random.shuffle(train_data)
            
            # Crear lotes
            batches = minibatch(train_data, size=compounding(4.0, batch_size, 1.001))
            
            # Inicializar pérdida
            losses = {}
            
            # Entrenar en lotes con procesamiento paralelo de ejemplos
            from concurrent.futures import ThreadPoolExecutor
            def create_example(item):
                text, annotations = item
                # Primero validar y fijar alineación
                cleaned_text, aligned_entities = self._validate_and_fix_alignment(text, annotations.get("entities", []))
                # Luego limpiar duplicados/solapamientos
                final_entities = self._clean_entities(aligned_entities, text_len=len(cleaned_text))
                # Crear doc con el texto limpiado
                doc = nlp.make_doc(cleaned_text)
                cleaned_annotations = {"entities": final_entities}
                try:
                    example = Example.from_dict(doc, cleaned_annotations)
                    return example
                except Exception as e:
                    logger.debug("No se pudo crear Example: %s", e)
                    return None
            
            # Entrenar en lotes
            for batch in batches:
                # Crear ejemplos en paralelo (limitado a 2 threads para no sobrecargar)
                examples = []
                with ThreadPoolExecutor(max_workers=2) as executor:
                    results = executor.map(create_example, batch)
                    examples = [ex for ex in results if ex is not None]
                
                # Actualizar modelo
                nlp.update(examples, drop=dropout, losses=losses)
            
            # Registrar pérdida
            fold_metrics['train_loss'].append(losses.get("ner", 0.0))
            
            # Evaluar en conjunto de validación si está disponible
            if val_data:
                val_metrics = self.evaluate_model(nlp, val_data)
                fold_metrics['val_precision'].append(val_metrics['precision'])
                fold_metrics['val_recall'].append(val_metrics['recall'])
                fold_metrics['val_f1'].append(val_metrics['f1'])
                
                logger.info("Epoch %d/%d", epoch+1, n_iter)
                logger.info("Train Loss: %.4f", losses.get('ner', 0.0))
                logger.info("Val Precision: %.4f", val_metrics['precision'])
                logger.info("Val Recall: %.4f", val_metrics['recall'])
                logger.info("Val F1: %.4f", val_metrics['f1'])
                epoch_elapsed = time.time() - epoch_start_time
                logger.info("Tiempo de época: %.2f segundos", epoch_elapsed)
                epoch_start_time = time.time()
                
                # Early stopping
                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info("Early stopping activado después de %d épocas", epoch+1)
                        break
            else:
                logger.info("Epoch %d/%d", epoch+1, n_iter)
                logger.info("Train Loss: %.4f", losses.get('ner', 0.0))
        
        return fold_metrics
    
    def evaluate_model(self, nlp, eval_data):
        """
        Evalúa el modelo en un conjunto de datos.
        
        Args:
            nlp: Modelo spaCy a evaluar.
            eval_data: Datos de evaluación.
            
        Returns:
            Diccionario con métricas de evaluación.
        """
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives
        
        for text, annotations in eval_data:
            # Obtener predicciones
            doc = nlp(text)
            gold_entities = set([(start, end, label) for start, end, label in annotations["entities"]])
            pred_entities = set([(e.start_char, e.end_char, e.label_) for e in doc.ents])
            
            # Calcular métricas
            tp += len(gold_entities & pred_entities)
            fp += len(pred_entities - gold_entities)
            fn += len(gold_entities - pred_entities)
        
        # Calcular precision, recall y F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def add_post_processing(self):
        """
        Agrega componente de post-procesamiento al pipeline de spaCy.
        El componente está registrado globalmente como 'sroie_post_process'.
        """
        if "sroie_post_process" not in self.nlp.pipe_names:
            # Añadir usando el nombre registrado de la fábrica
            self.nlp.add_pipe('sroie_post_process', after='ner')
    
    def predict(self, texts: List[str]) -> List[List[Tuple[str, int, int, str]]]:
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            texts: Lista de textos para predecir.
            
        Returns:
            Lista de listas de entidades predichas.
        """
        if self.nlp is None:
            raise ValueError("El modelo no está cargado.")
        
        predictions = []
        
        for text in texts:
            doc = self.nlp(text)
            entities = [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
            predictions.append(entities)
        
        return predictions
    
    def save_metrics(self, metrics: Dict[str, Any], output_dir: str) -> str:
        """
        Guarda las métricas de entrenamiento en archivo JSON.
        
        Args:
            metrics: Diccionario con métricas de entrenamiento.
            output_dir: Directorio para guardar los resultados.
            
        Returns:
            Ruta del archivo de métricas guardado.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Crear nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(output_dir, f"metrics_{timestamp}.json")
        
        # Convertir listas numpy a listas Python si es necesario
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, list):
                metrics_serializable[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
            elif isinstance(value, (np.floating, np.integer)):
                metrics_serializable[key] = float(value)
            else:
                metrics_serializable[key] = value
        
        # Agregar información adicional
        metrics_serializable['timestamp'] = timestamp
        metrics_serializable['model_type'] = 'spacy'
        
        # Guardar JSON
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_serializable, f, indent=4, ensure_ascii=False)
        
        logger.info("Métricas guardadas en: %s", metrics_file)
        return metrics_file
    
    def plot_metrics(self, metrics: Dict[str, Any], output_dir: str) -> str:
        """
        Grafica las métricas de entrenamiento (F1 y pérdida).
        
        Args:
            metrics: Diccionario con métricas de entrenamiento.
            output_dir: Directorio para guardar los gráficos.
            
        Returns:
            Ruta del archivo de gráfico guardado.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(output_dir, f"training_metrics_{timestamp}.png")
        
        # Crear figura con subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfico de pérdida de entrenamiento
        if 'train_loss' in metrics and metrics['train_loss']:
            axes[0].plot(metrics['train_loss'], label='Train Loss', marker='o')
            axes[0].set_xlabel('Época')
            axes[0].set_ylabel('Pérdida')
            axes[0].set_title('Pérdida de Entrenamiento por Época')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Gráfico de F1 de validación
        if 'val_f1' in metrics and metrics['val_f1']:
            axes[1].plot(metrics['val_f1'], label='Validation F1', marker='s', color='green')
            axes[1].set_xlabel('Época')
            axes[1].set_ylabel('F1 Score')
            axes[1].set_title('F1 Score de Validación por Época')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # Si hay otras métricas de validación
        if 'val_precision' in metrics and metrics['val_precision']:
            axes[1].plot(metrics['val_precision'], label='Validation Precision', marker='^', color='orange')
        if 'val_recall' in metrics and metrics['val_recall']:
            axes[1].plot(metrics['val_recall'], label='Validation Recall', marker='d', color='red')
        
        if 'val_precision' in metrics or 'val_recall' in metrics:
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info("Gráfico de métricas guardado en: %s", plot_file)
        plt.close()
        
        return plot_file
    
    def plot_cv_results(self, cv_f1_scores: List[float], output_dir: str) -> str:
        """
        Grafica resultados de validación cruzada.
        
        Args:
            cv_f1_scores: Lista de scores F1 para cada fold.
            output_dir: Directorio para guardar el gráfico.
            
        Returns:
            Ruta del archivo de gráfico guardado.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(output_dir, f"cv_results_{timestamp}.png")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        folds = [f"Fold {i+1}" for i in range(len(cv_f1_scores))]
        ax.bar(folds, cv_f1_scores, color='steelblue', alpha=0.7)
        ax.axhline(y=np.mean(cv_f1_scores), color='red', linestyle='--', label=f'Promedio: {np.mean(cv_f1_scores):.4f}')
        ax.set_ylabel('F1 Score')
        ax.set_title('Resultados de Validación Cruzada (5-Fold)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Agregar valores en las barras
        for i, (fold, score) in enumerate(zip(folds, cv_f1_scores)):
            ax.text(i, score + 0.01, f'{score:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info("Gráfico de validación cruzada guardado en: %s", plot_file)
        plt.close()
        
        return plot_file


if __name__ == "__main__":
    try:
        # Ejemplo de datos
        spacy_data = [
            (
                "Factura emitida por Empresa ABC con fecha 01/01/2023 por un total de $1500.00",
                {"entities": [(19, 30, "COMPANY"), (41, 51, "DATE"), (67, 75, "TOTAL")]}
            ),
            (
                "Recibo de Tienda XYZ del 15/02/2023 con monto total $750.50",
                {"entities": [(10, 20, "COMPANY"), (25, 35, "DATE"), (53, 60, "TOTAL")]}
            )
        ]
        
        # Crear aumentador
        augmenter = SROIESpacyAugmenter(use_gpu=False)
        
        # Inicializar spaCy
        augmenter.initialize_spacy()
        
        # Aumentar datos
        augmented_data = augmenter.augment_data(spacy_data)
        
        logger.info("Datos originales: %d", len(spacy_data))
        logger.info("Datos aumentados: %d", len(augmented_data))
        
        # Entrenar modelo con datos aumentados
        metrics = augmenter.train_model(
            augmented_data,
            n_iter=30,
            batch_size=2
        )
        
        # Agregar post-procesamiento
        augmenter.add_post_processing()
        
        # Realizar predicciones
        test_texts = ["Factura de Empresa DEF del 10/03/2023 por $2000.00"]
        predictions = augmenter.predict(test_texts)
        
        logger.info("Predicciones:")
        for text, entities in zip(test_texts, predictions):
            logger.info("Texto: %s", text)
            for entity_text, start, end, label in entities:
                logger.info("  %s (%s): %d-%d", entity_text, label, start, end)
    except Exception as e:
        logger.exception("Error al ejecutar spacy_sroie_augmentation: %s", e)
        raise

