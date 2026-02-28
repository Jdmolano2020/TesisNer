"""
Integración de Técnicas de Aumentación de Datos en la Solución spaCy para SROIE

Este script implementa las modificaciones necesarias para integrar técnicas de
aumentación de datos en la solución basada en spaCy para el dataset SROIE.
""" 

import os
import random
import warnings
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
import time

# Suprimir warning W036 de spaCy (entity_ruler sin patrones - es solo informativo)
warnings.filterwarnings("ignore", message=".*W036.*")

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
        Solo agrega el EntityRuler si hay patrones válidos.
        
        Args:
            patterns: Lista de patrones para el EntityRuler.
        """
        # Si no hay patrones, no agregar el EntityRuler para evitar W036
        if not patterns:
            logger.debug("Sin patrones para EntityRuler, saltando agregar reglas bases")
            return
        
        # Crear el EntityRuler si no existe (agregar por nombre y obtener el pipe)
        if self.entity_ruler is None:
            if "entity_ruler" not in self.nlp.pipe_names:
                # Añadimos el componente por su factory name
                self.nlp.add_pipe("entity_ruler")
            self.entity_ruler = self.nlp.get_pipe("entity_ruler")

        # Añadir patrones (solo si hay patrones válidos)
        if patterns:
            self.entity_ruler.add_patterns(patterns)
            logger.debug(f"Agregados {len(patterns)} patrones al EntityRuler")
    
    def load_data(self, data_dir: str) -> List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]]:
        """
        Carga los datos del dataset SROIE en formato para spaCy.
        Valida y limpia alineaciones de entidades evitando duplicaciones innecesarias.
        
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
                
                # Convertir anotaciones al formato de spaCy (usar tuplas para entidades)
                entities = []
                for entity_type, values in annotations.items():
                    found_positions = set()  # Rastrear posiciones ya encontradas
                    value_stripped = values.strip()
                    if not value_stripped:
                        continue

                    # Buscar posición en el texto (solo la primera ocurrencia válida)
                    # Nota: La anotación original ya debería tener la posición correcta
                    start = text.find(value_stripped)

                    if start != -1:
                        end = start + len(value_stripped)
                        # Evitar agregar duplicados exactos
                        pos_key = (start, end, entity_type)
                        if pos_key not in found_positions:
                            entities.append((start, end, entity_type))
                            found_positions.add(pos_key)
                    else:
                        logger.debug("Entidad no encontrada en texto: '%s' (tipo=%s, archivo=%s)",
                                    value_stripped[:50], entity_type, text_file)
                
                # Asegurar que las entidades sean tuplas (List[Tuple[int,int,str]])
                if entities:
                    entities = [tuple(ent) for ent in entities]
                    spacy_data.append((text, {"entities": entities}))
                    logger.info("Cargado archivo %s con %d entidades válidas", text_file, len(entities))
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
                    num_augmentations: int = 2, sample_fraction: float = 1.0,
                    rejected_dump_path: Optional[str] = None) -> List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]]:
        """
        Aumenta los datos aplicando técnicas de aumentación.
        
        Args:
            spacy_data: Lista de tuplas (texto, anotaciones) en formato spaCy.
            num_augmentations: Número de versiones aumentadas a generar por texto.
            sample_fraction: Fracción de los datos originales a usar como semilla para generar sintéticos (0.0-1.0).
            rejected_dump_path: Ruta opcional para volcar ejemplos rechazados (JSONL).
            
        Returns:
            Lista aumentada de tuplas (texto, anotaciones) en formato spaCy.
        """
        # Información inicial
        total_original = len(spacy_data)
        sample_fraction = float(sample_fraction) if sample_fraction is not None else 1.0
        if sample_fraction <= 0 or total_original == 0:
            logger.warning("spaCy: sample_fraction=%s no es válido o no hay datos; regresando datos originales sin aumentos", str(sample_fraction))
            return spacy_data

        # Determinar muestra a usar
        if sample_fraction >= 1.0:
            seed_data = spacy_data
            sample_size = total_original
        else:
            sample_size = max(1, int(round(total_original * sample_fraction)))
            # Mantener reproducibilidad
            random.seed(42)
            seed_data = random.sample(spacy_data, sample_size)

        logger.info("spaCy: datos cargados=%d, usando muestra=%d (%.1f%%) para generar aumentos (num_augmentations=%d)",
                    total_original, sample_size, sample_fraction * 100.0, num_augmentations)

        # Convertir datos de spaCy a formato de entidades (usando la muestra)
        texts, all_entities = self.convert_spacy_to_entities(seed_data)

        # Generar datos sintéticos
        synthetic_texts, synthetic_entities, synthetic_meta = self.data_augmenter.generate_synthetic_data(
            texts, all_entities, num_augmentations=num_augmentations,
            use_parallel=True, use_threads=True, num_workers=6
        )
        logger.info("spaCy: sintéticos generados=%d (antes del filtrado)", len(synthetic_texts))

        # Filtrar datos sintéticos de baja calidad
        if rejected_dump_path:
            logger.info("spaCy: los ejemplos rechazados se volcarán en: %s", rejected_dump_path)
        filtered_texts, filtered_entities = self.data_augmenter.filter_synthetic_data(
            texts, all_entities, synthetic_texts, synthetic_entities, synthetic_meta=synthetic_meta, rejected_dump_path=rejected_dump_path
        )
        logger.info("spaCy: sintéticos conservados tras filtrado=%d", len(filtered_texts))

        # Convertir datos sintéticos a formato spaCy
        synthetic_spacy_data = self.convert_entities_to_spacy(filtered_texts, filtered_entities)
        logger.info("spaCy: registros sintéticos convertidos a formato spaCy=%d", len(synthetic_spacy_data))

        # Combinar datos originales y sintéticos
        augmented_spacy_data = spacy_data + synthetic_spacy_data
        logger.info("spaCy: total registros tras aumentación=%d (originales=%d, sintéticos=%d)",
                    len(augmented_spacy_data), total_original, len(synthetic_spacy_data))

        return augmented_spacy_data
    
    def create_entity_patterns(self, spacy_data: List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]]) -> List[Dict]:
        """
        Crea patrones para el EntityRuler basados en los datos de entrenamiento.
        Retorna lista vacía si no hay datos o sin entidades.
        
        Args:
            spacy_data: Lista de tuplas (texto, anotaciones) en formato spaCy.
            
        Returns:
            Lista de patrones para el EntityRuler (puede estar vacía).
        """
        patterns = []
        entity_examples = {}
        
        # Validar que hay datos
        if not spacy_data:
            logger.debug("Sin datos de entrenamiento para crear patrones EntityRuler")
            return patterns
        
        # Recopilar ejemplos de entidades
        total_entities = 0
        for text, annotations in spacy_data:
            entities = annotations.get("entities", [])
            total_entities += len(entities)
            
            for start, end, label in entities:
                try:
                    entity_text = text[start:end]
                    if entity_text and not entity_text.isspace():
                        if label not in entity_examples:
                            entity_examples[label] = set()
                        entity_examples[label].add(entity_text)
                except Exception as e:
                    logger.debug(f"Error extrayendo patrón: {e}")
        
        if not entity_examples:
            logger.debug(f"Sin ejemplos de entidades trovados en {len(spacy_data)} muestras (total de ents: {total_entities})")
            return patterns
        
        # Crear patrones para cada tipo de entidad
        pattern_count = 0
        for label, examples in entity_examples.items():
            for example in examples:
                # Patrón exacto
                patterns.append({"label": label, "pattern": example})
                pattern_count += 1
                
                # Para fechas, agregar patrones de formato
                if label == "DATE":
                    # Detectar formato de fecha
                    if re.match(r'\d{2}/\d{2}/\d{4}', example):
                        patterns.append({"label": label, "pattern": [{"SHAPE": "dd/dd/dddd"}]})
                        pattern_count += 1
                    elif re.match(r'\d{2}-\d{2}-\d{4}', example):
                        patterns.append({"label": label, "pattern": [{"SHAPE": "dd-dd-dddd"}]})
                        pattern_count += 1
                
                # Para totales, agregar patrones de formato
                elif label == "TOTAL":
                    if re.match(r'\$\d+\.\d+', example):
                        patterns.append({"label": label, "pattern": [{"SHAPE": "$d+.d+"}]})
                        pattern_count += 1
                    elif re.match(r'\d+\.\d+', example):
                        patterns.append({"label": label, "pattern": [{"SHAPE": "d+.d+"}]})
                        pattern_count += 1
        
        logger.debug(f"Creados {pattern_count} patrones EntityRuler desde {len(entity_examples)} tipos de entidades")
        return patterns
    
    def train_model(self, spacy_data: List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]],
                   n_iter: int = 100, batch_size: int = 16,
                   dropout: float = 0.35, use_cross_validation: bool = True,
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
        
        # Validación y reparación previa (igual que antes)
        spacy_data, repair_stats = self.validate_and_repair_training_data(
            spacy_data, remove_invalid=True
        )
        if len(spacy_data) == 0:
            logger.error("Sin datos válidos tras reparación")
            return {}
        
        os.makedirs(model_dir, exist_ok=True)
        patterns = self.create_entity_patterns(spacy_data)
        if patterns:
            self.add_entity_patterns(patterns)
        
        if "ner" not in self.nlp.pipe_names:
            last_pipe = self.nlp.pipe_names[-1] if self.nlp.pipe_names else None
            self.ner = self.nlp.add_pipe("ner", last=True) if not last_pipe \
                    else self.nlp.add_pipe("ner", after=last_pipe)
        else:
            self.ner = self.nlp.get_pipe("ner")
        
        for _, annotations in spacy_data:
            for _, _, label in annotations["entities"]:
                self.ner.add_label(label)
        
        metrics = {'train_loss': [], 'val_precision': [], 'val_recall': [], 'val_f1': []}
        
        if use_cross_validation:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            data_indices = list(range(len(spacy_data)))
            cv_results = []
            
            # ── CLAVE: ruta del modelo "base" que se actualiza entre folds ──────
            # Fold 0: arranca desde self.nlp (blank)
            # Fold 1: arranca desde el mejor modelo del Fold 0
            # Fold 2: arranca desde el mejor modelo del Fold 1
            # ...así cada fold hereda el conocimiento del anterior
            
            base_model_dir = os.path.join(model_dir, "fold_base")
            
            # Guardar el modelo inicial (blank con labels) como punto de partida
            self.nlp.to_disk(base_model_dir)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(data_indices)):
                logger.info("═══ Fold %d/5 ═══", fold + 1)
                
                fold_train = [spacy_data[i] for i in train_idx]
                fold_val   = [spacy_data[i] for i in val_idx]
                
                # ── WARM-START: cargar el mejor modelo del fold anterior ─────────
                # En lugar de spacy.blank(), cargamos desde el checkpoint previo
                try:
                    fold_nlp = spacy.load(base_model_dir)
                    logger.info("Fold %d: warm-start desde %s", fold + 1, base_model_dir)
                except Exception as e:
                    logger.warning("No se pudo cargar warm-start (%s), iniciando desde blank", e)
                    fold_nlp = spacy.blank(self.nlp.lang)
                    # Re-añadir NER y labels si arranca desde blank
                    if "ner" not in fold_nlp.pipe_names:
                        fold_nlp.add_pipe("ner")
                    fold_ner = fold_nlp.get_pipe("ner")
                    for _, ann in fold_train:
                        for _, _, lbl in ann["entities"]:
                            fold_ner.add_label(lbl)
                
                # Ajustar n_iter para folds posteriores: menos épocas porque
                # ya viene pre-entrenado (fine-tuning, no entrenamiento desde cero)
                fold_n_iter = n_iter if fold == 0 else max(int(n_iter * 0.60), 20)
                
                logger.info(
                    "Fold %d: %d train, %d val, %d épocas",
                    fold + 1, len(fold_train), len(fold_val), fold_n_iter
                )
                
                fold_metrics = self._train_fold(
                    fold_nlp, fold_train, fold_val,
                    n_iter=fold_n_iter,
                    batch_size=batch_size,
                    dropout=dropout,
                    model_dir=model_dir,
                    model_name=f"fold_{fold + 1}"
                )
                
                best_fold_f1 = fold_metrics.get('best_val_f1', 0.0)
                cv_results.append(best_fold_f1)
                logger.info("Fold %d completado — Mejor F1: %.4f", fold + 1, best_fold_f1)
                
                # ── ACTUALIZAR BASE: el mejor modelo de este fold es el
                #    punto de partida del siguiente fold (transferencia progresiva)
                best_fold_path = fold_metrics.get('best_model_path')
                if best_fold_path and os.path.exists(best_fold_path):
                    # Solo actualizar la base si este fold mejoró respecto al anterior
                    prev_best = max(cv_results[:-1]) if len(cv_results) > 1 else 0.0
                    if best_fold_f1 >= prev_best:
                        import shutil
                        if os.path.exists(base_model_dir):
                            shutil.rmtree(base_model_dir)
                        shutil.copytree(best_fold_path, base_model_dir)
                        logger.info(
                            "Base actualizada con Fold %d (F1=%.4f > prev=%.4f)",
                            fold + 1, best_fold_f1, prev_best
                        )
                    else:
                        logger.info(
                            "Base NO actualizada: Fold %d (F1=%.4f) no mejoró prev=%.4f",
                            fold + 1, best_fold_f1, prev_best
                        )
            
            avg_f1 = sum(cv_results) / len(cv_results)
            logger.info("F1 promedio CV: %.4f | Por fold: %s", avg_f1, [f"{x:.4f}" for x in cv_results])
            metrics['cv_f1'] = avg_f1
            metrics['cv_f1_per_fold'] = cv_results
            
            # Para el entrenamiento final, partir del mejor modelo de CV (base_model_dir)
            try:
                self.nlp = spacy.load(base_model_dir)
                logger.info("Entrenamiento final con warm-start del mejor fold")
            except Exception:
                logger.warning("Usando self.nlp original para entrenamiento final")
        
        # Entrenamiento final (igual que antes, pero con warm-start)
        split_idx       = int(len(spacy_data) * 0.8)
        final_train     = spacy_data[:split_idx]
        final_eval      = spacy_data[split_idx:]
        
        final_metrics = self._train_fold(
            self.nlp, final_train, final_eval,
            n_iter=n_iter, batch_size=batch_size, dropout=dropout,
            model_dir=model_dir, model_name="final"
        )
        metrics.update(final_metrics)
        
        best_final_path = final_metrics.get('best_model_path')
        if best_final_path and os.path.exists(best_final_path):
            self.nlp = spacy.load(best_final_path)
        
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

        # Filtrar spans inválidos y normalizar (usar tuplas)
        cleaned = []
        for start, end, label in entities:
            if start is None or end is None:
                continue
            if not isinstance(start, int) or not isinstance(end, int):
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

    def _deduplicate_and_sort_entities(self, entities: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
        """
        Elimina entidades duplicadas y las ordena por posición de inicio.
        
        Args:
            entities: Lista de tuplas (start, end, label).
            
        Returns:
            Lista de entidades sin duplicados y ordenadas.
        """
        if not entities:
            return []
        
        # Eliminar duplicados exactos (misma posición e label)
        unique_entities = {}
        for start, end, label in entities:
            key = (start, end, label)
            unique_entities[key] = (start, end, label)
        
        # Convertir a lista y ordenar por posición de inicio
        result = list(unique_entities.values())
        result.sort(key=lambda x: (x[0], x[1]))  # Ordenar por start, luego por end
        
        if len(result) < len(entities):
            logger.debug("_deduplicate_and_sort_entities: eliminados %d duplicados, quedaron %d entidades",
                        len(entities) - len(result), len(result))
        
        return result

    def _validate_and_fix_alignment(self, text: str, entities: List[Tuple[int, int, str]]) -> Tuple[str, List[Tuple[int, int, str]]]:
        """
        Valida y corrige la alineación de entidades.
        IMPORTANTE: Trabaja de forma consistente usando solo el texto original.

        Args:
            text: Texto original.
            entities: Lista de tuplas (start, end, label) - índices referidos al texto original.

        Returns:
            Tuple con (texto_final, entidades_validadas).
        """
        # Primero validar todas las entidades contra el texto original
        initial_valid = []
        removed_before_truncate = 0
        
        for start, end, label in entities:
            # Validación básica de índices
            if start is None or end is None or label is None:
                removed_before_truncate += 1
                continue
            
            if not isinstance(start, int) or not isinstance(end, int):
                removed_before_truncate += 1
                continue
            
            # Validar rango
            if start < 0 or end < 0 or start >= end:
                logger.debug("Entidad inválida (índices negativos o invertidos): [%d:%d] %s", start, end, label)
                removed_before_truncate += 1
                continue
            
            if start > len(text) or end > len(text):
                logger.debug("Entidad fuera de rango: [%d:%d] (len=%d) %s", start, end, len(text), label)
                removed_before_truncate += 1
                continue
            
            # Extraer y validar contenido
            try:
                span_text = text[start:end]
                if not span_text or span_text.isspace():
                    removed_before_truncate += 1
                    continue
                
                initial_valid.append((start, end, label, span_text))
            except Exception as e:
                logger.debug("Error extrayendo span [%d:%d]: %s", start, end, e)
                removed_before_truncate += 1
        
        if removed_before_truncate > 0:
            logger.info("Removidas %d entidades inválidas en validación inicial", removed_before_truncate)
        
        # Normalizar espacios pero MANTENER longitudes iguales donde sea posible
        cleaned_text = normalize_text(text)
        
        # Si la limpieza cambió la longitud, intentar truncar a 512 tokens
        # max_tokens = 512
        # nlp_for_tokenization = self.nlp or spacy.blank('es')
        # doc = nlp_for_tokenization.make_doc(cleaned_text)
        # tokens = list(doc)
        
        # truncated = False
        # max_char_pos = len(cleaned_text)  # Por defecto, usar todo el texto limpiado
        
        # if len(tokens) > max_tokens:
        #     truncated = True
        #     tokens = tokens[:max_tokens]
        #     # Encontrar la posición del último token
        #     last_token = tokens[-1]
        #     max_char_pos = last_token.idx + len(last_token.text)
        #     cleaned_text_truncated = cleaned_text[:max_char_pos].rstrip()
        #     logger.info("Texto truncado a %d tokens (%d -> %d caracteres)", 
        #                max_tokens, len(cleaned_text), len(cleaned_text_truncated))
        #     cleaned_text = cleaned_text_truncated
        
        # Filtrar entidades que caben dentro del texto final
        valid_entities = []
        removed_after_truncate = 0
        
        for start, end, label, span_text in initial_valid:
            # Verificar que la entidad cabe dentro del texto truncado
            if end > len(cleaned_text):
                removed_after_truncate += 1
                logger.info("Entidad fuera de texto truncado: [%d:%d] vs len=%d", start, end, len(cleaned_text))
                continue
            
            # Verificar que el span aún es correcto en el texto limpiado
            try:
                cleaned_span = cleaned_text[start:end]
                # Hacer una comparación más flexible (ignorando espacios extras)
                if cleaned_span.strip() and (cleaned_span == span_text or cleaned_span.strip() == span_text.strip()):
                    valid_entities.append((start, end, label))
                else:
                    # El contenido cambió por limpieza - intentar encontrarlo
                    found = cleaned_text.find(span_text)
                    if found != -1:
                        valid_entities.append((found, found + len(span_text), label))
                        logger.info("Realineada entidad: [%d:%d] -> [%d:%d]", 
                                   start, end, found, found + len(span_text))
                    else:
                        # Búsqueda flexible
                        span_normalized = span_text.strip()
                        found = cleaned_text.find(span_normalized)
                        if found != -1:
                            valid_entities.append((found, found + len(span_normalized), label))
                        else:
                            removed_after_truncate += 1
                            logger.info("No se pudo realinear: '%s' (label=%s)", span_text[:30], label)
            except Exception as e:
                logger.info("Error en validación de entidad: %s", e)
                removed_after_truncate += 1
        
        if removed_after_truncate > 0:
            logger.info("Removidas %d entidades que no caben en texto truncado", removed_after_truncate)
        
        # Eliminar duplicados finales y ordenar
        if valid_entities:
            unique_entities = {}
            for ent in valid_entities:
                key = (ent[0], ent[1], ent[2])
                unique_entities[key] = ent
            valid_entities = sorted(unique_entities.values(), key=lambda x: (x[0], x[1]))
        
        return cleaned_text, valid_entities

    def validate_entity_alignment(self, text: str, entities: List[Tuple[int, int, str]]) -> Tuple[bool, List[str]]:
        """
        Valida si las entidades están correctamente alineadas con los tokens de spaCy.
        Usa offsets_to_biluo_tags() para detección rápida y char_span() para
        comprobación por entidad y diagnóstico.

        Args:
            text: Texto a validar.
            entities: Lista de tuplas (start, end, label).

        Returns:
            Tupla (is_valid, issues) donde is_valid es True si todas las entidades
            están bien alineadas; issues contiene mensajes con cuáles no se alinean.
        """
        from spacy.training import offsets_to_biluo_tags

        issues: List[str] = []

        # Validaciones básicas
        if not text or not entities:
            return True, []

        # Obtener un modelo spaCy para tokenizar
        try:
            test_nlp = self.nlp if self.nlp is not None else spacy.blank('es')
            doc = test_nlp.make_doc(text)

            # Primera aproximación: usar offsets_to_biluo_tags para detectar '-'
            try:
                biluo = offsets_to_biluo_tags(doc, entities)
                misaligned_count = sum(1 for t in biluo if t == '-')
                if misaligned_count == 0:
                    return True, []
                issues.append(f"{misaligned_count} entidades desalineadas (tags '-' encontrados)")
            except Exception:
                issues.append("offsets_to_biluo_tags falló; realizando comprobación per-entity")

            # Comprobación por entidad con char_span
            problematic_entities: List[Tuple[int, int, str]] = []
            for start, end, label in entities:
                if start < 0 or end < 0 or start >= end or start > len(text) or end > len(text):
                    issues.append(f"Índices inválidos: [{start}:{end}] label={label}")
                    problematic_entities.append((start, end, label))
                    continue

                span = None
                for mode in ("contract", "expand"):
                    try:
                        span = doc.char_span(start, end, label=label, alignment_mode=mode)
                        if span is not None:
                            break
                    except Exception:
                        span = None

                if span is None:
                    original_text = text[start:end]
                    issues.append(f"Entidad no alineada a tokens: [{start}:{end}] '{original_text}' label={label}")
                    problematic_entities.append((start, end, label))

            if problematic_entities or misaligned_count:
                return False, issues

            return True, []

        except Exception as e:
            issues.append(f"Error general en validación: {str(e)}")
            return False, issues

    def fix_misaligned_entities(self, text: str, entities: List[Tuple[int, int, str]], 
                               strict: bool = False) -> List[Tuple[int, int, str]]:
        """
        Corrige entidades desalineadas usando offsets_to_biluo_tags() para
        detección rápida y char_span(alignment_mode) para realizar la corrección.

        Args:
            text: Texto donde buscar las entidades.
            entities: Lista de tuplas (start, end, label).
            strict: Si True, elimina entidades que no se puedan alinear.
                   Si False, intenta búsquedas y normalizaciones adicionales.

        Returns:
            Lista de entidades corregidas (alineadas a tokens).
        """
        if not entities:
            return []

        nlp = self.nlp if self.nlp is not None else spacy.blank('es')
        doc = nlp.make_doc(text)

        # Intento rápido: si offsets_to_biluo_tags no reporta '-', no es necesario corregir
        try:
            from spacy.training import offsets_to_biluo_tags
            biluo = offsets_to_biluo_tags(doc, entities)
            if not any(t == '-' for t in biluo):
                return entities
        except Exception:
            # Si falla, seguimos al proceso por entidad
            pass

        fixed: List[Tuple[int, int, str]] = []

        for start, end, label in entities:
            # Validación básica
            if start is None or end is None or start < 0 or end <= start or start > len(text) or end > len(text):
                logger.info(f"Entidad ignorada (índices inválidos): [{start}:{end}] label={label}")
                continue

            original_span = text[start:end]
            if not original_span or original_span.isspace():
                logger.info(f"Entidad ignorada (vacía o solo espacios): [{start}:{end}]")
                continue

            # Estrategia 1: char_span contract -> expand
            aligned_span = None
            for mode in ("contract", "expand"):
                try:
                    span = doc.char_span(start, end, label=label, alignment_mode=mode)
                    if span is not None:
                        aligned_span = span
                        logger.info(f"Realineada [{start}:{end}] -> [{span.start_char}:{span.end_char}] usando mode={mode}")
                        break
                except Exception:
                    aligned_span = None

            if aligned_span is not None:
                fixed.append((aligned_span.start_char, aligned_span.end_char, label))
                continue

            # Estrategia 2: búsqueda exacta en el texto y alinear la posición encontrada
            found_pos = text.find(original_span)
            if found_pos != -1:
                try:
                    span = doc.char_span(found_pos, found_pos + len(original_span), label=label, alignment_mode="contract")
                    if span is not None:
                        fixed.append((span.start_char, span.end_char, label))
                        logger.info(f"Realineada por búsqueda exacta: [{start}:{end}] -> [{span.start_char}:{span.end_char}]")
                        continue
                except Exception:
                    pass

            # Estrategia 3: normalizar y buscar (si no strict)
            if not strict:
                norm = original_span.strip()
                found_pos = text.find(norm)
                if found_pos != -1:
                    try:
                        span = doc.char_span(found_pos, found_pos + len(norm), label=label, alignment_mode="contract")
                        if span is not None:
                            fixed.append((span.start_char, span.end_char, label))
                            logger.info(f"Realineada por normalización: [{start}:{end}] -> [{span.start_char}:{span.end_char}]")
                            continue
                    except Exception:
                        pass

            # Estrategia 4: búsqueda por tokens (subsecuencia) si no strict
            if not strict:
                try:
                    tokens = [t.text for t in doc]
                    seq = original_span.split()
                    for i in range(len(tokens) - len(seq) + 1):
                        window = " ".join(tokens[i:i+len(seq)])
                        if window == " ".join(seq):
                            start_char = doc[i].idx
                            end_char = doc[i+len(seq)-1].idx + len(doc[i+len(seq)-1].text)
                            span = doc.char_span(start_char, end_char, label=label, alignment_mode="contract")
                            if span is not None:
                                fixed.append((span.start_char, span.end_char, label))
                                break
                except Exception:
                    pass

            logger.info(f"Entidad descartada (no reparable): [{start}:{end}] '{original_span}' label={label}")

        # Deduplicar y ordenar
        if fixed:
            unique = {}
            for s, e, l in fixed:
                unique[(s, e, l)] = (s, e, l)
            fixed = sorted(unique.values(), key=lambda x: (x[0], x[1]))

        logger.info(f"fix_misaligned_entities: {len(entities)} -> {len(fixed)} entidades")
        return fixed

    def validate_and_repair_training_data(self, spacy_data: List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]],
                                         remove_invalid: bool = True) -> Tuple[List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]], Dict[str, Any]]:
        """
        Valida todos los datos de entrenamiento y repara entidades desalineadas.
        
        Args:
            spacy_data: Lista de tuplas (texto, anotaciones).
            remove_invalid: Si True, elimina ejemplos que no se pueden reparar.
            
        Returns:
            Tupla (datos_reparados, estadísticas_reparación).
        """
        repaired_data = []
        stats = {
            'total_samples': len(spacy_data),
            'valid_without_changes': 0,
            'repaired': 0,
            'removed_invalid': 0,
            'entities_fixed': 0,
            'entities_removed': 0,
            'sample_issues': []
        }
        
        for idx, (text, annotations) in enumerate(spacy_data):
            entities = annotations.get('entities', [])
            
            # Validar alineamiento actual
            is_valid, issues = self.validate_entity_alignment(text, entities)
            
            if is_valid:
                stats['valid_without_changes'] += 1
                repaired_data.append((text, annotations))
            else:
                # Intentar reparar
                fixed_entities = self.fix_misaligned_entities(text, entities, strict=False)
                
                if fixed_entities:
                    # Verificar que las entidades reparadas son válidas
                    is_valid_fixed, _ = self.validate_entity_alignment(text, fixed_entities)
                    
                    if is_valid_fixed:
                        stats['repaired'] += 1
                        stats['entities_fixed'] += len(fixed_entities) - len(entities)
                        repaired_data.append((text, {'entities': fixed_entities}))
                    elif remove_invalid:
                        stats['removed_invalid'] += 1
                        if len(stats['sample_issues']) < 5:  # Guardar solo 5 ejemplos
                            stats['sample_issues'].append({
                                'index': idx,
                                'text_sample': text[:100],
                                'issues': issues
                            })
                        logger.info(f"inválidos: {issues}")
                else:
                    if remove_invalid:
                        stats['removed_invalid'] += 1
                        if len(stats['sample_issues']) < 5:
                            stats['sample_issues'].append({
                                'index': idx,
                                'text_sample': text[:100],
                                'issues': issues
                            })
                        logger.info(f"inválidos: {issues}")
        
        logger.info(f"Validación de datos de entrenamiento completada:")
        logger.info(f"  Total: {stats['total_samples']}")
        logger.info(f"  Válidos sin cambios: {stats['valid_without_changes']}")
        logger.info(f"  Reparados: {stats['repaired']}")
        logger.info(f"  Eliminados (inválidos): {stats['removed_invalid']}")
        if stats['sample_issues']:
            logger.info(f"  Problemas encontrados (muestra):")
            for issue in stats['sample_issues']:
                logger.info(f"    - Índice {issue['index']}: {issue['issues']}")
        
        return repaired_data, stats

    def _train_fold(self, nlp, train_data, val_data, n_iter, batch_size, dropout, model_dir: Optional[str] = None, model_name: Optional[str] = None):
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
            'train_loss': [], 'val_precision': [],
            'val_recall': [], 'val_f1': []
        }
        
        optimizer = nlp.begin_training()
        best_val_f1      = 0.0
        best_model_path  = None
        
        # ── PATIENCE ADAPTATIVO ───────────────────────────────────────────────
        # Regla: patience = max(épocas_mínimas, n_iter * fracción)
        # 
        # Razonamiento:
        #   - Con n_iter=50:  patience = max(10, 50×0.20) = max(10,10) = 10
        #   - Con n_iter=100: patience = max(10, 100×0.20) = max(10,20) = 20
        #   - Con n_iter=200: patience = max(10, 200×0.20) = max(10,40) = 40
        #
        # La fracción 0.20 significa: "esperar al menos el 20% del total
        # de épocas sin mejora antes de parar"
        
        MIN_PATIENCE  = 10   # nunca menos de 10 épocas sin mejora
        PATIENCE_FRAC = 0.20 # 20% de n_iter como mínimo dinámico
        
        patience = max(MIN_PATIENCE, int(n_iter * PATIENCE_FRAC))
        logger.info("patience=%d (n_iter=%d, frac=%.0f%%)",
                    patience, n_iter, PATIENCE_FRAC * 100)
        
        patience_counter = 0
        
        # ── Historial de F1 para detección de plateau real ────────────────────
        # En lugar de comparar solo contra el máximo histórico (muy estricto),
        # usar una ventana deslizante para detectar mejora real
        f1_history    = []
        WINDOW_SIZE   = 5     # ventana para calcular tendencia
        MIN_DELTA     = 1e-4  # mejora mínima significativa (0.01%)
        
        # ── Cálculo dinámico de compound_factor ───────────────────────────────
        n_iter_to_max = max(1, int(n_iter * 0.30))
        if batch_size > 4:
            import math
            compound_factor = (batch_size / 4.0) ** (1.0 / n_iter_to_max)
        else:
            compound_factor = 1.0
        
        import time
        epoch_start = time.time()
        
        for epoch in range(n_iter):
            random.shuffle(train_data)
            
            batches = minibatch(
                train_data,
                size=compounding(4.0, float(batch_size), compound_factor)
            )
            
            losses = {}
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    try:
                        examples.append(Example.from_dict(doc, annotations))
                    except Exception as e:
                        logger.debug("Ejemplo inválido omitido: %s", e)
                if examples:
                    nlp.update(examples, drop=dropout, losses=losses)
            
            fold_metrics['train_loss'].append(losses.get("ner", 0.0))
            
            if val_data:
                val_m = self.evaluate_model(nlp, val_data)
                fold_metrics['val_precision'].append(val_m['precision'])
                fold_metrics['val_recall'].append(val_m['recall'])
                fold_metrics['val_f1'].append(val_m['f1'])
                
                current_f1 = val_m['f1']
                f1_history.append(current_f1)
                
                epoch_elapsed = time.time() - epoch_start
                epoch_start   = time.time()
                logger.info(
                    "Epoch %d/%d | Loss=%.4f | P=%.4f R=%.4f F1=%.4f | "
                    "patience=%d/%d | t=%.1fs",
                    epoch + 1, n_iter,
                    losses.get('ner', 0.0),
                    val_m['precision'], val_m['recall'], current_f1,
                    patience_counter, patience, epoch_elapsed
                )
                
                # ── CRITERIO DE MEJORA CON VENTANA DESLIZANTE ─────────────────
                # Comparar promedio de la ventana actual vs. ventana anterior
                # Esto filtra ruido y detecta tendencias reales
                
                if len(f1_history) >= WINDOW_SIZE * 2:
                    window_current  = f1_history[-WINDOW_SIZE:]
                    window_previous = f1_history[-WINDOW_SIZE * 2:-WINDOW_SIZE]
                    avg_current     = sum(window_current)  / WINDOW_SIZE
                    avg_previous    = sum(window_previous) / WINDOW_SIZE
                    trend_improving = (avg_current - avg_previous) > MIN_DELTA
                else:
                    # En las primeras épocas, usar comparación simple
                    trend_improving = current_f1 > (best_val_f1 + MIN_DELTA)
                
                # Guardar el mejor modelo absoluto (para recuperación)
                if current_f1 > best_val_f1 + MIN_DELTA:
                    best_val_f1      = current_f1
                    patience_counter = 0
                    
                    if model_dir and model_name:
                        os.makedirs(model_dir, exist_ok=True)
                        best_model_path = os.path.join(
                            model_dir, f"best_model_{model_name}"
                        )
                        nlp.to_disk(best_model_path)
                        fold_metrics['best_model_path'] = best_model_path
                        fold_metrics['best_val_f1']     = best_val_f1
                        logger.info("✓ Mejor F1=%.4f guardado en %s",
                                    best_val_f1, best_model_path)
                else:
                    # Solo incrementar patience si la TENDENCIA tampoco mejora
                    if not trend_improving:
                        patience_counter += 1
                    else:
                        # La tendencia sigue mejorando aunque el máximo no se rompió
                        # → reducir patience_counter (recuperación parcial)
                        patience_counter = max(0, patience_counter - 1)
                        logger.debug(
                            "Tendencia positiva, patience reducido a %d",
                            patience_counter
                        )
                    
                    if patience_counter >= patience:
                        logger.info(
                            "Early stopping en época %d | "
                            "Mejor F1=%.4f | sin mejora por %d épocas",
                            epoch + 1, best_val_f1, patience
                        )
                        break
            else:
                logger.info("Epoch %d/%d | Loss=%.4f",
                            epoch + 1, n_iter, losses.get('ner', 0.0))
        
        if best_model_path and 'best_model_path' not in fold_metrics:
            fold_metrics['best_model_path'] = best_model_path
            fold_metrics['best_val_f1']     = best_val_f1
        
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
            gold_entities = set([tuple(ent) for ent in annotations["entities"]])
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

