"""
Integración de Técnicas de Aumentación de Datos en la Solución DistilBERT para SROIE

Este script implementa las modificaciones necesarias para integrar técnicas de
aumentación de datos en la solución basada en DistilBERT para el dataset SROIE.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Any, Optional
import json
import re
import unicodedata
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from datetime import datetime
import random

# Importar el aumentador de datos
from sroie_data_augmentation import SROIEDataAugmenter, Entity, Entities
from logging_config import get_logger

logger = get_logger(__name__)

BASE_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")

def base_tokenize(text: str) -> List[str]:
    return BASE_TOKEN_PATTERN.findall(text)


def base_token_offsets(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in BASE_TOKEN_PATTERN.finditer(text)]

# Configuración
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

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


class SROIEDataset(Dataset):
    """Dataset personalizado para SROIE con DistilBERT."""
    
    def __init__(self, texts, tags, tokenizer, max_len=512):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2id = self._create_tag_map()
        self.id2tag = {v: k for k, v in self.tag2id.items()}
    
    def _create_tag_map(self):
        """Crea un mapeo de etiquetas a IDs."""
        unique_tags = sorted(list(set(tag for doc_tags in self.tags for tag in doc_tags)))
        return {tag: i for i, tag in enumerate(unique_tags)}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tags = self.tags[idx]
        
        # Tokenizar texto y alinear etiquetas
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # Eliminar la dimensión de lote
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Alinear etiquetas con tokens
        offset_mapping = encoding.pop('offset_mapping').numpy()
        label_ids = self._align_labels(tags, text, offset_mapping)
        
        # Convertir etiquetas a tensor
        encoding['labels'] = torch.tensor(label_ids)
        
        return encoding
    

    def _align_labels(self, labels, text, offset_mapping):
        """
        Alinea etiquetas BIO a nivel token base con los subtokens de DistilBERT.
        
        Args:
            labels (List[str]): BIO tags generados por _convert_to_bio_tags
            text (str): texto original
            offset_mapping (List[Tuple[int,int]]): offsets del tokenizer

        Returns:
            List[int]: label_ids alineados a subtokens
        """

        # Tokenización BASE (idéntica a _convert_to_bio_tags)
        base_offsets = base_token_offsets(text)
        base_tokens = [text[start:end] for start, end in base_offsets]

        if len(base_tokens) != len(labels):
            logger.warning(f"Mismatch base tokens vs labels: {len(base_tokens)} ≠ {len(labels)} for text: {text[:100]}...")
            # Retornar etiquetas ignoradas para evitar detener el entrenamiento
            return [-100] * len(offset_mapping)

        label_ids = []
        base_token_idx = 0
        previous_word_id = None

        for idx, (start, end) in enumerate(offset_mapping):

            # Tokens especiales ([CLS], [SEP], padding)
            if start == end == 0:
                label_ids.append(-100)
                continue

            # Avanzar el token base hasta que el offset actual pertenezca al token
            while base_token_idx < len(base_offsets) and not (
                base_offsets[base_token_idx][0] <= start < base_offsets[base_token_idx][1]
            ):
                base_token_idx += 1

            if base_token_idx >= len(labels):
                label_ids.append(-100)
                continue

            label = labels[base_token_idx]

            # Si es un subtoken (continuación)
            if base_token_idx == previous_word_id:
                if label.startswith("B-"):
                    label = label.replace("B-", "I-")

            label_ids.append(self.tag2id.get(label, self.tag2id.get("O", 0)))
            previous_word_id = base_token_idx

        return label_ids



class SROIEDistilBERTAugmenter:
    """Clase para integrar aumentación de datos en la solución DistilBERT para SROIE."""
    
    def __init__(self, use_gpu: bool = True):
        """
        Inicializa el aumentador para DistilBERT.
        
        Args:
            use_gpu: Si se debe usar GPU para el entrenamiento.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.data_augmenter = SROIEDataAugmenter(use_gpu=use_gpu)
        self.tokenizer = None
        self.model = None
        # NUEVO: Guardar los mapeos de manera global en la instancia
        self.tag2id = None
        self.id2tag = None
    
    def load_tokenizer(self, model_name: str = 'distilbert-base-multilingual-cased'):
        """
        Carga el tokenizador de DistilBERT.
        
        Args:
            model_name: Nombre del modelo DistilBERT a cargar.
        """
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    
    def load_model(self, num_labels: int, model_name: str = 'distilbert-base-multilingual-cased'):
        """
        Carga el modelo DistilBERT para clasificación de tokens.
        
        Args:
            num_labels: Número de etiquetas para clasificación.
            model_name: Nombre del modelo DistilBERT a cargar.
        """
        logger.info("Cargando modelo %s para clasificación de tokens con %d etiquetas...", model_name, num_labels)
        self.model = DistilBertForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)
        logger.info("Modelo cargado correctamente. Las capas de clasificación serán entrenadas desde cero.")


    def _convert_to_bio_tags(self, text: str, entities: Dict) -> List[str]:
        """
        Genera etiquetas BIO correctamente alineadas con los tokens.
        
        Args:
            text (str): Texto completo del documento
            entities (list): Lista de entidades con formato:
                [
                {"start": int, "end": int, "label": "COMPANY"},
                ...
                ]

        Returns:
            ListEtiquetas BIO alineadas token a token
        """

        # 1. Tokenización base (MISMA para todo el pipeline)
        tokens = base_tokenize(text)
        token_offsets = base_token_offsets(text)

        # 2. Crear mapa char → entidad
        char_labels = ["O"] * len(text)

        for ent in entities:
            start = ent["start"]
            end = ent["end"]
            label = ent["label"]

            char_labels[start] = f"B-{label}"
            for i in range(start + 1, end):
                char_labels[i] = f"I-{label}"

        # 3. Convertir char-level → token-level
        bio_tags = []
        cursor = 0

        for token, (token_start, token_end) in zip(tokens, token_offsets):
            # Extraer etiquetas del token
            token_chars = char_labels[token_start:token_end]

            if not token_chars or all(t == "O" for t in token_chars):
                bio_tags.append("O")
            else:
                # Priorizar B- si existe
                b_tags = [t for t in token_chars if t.startswith("B-")]
                if b_tags:
                    bio_tags.append(b_tags[0])
                else:
                    # tomar primer I-
                    i_tags = [t for t in token_chars if t.startswith("I-")]
                    bio_tags.append(i_tags[0])

            cursor = token_end

        # 4. Garantía final
        assert len(tokens) == len(bio_tags), (
            f"Desalineación interna: {len(tokens)} tokens vs {len(bio_tags)} tags"
        )

        return bio_tags
      
    
    def load_data(self, data_dir: str) -> Tuple[List[str], List[List[str]]]:
        
        """
        Carga los datos del dataset SROIE.
        
        Args:
            data_dir: Directorio con los archivos del dataset.
            
        Returns:
            Tuple con listas de textos y etiquetas.
        """
        
        logger.info("Inicio carga datos para DistilBERT...")
        texts = []
        all_tags = []
        
        # Implementar la carga de datos según el formato específico de SROIE
        
        # Carga de datos
        data_dir_texto = data_dir+"\\box"
        data_dir_tag = data_dir+"\\entities"
        text_files = [f for f in os.listdir(data_dir_texto) if f.endswith('.txt')]
        #text_files = text_files[:5] #para realizar pruebas con pocos archivos
        
        for text_file in text_files:
            # Cargar texto
            with open(os.path.join(data_dir_texto, text_file), 'r', encoding='utf-8', errors='ignore') as f:
                text = f.readlines()
            data = pd.DataFrame(list(map(parse, text)), columns=[*(f"coor{i}" for i in range(8)), "text"])
            data = data.dropna()
            #print("data",data)
            texto = build_text(data)
            
            # Cargar etiquetas correspondientes
            tag_file = text_file
            if os.path.exists(os.path.join(data_dir_tag, tag_file)):
                with open(os.path.join(data_dir_tag, tag_file), 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                # Convertir anotaciones a formato BIO
                entities = []
                for entity_type, values in annotations.items():
                    found_positions = set()  # Rastrear posiciones ya encontradas
                    value_stripped = values.strip()
                    if not value_stripped:
                        continue

                    # Buscar posición en el texto (solo la primera ocurrencia válida)
                    # Nota: La anotación original ya debería tener la posición correcta
                    start = texto.find(value_stripped)

                    if start != -1:
                        end = start + len(value_stripped)
                        # Evitar agregar duplicados exactos
                        pos_key = (start, end, entity_type)
                        if pos_key not in found_positions:
                            entities.append({'start': start, 'end': end, 'label': entity_type})
                            found_positions.add(pos_key)
                    else:
                        logger.debug("Entidad no encontrada en texto: '%s' (tipo=%s, archivo=%s)",
                                    value_stripped[:50], entity_type, text_file)
                tags = self._convert_to_bio_tags(texto, entities)
                
                texts.append(texto)
                all_tags.append(tags)
        logger.info("Fin carga datos para DistilBERT...,%d textos cargados, tags cargados % d", len(texts), len(all_tags))
        return texts, all_tags
    
    def convert_tags_to_entities(self, text: str, tags: List[str]) -> Entities:
        """
        Convierte etiquetas BIO a formato de entidades.
        
        Args:
            text: Texto original.
            tags: Lista de etiquetas BIO.
            
        Returns:
            Lista de entidades (texto, inicio, fin, tipo).
        """
        entities = []
        tokens = base_tokenize(text)
        token_offsets = base_token_offsets(text)
        
        if len(tokens) != len(tags):
            # Ajustar tags a la cantidad de tokens si hay desajuste leve.
            if len(tags) < len(tokens):
                tags = tags + ['O'] * (len(tokens) - len(tags))
            else:
                tags = tags[:len(tokens)]

        # Calcular posiciones de inicio de cada token
        token_positions = [start for start, _ in token_offsets]
        i = 0
        while i < len(tags):
            if tags[i].startswith('B-'):
                entity_type = tags[i][2:]
                start_idx = token_positions[i]
                entity_tokens = [tokens[i]]
                
                j = i + 1
                while j < len(tags) and tags[j].startswith('I-') and tags[j][2:] == entity_type:
                    entity_tokens.append(tokens[j])
                    j += 1
                
                entity_text = ' '.join(entity_tokens)
                end_idx = token_positions[i] + len(entity_text)
                
                entities.append((entity_text, start_idx, end_idx, entity_type))
                i = j
            else:
                i += 1
        
        return entities
    
    def convert_entities_to_tags(self, text: str, entities: Entities) -> List[str]:
        """
        Convierte entidades a etiquetas BIO con múltiples estrategias robustas.
        
        Args:
            text: Texto original.
            entities: Lista de entidades.
            
        Returns:
            Lista de etiquetas BIO.
        """
        tokens = base_tokenize(text)
        token_offsets = base_token_offsets(text)
        tags = ['O'] * len(tokens)

        if not tokens:
            return tags
        
        token_positions = [token_start for token_start, _ in token_offsets]
        
        def normalize_text(s: str) -> str:
            s = unicodedata.normalize('NFKC', s)
            s = s.lower().strip()
            s = re.sub(r"\s+", " ", s)
            return s
        
        # Procesar cada entidad
        for entity_text, start, end, entity_type in entities:
            entity_tokens = base_tokenize(entity_text)
            if not entity_tokens:
                continue

            token_index = None
            
            # Estrategia 1: coincidencia exacta de posición
            if isinstance(start, int) and start >= 0:
                for i, token_start in enumerate(token_positions):
                    if token_start == start:
                        token_index = i
                        break
            
            # Estrategia 2: búsqueda normalizada exacta
            if token_index is None:
                norm_entity = normalize_text(entity_text)
                best_ratio = 0.0
                best_index = None

                for i in range(len(tokens)):
                    if i + len(entity_tokens) > len(tokens):
                        continue
                    candidate = " ".join(tokens[i:i + len(entity_tokens)])
                    norm_candidate = normalize_text(candidate)
                    if norm_candidate == norm_entity:
                        token_index = i
                        break
                    ratio = SequenceMatcher(None, norm_candidate, norm_entity).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_index = i

                if token_index is None and best_ratio >= 0.5:
                    token_index = best_index

            # Estrategia 3: búsqueda en texto normalizado
            if token_index is None:
                norm_text = normalize_text(text)
                norm_ent = normalize_text(entity_text)
                matches = [(m.start(), m.end()) for m in re.finditer(re.escape(norm_ent), norm_text)]
                
                if matches:
                    # Usar la más cercana a start
                    if isinstance(start, int) and start >= 0:
                        best_match = min(matches, key=lambda m: abs(m[0] - start))
                        found_pos = best_match[0]
                    else:
                        found_pos = matches[0][0]
                    
                    for i, (token_start, token_end) in enumerate(token_offsets):
                        if token_start <= found_pos < token_end:
                            token_index = i
                            break

            # Estrategia 4: búsqueda case-insensitive simple
            if token_index is None:
                entity_lower = entity_text.lower().strip()
                text_lower = text.lower()
                idx = text_lower.find(entity_lower)
                if idx != -1:
                    for i, (token_start, token_end) in enumerate(token_offsets):
                        if token_start <= idx < token_end:
                            token_index = i
                            break

            # Estrategia 5: token más cercano
            if token_index is None and isinstance(start, int) and start >= 0:
                distances = [abs(tp - start) for tp in token_positions]
                if distances:
                    nearest_idx = min(range(len(distances)), key=lambda i: distances[i])
                    if distances[nearest_idx] <= 50:
                        token_index = nearest_idx

            if token_index is None:
                # Estrategia final: asignar al primer token 'O' disponible
                for i, tag in enumerate(tags):
                    if tag == 'O':
                        token_index = i
                        logger.debug("Usando estrategia final para entidad '%s' en token %d", 
                                   entity_text[:40], i)
                        break
                if token_index is None:
                    # Si no hay 'O', asignar al último token
                    token_index = len(tags) - 1
                    logger.debug("Asignando entidad '%s' al último token", entity_text[:40])

            # Asignar etiqueta B al primer token
            tags[token_index] = f'B-{entity_type}'
            
            # Asignar etiquetas I a tokens siguientes
            for j in range(1, len(entity_tokens)):
                if token_index + j < len(tags):
                    tags[token_index + j] = f'I-{entity_type}'
        
        return tags
    
    def augment_data(self, texts: List[str], all_tags: List[List[str]], 
                    num_augmentations: int = 2,
                    techniques: Optional[List[str]] = None,
                    rejected_dump_path: Optional[str] = None,
                    entity_preservation_threshold: float = 0.0,
                    diversity_threshold: float = 0.0) -> Tuple[List[str], List[List[str]]]:
        """
        Aumenta los datos aplicando técnicas de aumentación.
        Si entity_preservation_threshold==0.0, garantiza preservación exacta sin filtrado agresivo.
        
        Args:
            texts: Lista de textos originales.
            all_tags: Lista de listas de etiquetas.                    
            num_augmentations: Número de versiones aumentadas a generar por texto.
            rejected_dump_path: Ruta para volcar ejemplos sintéticos rechazados.
            entity_preservation_threshold: Umbral mínimo de preservación de entidades.
            diversity_threshold: Umbral mínimo de diversidad.
            
        Returns:
            Tuple con listas aumentadas de textos y etiquetas.
        """
        
        # Convertir etiquetas a formato de entidades
        all_entities = [
            self.convert_tags_to_entities(text, tags)
            for text, tags in zip(texts, all_tags)
        ]
        
        # Si los umbrales son 0.0, usar estrategia sin filtrado agresivo
        if entity_preservation_threshold == 0.0 and diversity_threshold == 0.0:
            logger.info("Augmentación con preservación exacta de entidades (sin filtrado agresivo)")
            
            # Generar datos sintéticos
            synthetic_texts, synthetic_entities, synthetic_meta = self.data_augmenter.generate_synthetic_data(
                texts, all_entities, techniques=techniques,
                num_augmentations=num_augmentations,
                use_parallel=True, use_threads=True, num_workers=6
            )
            
            # Garantizar preservación de entidades mediante reintentos
            filtered_texts = []
            filtered_entities = []
            entity_pool = self.data_augmenter.build_entity_pool(texts, all_entities)
            
            for i, (syn_text, syn_entities) in enumerate(zip(synthetic_texts, synthetic_entities)):
                orig_idx = i % len(texts)
                orig_text = texts[orig_idx]
                orig_entities = all_entities[orig_idx]
                
                # Validar preservación de tipos de entidad
                orig_labels = {label for _, _, _, label in orig_entities}
                syn_labels = {label for _, _, _, label in syn_entities}
                
                if orig_labels != syn_labels:
                    # Reintentar augmentation si falta algún tipo de entidad
                    technique = random.choice([
                        "back_translation", "ter", "cwr",
                        "back_translation+ter", "back_translation+cwr"
                    ])
                    aug_text, aug_entities, _ = self.data_augmenter.apply_combined_augmentation(
                        orig_text, orig_entities, entity_pool, technique
                    )
                    syn_text, syn_entities = self.data_augmenter._ensure_all_entity_labels(
                        orig_text, orig_entities, entity_pool, aug_text, aug_entities, technique
                    )
                
                filtered_texts.append(syn_text)
                filtered_entities.append(syn_entities)
        else:
            # Usar filtrado normal si hay umbrales no-cero
            synthetic_texts, synthetic_entities, synthetic_meta = self.data_augmenter.generate_synthetic_data(
                texts, all_entities, techniques=techniques,
                num_augmentations=num_augmentations,
                use_parallel=True, use_threads=True, num_workers=6
            )
            
            filtered_texts, filtered_entities = self.data_augmenter.filter_synthetic_data(
                texts, all_entities, synthetic_texts, synthetic_entities,
                synthetic_meta=synthetic_meta,
                entity_preservation_threshold=entity_preservation_threshold,
                diversity_threshold=diversity_threshold,
                rejected_dump_path=rejected_dump_path
            )

        # Convertir entidades a etiquetas
        synthetic_tags = [
            self.convert_entities_to_tags(text, entities)
            for text, entities in zip(filtered_texts, filtered_entities)
        ]
        
        # Verificar y regenerar si faltan entidades
        entity_pool = self.data_augmenter.build_entity_pool(texts, all_entities)
        for i, (syn_tags, syn_text, syn_entities) in enumerate(zip(synthetic_tags, filtered_texts, filtered_entities)):
            orig_idx = i % len(texts)
            orig_text = texts[orig_idx]
            orig_entities = all_entities[orig_idx]
            
            # Extraer entidades de las tags generadas
            syn_entities_extracted = self.convert_tags_to_entities(syn_text, syn_tags)
            orig_labels = {label for _, _, _, label in orig_entities}
            syn_labels = {label for _, _, _, label in syn_entities_extracted}
            
            if orig_labels != syn_labels:
                logger.warning(f"Ejemplo {i}: Faltan entidades {orig_labels - syn_labels}. Regenerando con TER.")
                # Regenerar con TER para asegurar preservación
                technique = "ter"
                aug_text, aug_entities, _ = self.data_augmenter.apply_combined_augmentation(
                    orig_text, orig_entities, entity_pool, technique
                )
                aug_text, aug_entities = self.data_augmenter._ensure_all_entity_labels(
                    orig_text, orig_entities, entity_pool, aug_text, aug_entities, technique
                )
                filtered_texts[i] = aug_text
                filtered_entities[i] = aug_entities
                synthetic_tags[i] = self.convert_entities_to_tags(aug_text, aug_entities)
        
        # Combinar datos originales y sintéticos
        augmented_texts = texts + filtered_texts
        augmented_tags = all_tags + synthetic_tags
        
        logger.info("Augmentación completada: %d originales + %d sintéticos = %d total",
                   len(texts), len(filtered_texts), len(augmented_texts))
        
        return augmented_texts, augmented_tags
    
    def train_model(self, train_texts: List[str], train_tags: List[List[str]],
                   val_texts: List[str] = None, val_tags: List[List[str]] = None,
                   batch_size: int = 8, num_epochs: int = 5,
                   learning_rate: float = 2e-5, use_class_weights: bool = True,
                   model_dir: str = './models') -> Dict[str, Any]:
        """
        Entrena el modelo DistilBERT con los datos aumentados.
        
        Args:
            train_texts: Lista de textos de entrenamiento.
            train_tags: Lista de listas de etiquetas de entrenamiento.
            val_texts: Lista de textos de validación.
            val_tags: Lista de listas de etiquetas de validación.
            batch_size: Tamaño del lote para entrenamiento.
            num_epochs: Número de épocas de entrenamiento.
            learning_rate: Tasa de aprendizaje.
            use_class_weights: Si se deben usar pesos de clase para manejar desbalance.
            model_dir: Directorio para guardar el modelo.
            
        Returns:
            Diccionario con métricas de entrenamiento.
        """

        if self.tokenizer is None:
            self.load_tokenizer()
        
        # Crear datasets
        train_dataset = SROIEDataset(train_texts, train_tags, self.tokenizer)
        # NUEVO: Conservar los mapeos reales detectados en el entrenamiento
        self.tag2id = train_dataset.tag2id
        self.id2tag = train_dataset.id2tag

        if val_texts is None or val_tags is None:
            # Dividir datos para validación si no se proporcionan
            train_texts, val_texts, train_tags, val_tags = train_test_split(
                train_texts, train_tags, test_size=0.1, random_state=42
            )
            train_dataset = SROIEDataset(train_texts, train_tags, self.tokenizer)
        
        val_dataset = SROIEDataset(val_texts, val_tags, self.tokenizer)
        
        # Crear dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Cargar modelo
        num_labels = len(train_dataset.tag2id)
        if self.model is None:
            self.load_model(num_labels)
        
        # Entrenamiento sin paralelismo para evitar overhead de memoria
        logger.info("Entrenamiento del modelo")
        
        # Configurar optimizador
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Configurar learning rate scheduler
        total_steps = len(train_dataloader) * num_epochs
        warmup_steps = max(1, int(total_steps * 0.1))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Configurar pesos de clase si es necesario
        if use_class_weights:
            # Aplanar todas las etiquetas
            all_labels = [tag for doc_tags in train_tags for tag in doc_tags]
            # Obtener etiquetas únicas como numpy.ndarray (requerido por sklearn)
            unique_labels = np.unique(all_labels)

            # Calcular pesos de clase (devuelve pesos en el mismo orden que 'unique_labels')
            class_weights = compute_class_weight('balanced', classes=unique_labels, y=all_labels)

            # Mapear label -> peso y luego a id de etiqueta
            class_weight_dict = {train_dataset.tag2id[label]: weight 
                                 for label, weight in zip(unique_labels.tolist(), class_weights)}
            
            # Convertir a tensor para PyTorch
            weights = torch.FloatTensor([class_weight_dict.get(i, 1.0) 
                                       for i in range(num_labels)]).to(self.device)
            
            # Modificar la función de pérdida para usar los pesos
            criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        o_label_id = train_dataset.tag2id.get('O', None)
        valid_eval_labels = [i for tag, i in train_dataset.tag2id.items() if tag != 'O']
        if not valid_eval_labels:
            valid_eval_labels = list(range(num_labels))
        
        # Variables para early stopping
        best_val_f1 = 0
        patience = 5
        patience_counter = 0
        best_model_path = os.path.join(model_dir, 'best_model.pt')
        
        # Crear directorio para modelos si no existe
        os.makedirs(model_dir, exist_ok=True)
        
        # Métricas de entrenamiento
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }
        
        # Entrenamiento
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            # Modo entrenamiento
            self.model.train()
            total_train_loss = 0
            
            try:
                for batch_idx, batch in enumerate(train_dataloader):
                    logger.info(f"Procesando batch {batch_idx+1}/{len(train_dataloader)}")
                    # Mover batch al dispositivo
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    try:
                        # Forward pass
                        outputs = self.model(**batch)
                        logits = outputs.logits
                        
                        # Calcular pérdida
                        if use_class_weights:
                            # Reshape para función de pérdida personalizada
                            active_loss = batch['attention_mask'].view(-1) == 1
                            active_logits = logits.view(-1, num_labels)
                            active_labels = torch.where(
                                active_loss,
                                batch['labels'].view(-1),
                                torch.tensor(-100).type_as(batch['labels'])
                            )
                            loss = criterion(active_logits, active_labels)
                        else:
                            loss = outputs.loss
                        
                        # Backward pass
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        # Actualizar parámetros
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        
                        total_train_loss += loss.item()
                        if (batch_idx + 1) % 10 == 0:
                            logger.info(f"Batch {batch_idx+1}/{len(train_dataloader)}, loss: {loss.item():.4f}")
                    except Exception as e:
                        logger.exception(f"Error en batch {batch_idx+1}: {e}")
                        raise
            except Exception as e:
                logger.exception(f"Error en epoch {epoch+1} durante entrenamiento: {e}")
                raise
            
            # Calcular pérdida promedio de entrenamiento
            avg_train_loss = total_train_loss / len(train_dataloader)
            metrics['train_loss'].append(avg_train_loss)
            
            # Modo evaluación
            self.model.eval()
            total_val_loss = 0
            val_predictions = []
            val_true_labels = []
            
            with torch.no_grad():
                for batch in val_dataloader:
                    # Mover batch al dispositivo
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    logits = outputs.logits
                    
                    # Calcular pérdida
                    if use_class_weights:
                        active_loss = batch['attention_mask'].view(-1) == 1
                        active_logits = logits.view(-1, num_labels)
                        active_labels = torch.where(
                            active_loss,
                            batch['labels'].view(-1),
                            torch.tensor(-100).type_as(batch['labels'])
                        )
                        loss = criterion(active_logits, active_labels)
                    else:
                        loss = outputs.loss
                    
                    total_val_loss += loss.item()
                    
                    # Obtener predicciones
                    predictions = torch.argmax(logits, dim=2)
                    
                    # Recopilar predicciones y etiquetas verdaderas
                    for i in range(predictions.shape[0]):
                        for j in range(predictions.shape[1]):
                            if batch['labels'][i, j] != -100:
                                val_predictions.append(predictions[i, j].item())
                                val_true_labels.append(batch['labels'][i, j].item())
            
            # Calcular métricas excluyendo la etiqueta O para evaluar entidades reales
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                val_true_labels, val_predictions, labels=valid_eval_labels, average='weighted', zero_division=0
            )
            
            # Calcular pérdida promedio de validación
            avg_val_loss = total_val_loss / len(val_dataloader)
            
            # Actualizar métricas
            metrics['val_loss'].append(avg_val_loss)
            metrics['val_precision'].append(val_precision)
            metrics['val_recall'].append(val_recall)
            metrics['val_f1'].append(val_f1)
            
            logger.info("Epoch %d/%d", epoch+1, num_epochs)
            logger.info("Train Loss: %.4f", avg_train_loss)
            logger.info("Val Loss: %.4f", avg_val_loss)
            logger.info("Val Precision: %.4f", val_precision)
            logger.info("Val Recall: %.4f", val_recall)
            logger.info("Val F1: %.4f", val_f1)
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                
                # Guardar el mejor modelo
                torch.save(self.model.state_dict(), best_model_path)
                logger.info("Nuevo mejor modelo guardado con F1: %.4f", val_f1)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping activado después de %d épocas", epoch+1)
                    break
        
        # Cargar el mejor modelo
        self.model.load_state_dict(torch.load(best_model_path))
        
        return metrics
    
    def predict(self, texts: List[str], batch_size: int = 8) -> List[List[str]]:
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            texts: Lista de textos para predecir.
            batch_size: Tamaño del lote para predicción.
            
        Returns:
            Lista de listas de etiquetas predichas.
        """
        # NUEVO: Validaciones de seguridad robustas
        if self.tokenizer is None:
            self.load_tokenizer()
        if self.model is None:
            raise ValueError("El modelo no está cargado. Llama a 'train_model' o a 'load_saved_model' primero.")
        if self.id2tag is None:
            raise ValueError("Mapeos de etiquetas ausentes. Pasa el 'tag2id' al cargar tu modelo.")
        
        # Crear dataset temporal para predicción (las etiquetas dummy ahora se ignoran con seguridad)
        dummy_tags = [['O'] * len(base_tokenize(text)) for text in texts]
        dataset = SROIEDataset(texts, dummy_tags, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        # Modo evaluación
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**{k: v for k, v in batch.items() if k != 'labels'})
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=2)
                
                for i in range(predictions.shape[0]):
                    pred_labels = []
                    for j in range(predictions.shape[1]):
                        if batch['attention_mask'][i, j] == 1 and j > 0:  # Ignorar [CLS]
                            pred_idx = predictions[i, j].item()
                            # CAMBIADO: Usar self.id2tag global en lugar de dataset.id2tag
                            if pred_idx in self.id2tag:
                                pred_labels.append(self.id2tag[pred_idx])
                    
                    text_tokens = texts[len(all_predictions)].split()
                    pred_labels = pred_labels[:len(text_tokens)]
                    if len(pred_labels) < len(text_tokens):
                        pred_labels.extend(['O'] * (len(text_tokens) - len(pred_labels)))
                    
                    all_predictions.append(pred_labels)
        
        return all_predictions
    
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
                metrics_serializable[key] = [float(v) if isinstance(v, (np.floating, np.integer, torch.Tensor)) else v for v in value]
            elif isinstance(value, (np.floating, np.integer, torch.Tensor)):
                metrics_serializable[key] = float(value)
            else:
                metrics_serializable[key] = value
        
        # Agregar información adicional
        metrics_serializable['timestamp'] = timestamp
        metrics_serializable['model_type'] = 'distilbert'
        
        # Guardar JSON
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_serializable, f, indent=4, ensure_ascii=False)
        
        logger.info("Métricas guardadas en: %s", metrics_file)
        return metrics_file
    
    def plot_metrics(self, metrics: Dict[str, Any], output_dir: str) -> str:
        """
        Grafica las métricas de entrenamiento (pérdida y métricas de validación).
        
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
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Gráfico 1: Pérdida de entrenamiento y validación
        if 'train_loss' in metrics and metrics['train_loss']:
            axes[0, 0].plot(metrics['train_loss'], label='Train Loss', marker='o', color='blue')
            axes[0, 0].set_xlabel('Época')
            axes[0, 0].set_ylabel('Pérdida')
            axes[0, 0].set_title('Pérdida de Entrenamiento')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Gráfico 2: Pérdida de validación
        if 'val_loss' in metrics and metrics['val_loss']:
            axes[0, 1].plot(metrics['val_loss'], label='Validation Loss', marker='s', color='red')
            axes[0, 1].set_xlabel('Época')
            axes[0, 1].set_ylabel('Pérdida')
            axes[0, 1].set_title('Pérdida de Validación')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Gráfico 3: Precisión, Recall y F1
        if 'val_precision' in metrics and metrics['val_precision']:
            axes[1, 0].plot(metrics['val_precision'], label='Precision', marker='^', color='orange')
            axes[1, 0].set_xlabel('Época')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('Precisión de Validación')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'val_recall' in metrics and metrics['val_recall']:
            axes[1, 1].plot(metrics['val_recall'], label='Recall', marker='d', color='green')
            if 'val_f1' in metrics and metrics['val_f1']:
                axes[1, 1].plot(metrics['val_f1'], label='F1', marker='s', color='purple')
            axes[1, 1].set_xlabel('Época')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Recall y F1 de Validación')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info("Gráfico de métricas guardado en: %s", plot_file)
        plt.close()
        
        return plot_file
    
    def load_saved_model(self, model_path: str, tag2id: Dict[str, int], model_name: str = 'distilbert-base-multilingual-cased'):
        """
        Carga un modelo previamente entrenado (.pt), su tokenizador y reconstruye 
        los mapas de etiquetas necesarios para la inferencia offline.
        """
        if self.tokenizer is None:
            self.load_tokenizer(model_name)
            
        self.tag2id = tag2id
        self.id2tag = {v: k for k, v in tag2id.items()}
        num_labels = len(tag2id)
        
        logger.info("Cargando arquitectura DistilBERT con %d etiquetas...", num_labels)
        self.model = DistilBertForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)
        
        logger.info("Cargando pesos (.pt) desde %s...", model_path)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        logger.info("Modelo y mapeos cargados exitosamente para inferencia.")


if __name__ == "__main__":
    try:
        # Ejemplo de datos
        train_texts = [
            "Factura emitida por Empresa ABC con fecha 01/01/2023 por un total de $1500.00",
            "Recibo de Tienda XYZ del 15/02/2023 con monto total $750.50"
        ]
        
        train_tags = [
            ['O', 'O', 'O', 'B-COMPANY', 'I-COMPANY', 'O', 'O', 'B-DATE', 'O', 'O', 'O', 'O', 'B-TOTAL'],
            ['O', 'O', 'B-COMPANY', 'I-COMPANY', 'O', 'B-DATE', 'O', 'O', 'O', 'B-TOTAL']
        ]
        
        # Crear aumentador
        augmenter = SROIEDistilBERTAugmenter(use_gpu=False)
        
        # Aumentar datos
        augmented_texts, augmented_tags = augmenter.augment_data(train_texts, train_tags)
        
        logger.info("Datos originales: %d", len(train_texts))
        logger.info("Datos aumentados: %d", len(augmented_texts))
        
        # Entrenar modelo con datos aumentados
        metrics = augmenter.train_model(
            augmented_texts, augmented_tags,
            batch_size=2, num_epochs=3
        )
        
        # Realizar predicciones
        test_texts = ["Factura de Empresa DEF del 10/03/2023 por $2000.00"]
        predictions = augmenter.predict(test_texts)
        
        logger.info("Predicciones:")
        for text, preds in zip(test_texts, predictions):
            tokens = text.split()
            logger.info("Texto: %s", text)
            for token, pred in zip(tokens, preds):
                logger.info("  %s: %s", token, pred)
    except Exception as e:
        logger.exception("Error al ejecutar distilbert_sroie_augmentation: %s", e)
        raise

