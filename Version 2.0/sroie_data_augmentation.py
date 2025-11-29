"""
Técnicas de Aumentación de Datos para NER en el Dataset SROIE

Este script implementa varias técnicas de aumentación de datos para mejorar
el rendimiento de modelos de reconocimiento de entidades nombradas (NER)
en el dataset SROIE de facturas escaneadas.

Técnicas implementadas:
1. Back Translation con preservación de entidades (sense-for-sense)
2. Targeted Entity Random Replacement (TER)
3. Contextual Word Replacement (CWR)
"""

import random
import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer, BertTokenizer, BertForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Any, Optional
from logging_config import get_logger

logger = get_logger(__name__)

# Configuración
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Definición de tipos para claridad
Entity = Tuple[str, int, int, str]  # (texto, inicio, fin, tipo)
Entities = List[Entity]
Text = str

class SROIEDataAugmenter:
    """Clase para aumentación de datos en el dataset SROIE."""
    
    def __init__(self, use_gpu: bool = False):
        """
        Inicializa el aumentador de datos.
        
        Args:
            use_gpu: Si se debe usar GPU para los modelos de transformers.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.translation_models = {}
        self.bert_model = None
        self.bert_tokenizer = None
        
    def load_translation_models(self, source_lang: str = 'es', target_lang: str = 'en'):
        """
        Carga los modelos de traducción para back translation.
        
        Args:
            source_lang: Código del idioma fuente.
            target_lang: Código del idioma objetivo.
        """
        logger.info("Cargando modelos de traducción %s-%s...", source_lang, target_lang)
        
        # Modelo de source_lang a target_lang
        model_name_src_tgt = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
        tokenizer_src_tgt = MarianTokenizer.from_pretrained(model_name_src_tgt)
        model_src_tgt = MarianMTModel.from_pretrained(model_name_src_tgt).to(self.device)
        
        # Modelo de target_lang a source_lang
        model_name_tgt_src = f'Helsinki-NLP/opus-mt-{target_lang}-{source_lang}'
        tokenizer_tgt_src = MarianTokenizer.from_pretrained(model_name_tgt_src)
        model_tgt_src = MarianMTModel.from_pretrained(model_name_tgt_src).to(self.device)
        
        self.translation_models = {
            f'{source_lang}-{target_lang}': {
                'tokenizer': tokenizer_src_tgt,
                'model': model_src_tgt
            },
            f'{target_lang}-{source_lang}': {
                'tokenizer': tokenizer_tgt_src,
                'model': model_tgt_src
            }
        }
        
        logger.info("Modelos de traducción cargados correctamente.")
    
    def load_bert_model(self, model_name: str = 'dccuchile/bert-base-spanish-wwm-cased'):
        """
        Carga el modelo BERT para reemplazo contextual de palabras.
        
        Args:
            model_name: Nombre del modelo BERT a cargar.
        """
        logger.info("Cargando modelo BERT: %s...", model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertForMaskedLM.from_pretrained(model_name).to(self.device)
        logger.info("Modelo BERT cargado correctamente.")
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Traduce un texto de un idioma a otro.
        
        Args:
            text: Texto a traducir.
            source_lang: Código del idioma fuente.
            target_lang: Código del idioma objetivo.
            
        Returns:
            Texto traducido.
        """
        model_key = f'{source_lang}-{target_lang}'
        if model_key not in self.translation_models:
            raise ValueError(f"Modelo de traducción {model_key} no cargado.")
        
        tokenizer = self.translation_models[model_key]['tokenizer']
        model = self.translation_models[model_key]['model']
        
        # Tokenizar y traducir
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        outputs = model.generate(**inputs)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translated_text
    
    def back_translate(self, text: str, entities: Entities, 
                       source_lang: str = 'es', target_lang: str = 'en') -> Tuple[str, Entities]:
        """
        Implementa back translation con preservación de entidades.
        
        Args:
            text: Texto original.
            entities: Lista de entidades en el texto.
            source_lang: Código del idioma fuente.
            target_lang: Código del idioma objetivo.
            
        Returns:
            Tuple con el texto traducido y las entidades actualizadas.
        """
        if not self.translation_models:
            self.load_translation_models(source_lang, target_lang)
        
        # 1. Aplicar máscaras a las entidades
        masked_text = text
        entity_map = {}
        
        # Ordenar entidades por posición de inicio (de mayor a menor)
        # para evitar problemas con reemplazos que se solapan
        sorted_entities = sorted(entities, key=lambda e: e[1], reverse=True)
        
        for i, (entity_text, start, end, entity_type) in enumerate(sorted_entities):
            mask = f"ENTITY_{i}_{entity_type}"
            masked_text = masked_text[:start] + mask + masked_text[end:]
            entity_map[mask] = (entity_text, start, end, entity_type)
        
        # 2. Traducir al idioma objetivo
        translated_text = self.translate(masked_text, source_lang, target_lang)
        
        # 3. Traducir de vuelta al idioma original
        back_translated_text = self.translate(translated_text, target_lang, source_lang)
        
        # 4. Reinsertar las entidades originales y recalcular posiciones
        new_entities = []
        final_text = back_translated_text
        
        # Buscar las máscaras en el texto traducido y reemplazarlas
        for mask, (entity_text, _, _, entity_type) in entity_map.items():
            if mask in final_text:
                start_idx = final_text.find(mask)
                end_idx = start_idx + len(entity_text)
                final_text = final_text.replace(mask, entity_text, 1)
                new_entities.append((entity_text, start_idx, end_idx, entity_type))
        
        # Ordenar entidades por posición de inicio
        new_entities = sorted(new_entities, key=lambda e: e[1])
        
        return final_text, new_entities
    
    def build_entity_pool(self, texts: List[str], all_entities: List[Entities]) -> Dict[str, List[str]]:
        """
        Construye un pool de entidades agrupadas por tipo.
        
        Args:
            texts: Lista de textos.
            all_entities: Lista de listas de entidades para cada texto.
            
        Returns:
            Diccionario con entidades agrupadas por tipo.
        """
        entity_pool = {}
        
        for entities in all_entities:
            for entity_text, _, _, entity_type in entities:
                if entity_type not in entity_pool:
                    entity_pool[entity_type] = []
                if entity_text not in entity_pool[entity_type]:
                    entity_pool[entity_type].append(entity_text)
        
        return entity_pool
    
    def apply_ter(self, text: str, entities: Entities, 
                 entity_pool: Dict[str, List[str]], 
                 replacement_prob: float = 0.5) -> Tuple[str, Entities]:
        """
        Implementa Targeted Entity Random Replacement (TER).
        
        Args:
            text: Texto original.
            entities: Lista de entidades en el texto.
            entity_pool: Pool de entidades agrupadas por tipo.
            replacement_prob: Probabilidad de reemplazar una entidad.
            
        Returns:
            Tuple con el texto modificado y las entidades actualizadas.
        """
        # Crear copia del texto y entidades
        new_text = text
        new_entities = []
        
        # Ordenar entidades por posición de inicio (de mayor a menor)
        sorted_entities = sorted(entities, key=lambda e: e[1], reverse=True)
        
        # Calcular desplazamiento acumulado para ajustar posiciones
        offset = 0
        
        for entity_text, start, end, entity_type in sorted_entities:
            if (entity_type in entity_pool and 
                len(entity_pool[entity_type]) > 1 and 
                random.random() < replacement_prob):
                
                # Seleccionar una entidad diferente del mismo tipo
                replacements = [e for e in entity_pool[entity_type] if e != entity_text]
                if replacements:
                    replacement = random.choice(replacements)
                    
                    # Ajustar posiciones con el offset acumulado
                    adjusted_start = start + offset
                    adjusted_end = end + offset
                    
                    # Reemplazar en el texto
                    new_text = new_text[:adjusted_start] + replacement + new_text[adjusted_end:]
                    
                    # Calcular nuevo offset
                    length_diff = len(replacement) - len(entity_text)
                    offset += length_diff
                    
                    # Agregar la nueva entidad
                    new_entities.append((replacement, adjusted_start, adjusted_start + len(replacement), entity_type))
                else:
                    # Si no hay reemplazos disponibles, mantener la entidad original
                    adjusted_start = start + offset
                    adjusted_end = end + offset
                    new_entities.append((entity_text, adjusted_start, adjusted_end, entity_type))
            else:
                # Mantener la entidad original, pero ajustar posiciones
                adjusted_start = start + offset
                adjusted_end = end + offset
                new_entities.append((entity_text, adjusted_start, adjusted_end, entity_type))
        
        # Ordenar entidades por posición de inicio
        new_entities = sorted(new_entities, key=lambda e: e[1])
        
        return new_text, new_entities
    
    def apply_cwr(self, text: str, entities: Entities, 
                 replacement_prob: float = 0.3, 
                 top_k: int = 5) -> Tuple[str, Entities]:
        """
        Implementa Contextual Word Replacement (CWR) usando BERT.
        
        Args:
            text: Texto original.
            entities: Lista de entidades en el texto.
            replacement_prob: Probabilidad de reemplazar una palabra.
            top_k: Número de candidatos a considerar para reemplazo.
            
        Returns:
            Tuple con el texto modificado y las entidades actualizadas.
        """
        if self.bert_model is None or self.bert_tokenizer is None:
            self.load_bert_model()
        
        # Identificar palabras que no son entidades
        entity_spans = [(start, end) for _, start, end, _ in entities]
        
        # Tokenizar el texto
        tokens = text.split()
        new_tokens = tokens.copy()
        
        # Calcular posiciones de inicio de cada token
        token_positions = []
        pos = 0
        for i, token in enumerate(tokens):
            token_positions.append(pos)
            pos += len(token) + (1 if i < len(tokens) - 1 else 0)
        
        # Procesar cada token
        for i, token in enumerate(tokens):
            # Calcular posición del token en el texto
            token_start = token_positions[i]
            token_end = token_start + len(token)
            
            # Verificar si el token es parte de una entidad
            is_entity = any(start <= token_start and token_end <= end for start, end in entity_spans)
            
            # Si no es una entidad y cumple con la probabilidad, reemplazarlo
            if not is_entity and random.random() < replacement_prob:
                # Crear una copia del texto con el token enmascarado
                masked_text = ' '.join(tokens[:i] + ['[MASK]'] + tokens[i+1:])
                
                # Tokenizar para BERT
                inputs = self.bert_tokenizer(masked_text, return_tensors="pt").to(self.device)
                
                # Obtener predicciones
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    predictions = outputs.logits
                
                # Encontrar el índice del token [MASK]
                mask_token_index = (inputs.input_ids == self.bert_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
                
                if len(mask_token_index) > 0:
                    mask_idx = mask_token_index[0].item()
                    
                    # Obtener las top_k mejores predicciones
                    top_k_indices = torch.topk(predictions[0, mask_idx], top_k).indices.tolist()
                    
                    # Seleccionar un reemplazo aleatorio entre las mejores predicciones
                    replacement_id = random.choice(top_k_indices)
                    replacement = self.bert_tokenizer.decode([replacement_id]).strip()
                    
                    # Reemplazar el token
                    new_tokens[i] = replacement
        
        # Reconstruir el texto con los tokens reemplazados
        new_text = ' '.join(new_tokens)
        
        # Recalcular posiciones de entidades en el nuevo texto
        new_entities = self.realign_entities(new_text, text, entities)
        
        return new_text, new_entities
    
    def realign_entities(self, new_text: str, original_text: str, 
                        original_entities: Entities) -> Entities:
        """
        Recalcula las posiciones de las entidades en el texto modificado.
        
        Args:
            new_text: Texto modificado.
            original_text: Texto original.
            original_entities: Entidades en el texto original.
            
        Returns:
            Entidades con posiciones actualizadas.
        """
        # Esta es una implementación simplificada que asume que las entidades
        # se mantienen intactas y solo cambia el texto alrededor de ellas.
        # Para una implementación más robusta, se necesitaría un algoritmo
        # de alineación de secuencias.
        
        new_entities = []
        for entity_text, _, _, entity_type in original_entities:
            start_idx = new_text.find(entity_text)
            if start_idx != -1:
                end_idx = start_idx + len(entity_text)
                new_entities.append((entity_text, start_idx, end_idx, entity_type))
        
        return new_entities
    
    def apply_combined_augmentation(self, text: str, entities: Entities,
                                   entity_pool: Dict[str, List[str]],
                                   technique: str = "random") -> Tuple[str, Entities]:
        """
        Aplica una combinación de técnicas de aumentación.
        
        Args:
            text: Texto original.
            entities: Lista de entidades en el texto.
            entity_pool: Pool de entidades agrupadas por tipo.
            technique: Técnica a aplicar o "random" para selección aleatoria.
            
        Returns:
            Tuple con el texto aumentado y las entidades actualizadas.
        """
        if technique == "random":
            technique = random.choice([
                "back_translation", "ter", "cwr", 
                "back_translation+ter", "back_translation+cwr"
            ])
        
        if technique == "back_translation":
            return self.back_translate(text, entities)
        
        elif technique == "ter":
            return self.apply_ter(text, entities, entity_pool)
        
        elif technique == "cwr":
            return self.apply_cwr(text, entities)
        
        elif technique == "back_translation+ter":
            # Aplicar back translation primero
            bt_text, bt_entities = self.back_translate(text, entities)
            # Luego aplicar TER
            return self.apply_ter(bt_text, bt_entities, entity_pool)
        
        elif technique == "back_translation+cwr":
            # Aplicar back translation primero
            bt_text, bt_entities = self.back_translate(text, entities)
            # Luego aplicar CWR
            return self.apply_cwr(bt_text, bt_entities)
        
        else:
            raise ValueError(f"Técnica desconocida: {technique}")
    
    def generate_synthetic_data(self, texts: List[str], all_entities: List[Entities],
                               techniques: List[str] = None,
                               num_augmentations: int = 2) -> Tuple[List[str], List[Entities]]:
        """
        Genera datos sintéticos aplicando técnicas de aumentación.
        
        Args:
            texts: Lista de textos originales.
            all_entities: Lista de listas de entidades para cada texto.
            techniques: Lista de técnicas a aplicar. Si es None, se seleccionan aleatoriamente.
            num_augmentations: Número de versiones aumentadas a generar por texto.
            
        Returns:
            Tuple con listas de textos y entidades aumentados.
        """
        if techniques is None:
            techniques = ["back_translation", "ter", "cwr", 
                         "back_translation+ter", "back_translation+cwr"]
        
        # Construir pool de entidades
        entity_pool = self.build_entity_pool(texts, all_entities)
        
        synthetic_texts = []
        synthetic_entities = []
        
        for i, (text, entities) in enumerate(zip(texts, all_entities)):
            logger.info("Generando datos sintéticos para texto %d/%d...", i+1, len(texts))
            
            for j in range(num_augmentations):
                # Seleccionar técnica
                technique = random.choice(techniques)
                
                # Aplicar técnica
                aug_text, aug_entities = self.apply_combined_augmentation(
                    text, entities, entity_pool, technique
                )
                
                synthetic_texts.append(aug_text)
                synthetic_entities.append(aug_entities)
        
        return synthetic_texts, synthetic_entities
    
    def evaluate_synthetic_data_quality(self, original_texts: List[str], 
                                       original_entities: List[Entities],
                                       synthetic_texts: List[str], 
                                       synthetic_entities: List[Entities]) -> Dict[str, float]:
        """
        Evalúa la calidad de los datos sintéticos generados.
        
        Args:
            original_texts: Lista de textos originales.
            original_entities: Lista de listas de entidades originales.
            synthetic_texts: Lista de textos sintéticos.
            synthetic_entities: Lista de listas de entidades sintéticas.
            
        Returns:
            Diccionario con métricas de calidad.
        """
        metrics = {
            "entity_preservation": 0.0,
            "diversity": 0.0
        }
        
        # 1. Evaluar preservación de entidades
        total_entities = 0
        preserved_entities = 0
        
        for orig_entities, syn_entities in zip(original_entities, synthetic_entities):
            # Contar entidades originales
            total_entities += len(orig_entities)
            
            # Verificar cuántas entidades se preservaron
            for orig_entity in orig_entities:
                orig_text, _, _, orig_label = orig_entity
                
                # Buscar si la entidad existe en los datos sintéticos
                for syn_entity in syn_entities:
                    syn_text, _, _, syn_label = syn_entity
                    if syn_text == orig_text and syn_label == orig_label:
                        preserved_entities += 1
                        break
        
        if total_entities > 0:
            metrics["entity_preservation"] = preserved_entities / total_entities
        
        # 2. Evaluar diversidad (usando TF-IDF y similitud coseno)
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Vectorizar textos originales y sintéticos
        vectorizer = TfidfVectorizer()
        all_texts = original_texts + synthetic_texts
        
        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Calcular similitud promedio entre textos originales y sintéticos
            n_orig = len(original_texts)
            orig_vectors = tfidf_matrix[:n_orig]
            syn_vectors = tfidf_matrix[n_orig:]
            
            # Calcular similitud promedio
            similarities = cosine_similarity(orig_vectors, syn_vectors)
            avg_similarity = similarities.mean()
            
            # La diversidad es el complemento de la similitud
            metrics["diversity"] = 1.0 - avg_similarity
        except:
            # En caso de error, asignar un valor predeterminado
            metrics["diversity"] = 0.5
        
        return metrics
    
    def filter_synthetic_data(self, original_texts: List[str], 
                             original_entities: List[Entities],
                             synthetic_texts: List[str], 
                             synthetic_entities: List[Entities],
                             entity_preservation_threshold: float = 0.8,
                             diversity_threshold: float = 0.3) -> Tuple[List[str], List[Entities]]:
        """
        Filtra los datos sintéticos de baja calidad.
        
        Args:
            original_texts: Lista de textos originales.
            original_entities: Lista de listas de entidades originales.
            synthetic_texts: Lista de textos sintéticos.
            synthetic_entities: Lista de listas de entidades sintéticas.
            entity_preservation_threshold: Umbral mínimo de preservación de entidades.
            diversity_threshold: Umbral mínimo de diversidad.
            
        Returns:
            Tuple con listas filtradas de textos y entidades sintéticos.
        """
        filtered_texts = []
        filtered_entities = []
        
        # Evaluar cada ejemplo sintético individualmente
        for i, (syn_text, syn_entities) in enumerate(zip(synthetic_texts, synthetic_entities)):
            # Determinar el texto original correspondiente
            orig_idx = i % len(original_texts)
            
            # Evaluar calidad de este ejemplo específico
            example_quality = self.evaluate_synthetic_data_quality(
                [original_texts[orig_idx]], 
                [original_entities[orig_idx]],
                [syn_text], 
                [syn_entities]
            )
            
            # Filtrar basado en calidad
            if (example_quality["entity_preservation"] >= entity_preservation_threshold and
                example_quality["diversity"] >= diversity_threshold):
                filtered_texts.append(syn_text)
                filtered_entities.append(syn_entities)
        
        return filtered_texts, filtered_entities


# Ejemplo de uso
if __name__ == "__main__":
    # Datos de ejemplo
    texts = [
        "Factura emitida por Empresa ABC con fecha 01/01/2023 por un total de $1500.00",
        "Recibo de Tienda XYZ del 15/02/2023 con monto total $750.50"
    ]
    
    entities = [
        [
            ("Empresa ABC", 19, 30, "COMPANY"),
            ("01/01/2023", 41, 51, "DATE"),
            ("$1500.00", 67, 75, "TOTAL")
        ],
        [
            ("Tienda XYZ", 10, 20, "COMPANY"),
            ("15/02/2023", 25, 35, "DATE"),
            ("$750.50", 53, 60, "TOTAL")
        ]
    ]
    
    # Crear aumentador de datos
    augmenter = SROIEDataAugmenter(use_gpu=False)
    
    # Generar datos sintéticos
    synthetic_texts, synthetic_entities = augmenter.generate_synthetic_data(
        texts, entities, num_augmentations=2
    )
    
    # Evaluar calidad
    quality = augmenter.evaluate_synthetic_data_quality(
        texts, entities, synthetic_texts, synthetic_entities
    )
    
    try:
        logger.info("Datos sintéticos generados:")
        for i, (text, ents) in enumerate(zip(synthetic_texts, synthetic_entities)):
            logger.info("Texto %d: %s", i+1, text)
            logger.info("Entidades:")
            for ent in ents:
                logger.info("  - %s (%s)", ent[0], ent[3])
        
        logger.info("Métricas de calidad:")
        logger.info("Preservación de entidades: %.2f", quality['entity_preservation'])
        logger.info("Diversidad: %.2f", quality['diversity'])
    except Exception as e:
        logger.exception("Error en el ejemplo de uso de sroie_data_augmentation: %s", e)
        raise

