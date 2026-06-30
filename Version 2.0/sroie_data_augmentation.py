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
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
from difflib import SequenceMatcher
import re

# Variable global que usarán los workers para evitar recrear objetos en cada tarea
WORKER_AUGMENTER = None


def _worker_init(techniques_to_load: Optional[list] = None, use_gpu: bool = False):
    """
    Inicializador para cada proceso del Pool. Crea una instancia de
    SROIEDataAugmenter por worker y precarga solo los modelos necesarios.

    Args:
        techniques_to_load: Lista de técnicas (strings) que indicarán qué modelos
            deben precargarse (por ejemplo 'back_translation' o 'cwr').
        use_gpu: Si se debe usar GPU en el worker (por defecto False para evitar
            conflictos en máquinas sin GPU o para evitar gran uso de memoria GPU).
    """
    global WORKER_AUGMENTER
    try:
        WORKER_AUGMENTER = SROIEDataAugmenter(use_gpu=use_gpu)

        # Precargar solo los modelos necesarios para ahorrar memoria/tiempo
        techniques = set(techniques_to_load or [])
        if any(t.startswith('back_translation') or 'back_translation' in t for t in techniques):
            try:
                WORKER_AUGMENTER.load_translation_models()
            except Exception:
                # Si falla la carga en el worker, dejaremos que el worker intente
                # cargarlos bajo demanda; ya predescargamos en el proceso padre.
                pass

        if any('cwr' in t for t in techniques):
            try:
                WORKER_AUGMENTER.load_bert_model()
            except Exception:
                pass
    except Exception:
        WORKER_AUGMENTER = None

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
        
    def load_translation_models(self, source_lang: str = 'en', target_lang: str = 'de'):
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

        # Determinar longitud máxima soportada por el tokenizer/modelo
        max_len = getattr(tokenizer, 'model_max_length', None) or 512

        # Tokenizar con truncamiento seguro al máximo del modelo
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(self.device)

        # Generación (si el input fue truncado, queda en el límite del modelo)
        outputs = model.generate(**inputs)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return translated_text

    def batch_translate(self, texts: List[str], source_lang: str, target_lang: str, batch_size: int = 8) -> List[str]:
        """
        Traduce una lista de textos en batch para mejorar rendimiento.

        Args:
            texts: Lista de textos a traducir.
            source_lang: Idioma fuente.
            target_lang: Idioma objetivo.
            batch_size: Tamaño de lote para la traducción.

        Returns:
            Lista de textos traducidos en el mismo orden.
        """
        if not texts:
            return []

        model_key = f'{source_lang}-{target_lang}'
        if model_key not in self.translation_models:
            raise ValueError(f"Modelo de traducción {model_key} no cargado.")

        tokenizer = self.translation_models[model_key]['tokenizer']
        model = self.translation_models[model_key]['model']

        max_len = getattr(tokenizer, 'model_max_length', None) or 512
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # Tokenizar en batch
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(self.device)
            outputs = model.generate(**inputs)
            decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            results.extend(decoded)

        return results
    
    def back_translate(self, text: str, entities: Entities,
                   source_lang: str = 'en', target_lang: str = 'de') -> Tuple[str, Entities, dict]:
        """
        Implementa back translation con preservación de entidades.
        Para textos largos, los divide en segmentos, aplica BT a cada uno y recombina.
        Devuelve además metadata sobre máscaras usadas para facilitar debug y dump.
        
        Args:
            text: Texto original.
            entities: Lista de entidades en el texto.
            source_lang: Código del idioma fuente.
            target_lang: Código del idioma objetivo.
            
        Returns:
            Tuple con el texto traducido, las entidades actualizadas y un dict con 'success' (bool), 'technique' y 'mask_meta'.
        """
        if not self.translation_models:
            self.load_translation_models(source_lang, target_lang)
        
        tokenizer_src_tgt = self.translation_models[f'{source_lang}-{target_lang}']['tokenizer']
        max_len = getattr(tokenizer_src_tgt, 'model_max_length', None) or 512
        
        # Aplicar máscaras robustas a las entidades (ASCII-safe para MT)
        masked_text = text
        entity_map = {}
        mask_meta = []
        
        # Ordenar entidades por posición de inicio (de mayor a menor) para reemplazos seguros
        sorted_entities = sorted(entities, key=lambda e: e[1], reverse=True)
        
        for i, (entity_text, start, end, entity_type) in enumerate(sorted_entities):
            # Máscara más robusta: rodeada de subrayados y espacios
            # para aislarla del texto y reducir la probabilidad de ser alterada
            # por el modelo de traducción. Ej: " __ENTX0__ "
            mask = f"__ENTX{i}__"
            # insertar espacios alrededor para evitar que quede pegada a caracteres
            masked_text = masked_text[:start] + " " + mask + " " + masked_text[end:]
            entity_map[mask] = (entity_text, entity_type)
            mask_meta.append({
                'mask': mask, 'entity_text': entity_text,
                'entity_type': entity_type, 'mask_found': False,
                'matched_method': None
            })
        
        # Verificar si el texto cabe en el modelo
        tokenized_len = len(tokenizer_src_tgt.encode(masked_text, add_special_tokens=True))
        
        if tokenized_len <= max_len:
            # Caso 1: Texto pequeño, procesar normalmente
            translated_text = self.translate(masked_text, source_lang, target_lang)
            # detectar si la traducción comprime el texto
            try:
                trans_len = len(tokenizer_src_tgt.encode(translated_text, add_special_tokens=True))
            except Exception:
                pass
            back_translated_text = self.translate(translated_text, target_lang, source_lang)
        else:
            # Caso 2: Texto grande, dividir por oraciones/puntos y procesarlo en partes
            
            # Dividir por puntos finales (. ! ?)
            segments = re.split(r'(\.\s+|\!\s+|\?\s+)', masked_text)
            
            # Recombinar segmentos preservando delimitadores
            combined_segments = []
            i = 0
            while i < len(segments):
                segment = segments[i]
                if segment and segment[0] not in '.!?':
                    # Es un texto normal
                    combined_segments.append(segment)
                    if i + 1 < len(segments) and segments[i + 1] and segments[i + 1][0] in '.!?':
                        combined_segments[-1] += segments[i + 1]
                        i += 2
                        continue
                i += 1
            
            if not combined_segments:
                combined_segments = [masked_text]
            
            # Procesar cada segmento usando traducción por lotes cuando sea posible
            translated_parts = []
            # Primero, procesar segmentos que quepan en max_len y preparar chunks para los muy largos
            small_segments = []
            small_indices = []
            large_segment_chunks = []  # list of lists (chunks per segment)

            for seg_idx, segment in enumerate(combined_segments):
                if not segment.strip():
                    continue
                segment_tokens = len(tokenizer_src_tgt.encode(segment, add_special_tokens=True))
                if segment_tokens <= max_len:
                    small_segments.append(segment)
                    small_indices.append(seg_idx)
                else:
                    # Dividir por palabras en chunks para el segmento grande
                    words = segment.split()
                    word_chunks = []
                    current_chunk = []
                    current_len = 0
                    for word in words:
                        word_token_len = len(tokenizer_src_tgt.encode(word, add_special_tokens=False))
                        if current_len + word_token_len < max_len - 50:
                            current_chunk.append(word)
                            current_len += word_token_len
                        else:
                            if current_chunk:
                                word_chunks.append(' '.join(current_chunk))
                            current_chunk = [word]
                            current_len = word_token_len
                    if current_chunk:
                        word_chunks.append(' '.join(current_chunk))
                    large_segment_chunks.append((seg_idx, word_chunks))

            # Traducir small_segments en batch si hay alguna
            if small_segments:
                try:
                    trans_small = self.batch_translate(small_segments, source_lang, target_lang)
                    back_small = self.batch_translate(trans_small, target_lang, source_lang)
                except Exception as e:
                    logger.warning("Batch translation failed for small segments: %s. Falling back to single translate.", e)
                    back_small = []
                    for s in small_segments:
                        try:
                            t = self.translate(s, source_lang, target_lang)
                            bt = self.translate(t, target_lang, source_lang)
                            back_small.append(bt)
                        except Exception:
                            back_small.append(s)

                # Place back_small results into translated_parts at appropriate indices
                # Initialize translated_parts with empty strings same length as combined_segments
            translated_parts = [''] * len(combined_segments)
            for idx, seg_idx in enumerate(small_indices):
                translated_parts[seg_idx] = back_small[idx]

            # Procesar los segmentos grandes por chunks y traducir cada chunk en batch
            for seg_idx, chunks in large_segment_chunks:
                back_chunks = []
                try:
                    trans_chunks = self.batch_translate(chunks, source_lang, target_lang)
                    back_chunks = self.batch_translate(trans_chunks, target_lang, source_lang)
                except Exception as e:
                    back_chunks = []
                    for c in chunks:
                        try:
                            t = self.translate(c, source_lang, target_lang)
                            bt = self.translate(t, target_lang, source_lang)
                            back_chunks.append(bt)
                        except Exception:
                            back_chunks.append(c)

                translated_parts[seg_idx] = ' '.join(back_chunks)

            # Reemplazar cualquier segmento vacío por su original limpio
            for i, part in enumerate(translated_parts):
                if not part:
                    translated_parts[i] = combined_segments[i]

            back_translated_text = ' '.join(translated_parts)
        
        # Antes de reinserta las entidades, verificar que el texto no se haya comprimido demasiado
        if abs(len(back_translated_text) - len(masked_text)) > len(masked_text) * 0.3:
            # evitar devolver un texto muy comprimido: regresamos la versión original
            # con entidades intactas para no introducir ejemplos corruptos
            return text, entities, {'success': False, 'technique': 'back_translation_aborted_short', 'mask_meta': mask_meta}

        # Reinsertar las entidades originales y recalcular posiciones
        new_entities = []
        # eliminar múltiples espacios que puedan haber surgido al insertar máscaras
        final_text = re.sub(r"\s+", " ", back_translated_text)
        
        # Buscar las máscaras en el texto traducido y reemplazarlas con estrategias de fallback
        for mask, (entity_text, entity_type) in entity_map.items():
            # Localizar el registro meta correspondiente
            meta_idx = next((k for k, m in enumerate(mask_meta) if m['mask'] == mask), None)

            # 1) Búsqueda exacta
            # primero tratamos de localizar la máscara tal cual
            idx = final_text.find(mask)
            found_len = len(mask)
            method = None
            score = None

            if idx != -1:
                method = 'exact'
                score = 1.0

            # 2) búsqueda case-insensitive
            if idx == -1:
                m = re.search(re.escape(mask), final_text, flags=re.IGNORECASE)
                if m:
                    idx = m.start()
                    found_len = m.end() - m.start()
                    method = 'case_insensitive'
                    score = 1.0

            # 3) patrón con espacios opcionales dentro de la máscara
            if idx == -1:
                # por ejemplo "__ENTX0__" -> "__\s*E\s*N\s*T\s*X\s*0\s*__"
                spaced_pattern = r"".join([re.escape(ch) + r"\s*" for ch in mask])
                m = re.search(spaced_pattern, final_text, flags=re.IGNORECASE)
                if m:
                    idx = m.start()
                    found_len = m.end() - m.start()
                    method = 'spaced'
                    score = 0.9

            # 4) Fallback fuzzy (SequenceMatcher) buscando la mejor ventana aproximada
            if idx == -1:
                best_ratio = 0.0
                best_span = None
                mask_len = len(mask)
                # Barrer con paso para reducir costo, buscar ventanas del tamaño de la máscara
                step = max(1, mask_len // 4)
                for i in range(0, max(1, len(final_text) - mask_len + 1), step):
                    window = final_text[i:i + mask_len]
                    ratio = SequenceMatcher(None, window, mask).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_span = (i, i + len(window))
                # reducir umbral para ser más permisivo
                if best_ratio >= 0.5 and best_span is not None:
                    idx = best_span[0]
                    found_len = best_span[1] - best_span[0]
                    method = 'fuzzy'
                    score = best_ratio

            if idx == -1:
                # Intentar búsqueda directa de la entidad (normalized) como fallback
                norm_final = final_text.lower().strip()
                norm_ent = entity_text.lower().strip()
                m_ent = re.search(re.escape(norm_ent), norm_final, flags=re.IGNORECASE)
                if m_ent:
                    idx = m_ent.start()
                    found_len = len(m_ent.group(0))
                    method = 'entity_direct'
                    score = 1.0
                else:
                    # Registrar en meta que no se encontró (por ahora)
                    if meta_idx is not None:
                        mask_meta[meta_idx]['mask_found'] = False
                        mask_meta[meta_idx]['matched_method'] = None
                        mask_meta[meta_idx]['match_score'] = None
                        mask_meta[meta_idx]['replaced_pos'] = None
                    # Intentar fuzzy matching sobre la entidad (no solo la máscara)
                    best_ratio_ent = 0.0
                    best_span_ent = None
                    ent_len = len(entity_text)
                    step_ent = max(1, ent_len // 4)
                    for i in range(0, max(1, len(final_text) - ent_len + 1), step_ent):
                        window = final_text[i:i + ent_len]
                        ratio = SequenceMatcher(None, window, entity_text).ratio()
                        if ratio > best_ratio_ent:
                            best_ratio_ent = ratio
                            best_span_ent = (i, i + len(window))
                    if best_ratio_ent >= 0.65 and best_span_ent is not None:
                        idx = best_span_ent[0]
                        found_len = best_span_ent[1] - best_span_ent[0]
                        method = 'fuzzy_entity'
                        score = best_ratio_ent
                        logger.debug("Entity fuzzy matched: %s -> ratio=%.2f at pos=%d", entity_text, best_ratio_ent, idx)

            if idx == -1:
                # Si aún no se encontró, continuar (entidad perdida)
                return text, entities, {'success': False, 'technique': 'back_translation_aborted_short', 'mask_meta': mask_meta}

            # Reemplazar la porción encontrada por el texto de la entidad original
            final_text = final_text[:idx] + entity_text + final_text[idx + found_len:]
            new_entities.append((entity_text, idx, idx + len(entity_text), entity_type))

            # Actualizar meta con información de coincidencia
            if meta_idx is not None:
                mask_meta[meta_idx]['mask_found'] = True
                mask_meta[meta_idx]['matched_method'] = method
                mask_meta[meta_idx]['match_score'] = score
                mask_meta[meta_idx]['replaced_pos'] = (idx, idx + len(entity_text))
        
        # Ordenar entidades por posición de inicio
        new_entities = sorted(new_entities, key=lambda e: e[1])
        
        return final_text, new_entities, {'success': True, 'mask_meta': mask_meta}
    
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
        Implementa Targeted Entity Random Replacement (TER) - OPTIMIZADA.
        
        Args:
            text: Texto original.
            entities: Lista de entidades en el texto.
            entity_pool: Pool de entidades agrupadas por tipo.
            replacement_prob: Probabilidad de reemplazar una entidad.
            
        Returns:
            Tuple con el texto modificado y las entidades actualizadas.
        """
        # OPTIMIZACIÓN: si no hay entidades o pool vacío, devolver original
        if not entities or not entity_pool:
            return text, entities
        
        # Crear copia del texto y entidades
        new_text = text
        new_entities = []
        
        # Ordenar entidades por posición de inicio (de mayor a menor)
        sorted_entities = sorted(entities, key=lambda e: e[1], reverse=True)
        
        # Calcular desplazamiento acumulado para ajustar posiciones
        offset = 0
        
        for entity_text, start, end, entity_type in sorted_entities:
            # OPTIMIZACIÓN: pre-computar lista de reemplazos válidos
            if entity_type in entity_pool:
                replacements = [e for e in entity_pool[entity_type] if e != entity_text]
            else:
                replacements = []
            
            if replacements and random.random() < replacement_prob:
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

                # Comprobar longitud de la secuencia para BERT
                bert_max_len = getattr(self.bert_tokenizer, 'model_max_length', None) or 512
                try:
                    tokenized_len = len(self.bert_tokenizer.encode(masked_text, add_special_tokens=True))
                except Exception:
                    tokenized_len = 0

                # Si excede el máximo, saltamos este token para evitar errores
                if tokenized_len == 0 or tokenized_len > bert_max_len:
                    logger.debug("Omitiendo reemplazo CWR para token '%s' (tokens=%d, max=%d)", token, tokenized_len, bert_max_len)
                    continue

                # Tokenizar para BERT con truncamiento seguro
                inputs = self.bert_tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=bert_max_len).to(self.device)

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
    
    def _is_noisy_for_bt(self, text: str, non_alnum_ratio_thresh: float = 0.2, repeated_char_ratio: float = 0.05) -> bool:
        """
        Heurística simple para decidir si un texto es demasiado ruidoso para aplicar back-translation.
        - non_alnum_ratio_thresh: proporción de caracteres no alfanuméricos (espacios y puntuación) sobre el total
        - repeated_char_ratio: proporción de caracteres con repeticiones extensas (ej: 'A_A_A_...') sobre el total
        """
        if not text:
            return False
        total = len(text)
        non_alnum = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if total == 0:
            return True
        if (non_alnum / total) > non_alnum_ratio_thresh:
            return True
        # Detectar runs largos de pattern char + underscore (ej 'A_A_A_')
        long_runs = re.findall(r'(\w)(?:_\1){3,}', text)
        if long_runs:
            # Si hay muchos de estos patterns, considerarlo ruidoso
            if len(long_runs) / max(1, total) > repeated_char_ratio:
                return True
        return False

    def apply_combined_augmentation(self, text: str, entities: Entities,
                                   entity_pool: Dict[str, List[str]],
                                   technique: str = "random") -> Tuple[str, Entities, dict]:
        """
        Aplica una combinación de técnicas de aumentación.
        
        Args:
            text: Texto original.
            entities: Lista de entidades en el texto.
            entity_pool: Pool de entidades agrupadas por tipo.
            technique: Técnica a aplicar o "random" para selección aleatoria.
            
        Returns:
            Tuple con el texto aumentado, las entidades actualizadas y un dict con metadata (incluyendo 'success' para back_translation).
        """        
        if technique == "random":
            technique = random.choice([
                "back_translation", "ter", "cwr", 
                "back_translation+ter", "back_translation+cwr"
            ])
        
        if technique == "back_translation":
            # Si el texto es muy ruidoso (ej: patrones A_A_A), evitar BT y volver con metadata indicando skip
            if self._is_noisy_for_bt(text):
                return text, entities, {'success': False, 'technique': 'back_translation_skipped_noise', 'mask_meta': []}
            # back_translate returns (text, entities, meta_dict)
            bt_text, bt_entities, bt_meta = self.back_translate(text, entities)
            if not bt_meta.get('success', True):
                # Si BT falló, aplicar TER como compensación
                final_text, final_entities = self.apply_ter(text, entities, entity_pool)
                return final_text, final_entities, {'success': True, 'technique': 'ter_due_to_bt_failure', 'mask_meta': bt_meta['mask_meta']}
            return bt_text, bt_entities, bt_meta
        
        elif technique == "ter":
            new_text, new_entities = self.apply_ter(text, entities, entity_pool)
            return new_text, new_entities, {'technique': 'ter', 'mask_meta': []}
        
        elif technique == "cwr":
            new_text, new_entities = self.apply_cwr(text, entities)
            return new_text, new_entities, {'technique': 'cwr', 'mask_meta': []}
        
        elif technique == "back_translation+ter":
            # Si el texto es ruidoso, hacer TER directamente
            if self._is_noisy_for_bt(text):
                final_text, final_entities = self.apply_ter(text, entities, entity_pool)
                return final_text, final_entities, {'success': True, 'technique': 'ter_due_to_bt_noise', 'mask_meta': []}
            # Aplicar back translation primero
            bt_text, bt_entities, bt_meta = self.back_translate(text, entities)
            if not bt_meta.get('success', True):
                # Si BT falló, aplicar solo TER
                final_text, final_entities = self.apply_ter(text, entities, entity_pool)
                return final_text, final_entities, {'success': True, 'technique': 'ter_due_to_bt_failure', 'mask_meta': bt_meta['mask_meta']}
            # Luego aplicar TER
            final_text, final_entities = self.apply_ter(bt_text, bt_entities, entity_pool)
            return final_text, final_entities, {'success': True, 'technique': 'back_translation+ter', 'mask_meta': bt_meta['mask_meta']}
        
        elif technique == "back_translation+cwr":
            # Si el texto es ruidoso, hacer CWR directamente
            if self._is_noisy_for_bt(text):
                final_text, final_entities = self.apply_cwr(text, entities)
                return final_text, final_entities, {'success': True, 'technique': 'cwr_due_to_bt_noise', 'mask_meta': []}
            # Aplicar back translation primero
            bt_text, bt_entities, bt_meta = self.back_translate(text, entities)
            if not bt_meta.get('success', True):
                # Si BT falló, aplicar solo CWR
                final_text, final_entities = self.apply_cwr(text, entities)
                return final_text, final_entities, {'success': True, 'technique': 'cwr_due_to_bt_failure', 'mask_meta': bt_meta['mask_meta']}
            # Luego aplicar CWR
            final_text, final_entities = self.apply_cwr(bt_text, bt_entities)
            return final_text, final_entities, {'success': True, 'technique': 'back_translation+cwr', 'mask_meta': bt_meta['mask_meta']}
        
        else:
            raise ValueError(f"Técnica desconocida: {technique}")
    
    def _ensure_all_entity_labels(self, text: str, entities: Entities,
                                  entity_pool: Dict[str, List[str]],
                                  aug_text: str, aug_entities: Entities,
                                  technique: str) -> Tuple[str, Entities]:
        """
        Verifica que el conjunto de tipos de entidad en el ejemplo aumentado
        coincida con el original. Si no, aplica TER como fallback para mantener
        el conteo exacto de entidades.
        """
        orig_labels = {label for _, _, _, label in entities}
        syn_labels = {label for _, _, _, label in aug_entities}
        if orig_labels != syn_labels:
            aug_text, aug_entities = self.apply_ter(text, entities, entity_pool)
        return aug_text, aug_entities

    def generate_synthetic_data(self, texts: List[str], all_entities: List[Entities],
                                   techniques: List[str] = None,
                                   num_augmentations: int = 2,
                                   use_parallel: bool = False,
                                   use_threads: bool = True,
                                   num_workers: int = 4) -> Tuple[List[str], List[Entities]]:
        """
        Genera datos sintéticos aplicando técnicas de aumentación.
        
        Args:
            texts: Lista de textos originales.
            all_entities: Lista de listas de entidades para cada texto.
            techniques: Lista de técnicas a aplicar. Si es None, se seleccionan aleatoriamente.
            num_augmentations: Número de versiones aumentadas a generar por texto.
            use_parallel: Si se debe usar multiprocessing para paralelizar.
            use_threads: Si se debe usar hilos (ThreadPool) para compartir modelos entre workers.
            num_workers: Número de procesos/hilos paralelos.
            
        Returns:
            Tuple con listas de textos y entidades aumentados.
        """
        if techniques is None:
            # Usar todas las técnicas disponibles (incluyendo back_translation)
            techniques = ["back_translation", "ter", "cwr", 
                         "back_translation+ter", "back_translation+cwr"]
        
        # Construir pool de entidades una sola vez
        entity_pool = self.build_entity_pool(texts, all_entities)
        
        synthetic_texts = []
        synthetic_entities = []
        synthetic_meta = []  # metadata per synthetic example (e.g., mask info)
        
        if use_parallel:
            # Paralelizar con multiprocessing
            from multiprocessing import Pool

            # Crear lista de tareas: (texto, entidades, entity_pool, technique_choice, índice)
            tasks = []
            chosen_techniques = set()
            for i, (text, entities) in enumerate(zip(texts, all_entities)):
                for j in range(num_augmentations):
                    technique = random.choice(techniques)
                    tasks.append((text, entities, entity_pool, technique, i, j))
                    chosen_techniques.add(technique)

            logger.info("Iniciando paralelización con %d workers para %d tareas...", num_workers, len(tasks))

            # Predescargar modelos en el proceso principal para evitar descargas simultáneas
            try:
                main_augmenter = SROIEDataAugmenter(use_gpu=False)
                if any(t.startswith('back_translation') or 'back_translation' in t for t in chosen_techniques):
                    main_augmenter.load_translation_models()
                if any('cwr' in t for t in chosen_techniques):
                    main_augmenter.load_bert_model()
            except Exception as e:
                logger.warning("No se pudieron predescargar todos los modelos en el proceso principal: %s", e)

            # Si se prefieren threads (comparten memoria/modelos), usar ThreadPoolExecutor.
            def _thread_worker_augment(shared_augmenter: SROIEDataAugmenter, text: str, entities: Entities, entity_pool: Dict[str, List[str]], technique: str, task_idx: int, sub_idx: int):
                try:
                    aug_text, aug_entities, aug_meta = shared_augmenter.apply_combined_augmentation(
                        text, entities, entity_pool, technique
                    )
                    aug_text, aug_entities = shared_augmenter._ensure_all_entity_labels(
                        text, entities, entity_pool, aug_text, aug_entities, technique
                    )
                    return aug_text, aug_entities, aug_meta, task_idx
                except Exception as e:
                    logger.error("Error en thread worker para tarea %d.%d: %s", task_idx, sub_idx, e)
                    return text, entities, [], task_idx

            # Iniciar Pool pasando las técnicas necesarias al inicializador si usamos procesos.
            try:
                total_tasks = len(tasks)
                logger.info("Total de tareas a procesar: %d", total_tasks)
                if use_threads:
                    # Compartir la misma instancia de augmenter entre hilos
                    main_augmenter = SROIEDataAugmenter(use_gpu=False)
                    if any(t.startswith('back_translation') or 'back_translation' in t for t in chosen_techniques):
                        main_augmenter.load_translation_models()
                    if any('cwr' in t for t in chosen_techniques):
                        main_augmenter.load_bert_model()

                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        futures = [executor.submit(_thread_worker_augment, main_augmenter, text, entities, entity_pool, technique, i, j)
                                   for (text, entities, entity_pool, technique, i, j) in tasks]

                        completed = 0
                        log_step = max(1, total_tasks // 20)
                        for fut in as_completed(futures):
                            aug_text, aug_entities, aug_meta, task_idx = fut.result()
                            synthetic_texts.append(aug_text)
                            synthetic_entities.append(aug_entities)
                            synthetic_meta.append(aug_meta)
                            completed += 1
                            if completed % log_step == 0 or completed == total_tasks:
                                logger.info("Progreso threads: %d/%d (%.1f%%)", completed, total_tasks, completed * 100.0 / total_tasks)

                    logger.info("Paralelización con threads completada. Generados %d textos sintéticos.", len(synthetic_texts))
                else:
                    # Usar imap_unordered para procesar resultados conforme se completan y mostrar progreso
                    with Pool(processes=num_workers, initializer=_worker_init, initargs=(list(chosen_techniques), False)) as pool:
                        completed = 0
                        log_step = max(1, total_tasks // 20)
                        for aug_text, aug_entities, aug_meta, task_idx in pool.imap_unordered(_worker_augment_star, tasks):
                            synthetic_texts.append(aug_text)
                            synthetic_entities.append(aug_entities)
                            synthetic_meta.append(aug_meta)
                            completed += 1
                            if completed % log_step == 0 or completed == total_tasks:
                                logger.info("Progreso procesos: %d/%d (%.1f%%)", completed, total_tasks, completed * 100.0 / total_tasks)

                    logger.info("Paralelización con procesos completada. Generados %d textos sintéticos.", len(synthetic_texts))

            except (MemoryError, RuntimeError, torch.cuda.CudaError) as e:
                # Si ocurre un error de asignación de memoria u otro problema relacionado,
                # hacer fallback a ejecución secuencial reduciendo uso de memoria.
                logger.error("Paralelización fallida por error: %s. Reintentando de forma secuencial.", e)
                # Reintentar secuencialmente
                for i, (text, entities) in enumerate(zip(texts, all_entities)):
                    logger.info("Generando datos sintéticos (secuencial) para texto %d/%d...", i+1, len(texts))
                    for j in range(num_augmentations):
                        technique = random.choice(techniques)
                        aug_text, aug_entities, aug_meta = self.apply_combined_augmentation(
                            text, entities, entity_pool, technique
                        )
                        aug_text, aug_entities = self._ensure_all_entity_labels(
                            text, entities, entity_pool, aug_text, aug_entities, technique
                        )
                        synthetic_texts.append(aug_text)
                        synthetic_entities.append(aug_entities)
                        synthetic_meta.append(aug_meta)

            except Exception as e:
                # Capturar otros errores y hacer fallback
                logger.exception("Error inesperado al paralelizar: %s. Reintentando de forma secuencial.", e)
                for i, (text, entities) in enumerate(zip(texts, all_entities)):
                    logger.info("Generando datos sintéticos (secuencial) para texto %d/%d...", i+1, len(texts))
                    for j in range(num_augmentations):
                        technique = random.choice(techniques)
                        aug_text, aug_entities, aug_meta = self.apply_combined_augmentation(
                            text, entities, entity_pool, technique
                        )
                        synthetic_texts.append(aug_text)
                        synthetic_entities.append(aug_entities)
                        synthetic_meta.append(aug_meta)
        else:
            # Versión secuencial (original) - optimizada en logging
            total_tasks = len(texts) * max(1, num_augmentations)
            log_step = max(1, total_tasks // 20)
            task_counter = 0
            for i, (text, entities) in enumerate(zip(texts, all_entities)):
                for j in range(num_augmentations):
                    # Seleccionar técnica
                    technique = random.choice(techniques)

                    # Aplicar técnica
                    aug_text, aug_entities, aug_meta = self.apply_combined_augmentation(
                        text, entities, entity_pool, technique
                    )
                    aug_text, aug_entities = self._ensure_all_entity_labels(
                        text, entities, entity_pool, aug_text, aug_entities, technique
                    )

                    synthetic_texts.append(aug_text)
                    synthetic_entities.append(aug_entities)
                    synthetic_meta.append(aug_meta)

                    task_counter += 1
                    if task_counter % log_step == 0 or task_counter == total_tasks:
                        logger.info("Progreso secuencial: %d/%d (%.1f%%)", task_counter, total_tasks, task_counter * 100.0 / total_tasks)
        
        return synthetic_texts, synthetic_entities, synthetic_meta
    
    def evaluate_synthetic_data_quality(self, original_texts: List[str], 
                                       original_entities: List[Entities],
                                       synthetic_texts: List[str], 
                                       synthetic_entities: List[Entities],
                                       entity_similarity_threshold: float = 0.75) -> Dict[str, float]:
        """
        Evalúa la calidad de los datos sintéticos generados.
        
        Args:
            original_texts: Lista de textos originales.
            original_entities: Lista de listas de entidades originales.
            synthetic_texts: Lista de textos sintéticos.
            synthetic_entities: Lista de listas de entidades sintéticas.
            entity_similarity_threshold: Umbral (0-1) para considerar que dos cadenas de entidad son similares.
            
        Returns:
            Diccionario con métricas de calidad.
        """
        from difflib import SequenceMatcher
        import unicodedata
        import re

        def normalize(s: str) -> str:
            s = unicodedata.normalize('NFKC', s).lower().strip()
            s = re.sub(r'\s+', ' ', s)
            return s

        def similar(a: str, b: str, thresh: float) -> bool:
            a_n = normalize(a)
            b_n = normalize(b)
            if a_n == b_n:
                return True
            if a_n in b_n or b_n in a_n:
                return True
            ratio = SequenceMatcher(None, a_n, b_n).ratio()
            return ratio >= thresh

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
            
            # Verificar cuántas entidades se preservaron (comparación relajada)
            for orig_entity in orig_entities:
                orig_text, _, _, orig_label = orig_entity
                found = False
                for syn_entity in syn_entities:
                    syn_text, _, _, syn_label = syn_entity
                    if syn_label != orig_label:
                        continue
                    if similar(orig_text, syn_text, entity_similarity_threshold):
                        preserved_entities += 1
                        found = True
                        break
                if not found:
                    # También considerar búsqueda dentro del texto sintético general (fallback)
                    # si la entidad original aparece en el texto sintético
                    for syn_text_full in synthetic_texts:
                        if normalize(orig_text) in normalize(syn_text_full):
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
            metrics["diversity"] = float(max(0.0, min(1.0, 1.0 - avg_similarity)))
        except Exception:
            # En caso de error, asignar un valor predeterminado
            metrics["diversity"] = 0.5

        return metrics

    def filter_synthetic_data(self, original_texts: List[str], 
                             original_entities: List[Entities],
                             synthetic_texts: List[str], 
                             synthetic_entities: List[Entities],
                             synthetic_meta: Optional[List[dict]] = None,
                             entity_preservation_threshold: float = 0.0,
                             diversity_threshold: float = 0.0,
                             rejected_dump_path: Optional[str] = None) -> Tuple[List[str], List[Entities]]:
        """
        Filtra los datos sintéticos de baja calidad.
        
        Args:
            original_texts: Lista de textos originales.
            original_entities: Lista de listas de entidades originales.
            synthetic_texts: Lista de textos sintéticos.
            synthetic_entities: Lista de listas de entidades sintéticas.
            synthetic_meta: Lista opcional de metadata por ejemplo sintético (p.ej. mask_info).
            entity_preservation_threshold: Umbral mínimo de preservación de entidades.
            diversity_threshold: Umbral mínimo de diversidad.
            rejected_dump_path: Ruta opcional para volcar ejemplos rechazados (JSONL).
            
        Returns:
            Tuple con listas filtradas de textos y entidades sintéticos.
        """
        import datetime
        filtered_texts = []
        filtered_entities = []
        rejected_examples = []
        
        # Evaluar cada ejemplo sintético individualmente
        for i, (syn_text, syn_entities) in enumerate(zip(synthetic_texts, synthetic_entities)):
            # Determinar el texto original correspondiente
            orig_idx = i % len(original_texts)
            orig_text = original_texts[orig_idx]
            orig_ents = original_entities[orig_idx]

            # Sanitizar y arreglar alineaciones de entidades sintéticas (intentos simples)
            entity_fixups = []
            clean_syn_entities = []
            for ent_text, start, end, label in syn_entities:
                valid = True
                if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end > len(syn_text) or start >= end:
                    # Intentar buscar la entidad por texto en el sin_text
                    m = re.search(re.escape(ent_text), syn_text, flags=re.IGNORECASE)
                    if m:
                        ns, ne = m.start(), m.end()
                        entity_fixups.append({'orig': (ent_text, start, end, label), 'fixed': (ns, ne)})
                        clean_syn_entities.append((ent_text, ns, ne, label))
                    else:
                        # Intentar búsqueda normalizada
                        norm_ent = ent_text.lower().strip()
                        m2 = re.search(re.escape(norm_ent), syn_text.lower())
                        if m2:
                            ns, ne = m2.start(), m2.end()
                            entity_fixups.append({'orig': (ent_text, start, end, label), 'fixed': (ns, ne)})
                            clean_syn_entities.append((ent_text, ns, ne, label))
                        else:
                            # No se pudo arreglar
                            entity_fixups.append({'orig': (ent_text, start, end, label), 'fixed': None})
                            valid = False
                else:
                    # Comprueba coherencia de texto
                    if syn_text[start:end] != ent_text:
                        # Intentar localizar la entidad por contenido
                        m = re.search(re.escape(ent_text), syn_text, flags=re.IGNORECASE)
                        if m:
                            ns, ne = m.start(), m.end()
                            entity_fixups.append({'orig': (ent_text, start, end, label), 'fixed': (ns, ne)})
                            clean_syn_entities.append((ent_text, ns, ne, label))
                        else:
                            entity_fixups.append({'orig': (ent_text, start, end, label), 'fixed': None})
                            valid = False
                    else:
                        clean_syn_entities.append((ent_text, start, end, label))

            # Reemplazar con la versión limpiada
            syn_entities = clean_syn_entities

            # Intentar recuperar entidades originales perdidas por error de alineación
            def normalize_string(s: str) -> str:
                return re.sub(r"\s+", " ", s.strip().lower())

            def find_normalized_substring(text: str, pattern: str) -> Optional[tuple]:
                text_compact = re.sub(r"\s+", "", text.lower())
                pattern_compact = re.sub(r"\s+", "", pattern.lower())
                idx = text_compact.find(pattern_compact)
                if idx == -1:
                    return None

                # Mapear índice compacto de regreso a la posición original
                char_count = 0
                start = None
                for pos, ch in enumerate(text):
                    if ch.isspace():
                        continue
                    if char_count == idx:
                        start = pos
                        break
                    char_count += 1
                if start is None:
                    return None

                end = start
                matched = 0
                while end < len(text) and matched < len(pattern_compact):
                    if not text[end].isspace():
                        matched += 1
                    end += 1
                return (start, end)

            recovered_entities = []
            for orig_entity_text, _, _, orig_label in orig_ents:
                normalized_orig = normalize_string(orig_entity_text)
                if any(normalize_string(ent_text) == normalized_orig and ent_label == orig_label
                       for ent_text, _, _, ent_label in syn_entities):
                    continue

                m = re.search(re.escape(orig_entity_text), syn_text, flags=re.IGNORECASE)
                if m:
                    recovered_entities.append((orig_entity_text, m.start(), m.end(), orig_label))
                else:
                    # Si no se encuentra con coincidencia exacta, buscar versión normalizada
                    match_pos = find_normalized_substring(syn_text, orig_entity_text)
                    if match_pos:
                        recovered_entities.append((orig_entity_text, match_pos[0], match_pos[1], orig_label))

            if recovered_entities:
                syn_entities.extend(recovered_entities)
                syn_entities = sorted(syn_entities, key=lambda e: e[1])

            # Evaluar calidad de este ejemplo específico
            example_quality = self.evaluate_synthetic_data_quality(
                [orig_text], 
                [orig_ents],
                [syn_text], 
                [syn_entities]
            )
            
            # Filtrar basado en calidad
            passed = (example_quality["entity_preservation"] >= entity_preservation_threshold and
                      example_quality["diversity"] >= diversity_threshold)
            if passed:
                filtered_texts.append(syn_text)
                filtered_entities.append(syn_entities)
            else:
                # Registrar motivo(s) de rechazo
                reasons = []
                if example_quality["entity_preservation"] < entity_preservation_threshold:
                    reasons.append("low_entity_preservation")
                if example_quality["diversity"] < diversity_threshold:
                    reasons.append("low_diversity")

                rejected_examples.append({
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    "orig_index": orig_idx,
                    "orig_text": orig_text,
                    "orig_entities": orig_ents,
                    "syn_index": i,
                    "syn_text": syn_text,
                    "syn_entities": syn_entities,
                    "entity_preservation": example_quality.get("entity_preservation"),
                    "diversity": example_quality.get("diversity"),
                    "reasons": reasons,
                    "technique": (synthetic_meta[i].get('technique') if synthetic_meta and i < len(synthetic_meta) and isinstance(synthetic_meta[i], dict) else None),
                    "mask_info": (synthetic_meta[i] if synthetic_meta and i < len(synthetic_meta) else None),
                    "entity_alignment_fixes": entity_fixups
                })
        
        # Volcar ejemplos rechazados si se indicó una ruta
        if rejected_dump_path and rejected_examples:
            try:
                os.makedirs(os.path.dirname(rejected_dump_path), exist_ok=True)
                with open(rejected_dump_path, 'w', encoding='utf-8') as rf:
                    for item in rejected_examples:
                        rf.write(json.dumps(item, ensure_ascii=False) + "\n")
                logger.info("Dump de ejemplos rechazados guardado en: %s (total=%d)", rejected_dump_path, len(rejected_examples))
            except Exception as e:
                logger.exception("No se pudo guardar dump de ejemplos rechazados: %s", e)
        
        return filtered_texts, filtered_entities


def _worker_augment(text: str, entities: Entities, entity_pool: Dict[str, List[str]], 
                   technique: str, task_idx: int, sub_idx: int) -> Tuple[str, Entities, Optional[List[dict]], int]:
    """
    Worker function para paralelización con multiprocessing.
    Crea una instancia local del aumentador sin GPU.
    
    Args:
        text: Texto a aumentar.
        entities: Entidades del texto.
        entity_pool: Pool de entidades compartido.
        technique: Técnica a usar.
        task_idx: Índice de la tarea (para logging).
        sub_idx: Sub-índice dentro de la tarea.
    
    Returns:
        Tuple (texto_aumentado, entidades_aumentadas, metadata, task_idx).
    """
    try:
        # Reutilizar el augmenter creado por _worker_init si existe
        global WORKER_AUGMENTER
        if WORKER_AUGMENTER is None:
            augmenter = SROIEDataAugmenter(use_gpu=False)
        else:
            augmenter = WORKER_AUGMENTER

        # Aplicar técnica
        aug_text, aug_entities, aug_meta = augmenter.apply_combined_augmentation(
            text, entities, entity_pool, technique
        )

        return aug_text, aug_entities, aug_meta, task_idx
    except Exception as e:
        logger.error("Error en worker para tarea %d.%d: %s", task_idx, sub_idx, e)
        # En caso de error, devolver el texto original
        return text, entities, [], task_idx


def _worker_augment_star(args_tuple):
    """
    Wrapper para permitir imap_unordered con Pool pasando una tupla de argumentos.
    """
    return _worker_augment(*args_tuple)


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
    
    # Generar datos sintéticos (AHORA CON PARALELIZACIÓN)
    # use_parallel=True activa multiprocessing para acelerar
    # num_workers: ajustar según CPU disponible (ej: 4, 8, etc.)
    import time
    inicio = time.time()
    
    synthetic_texts, synthetic_entities, synthetic_meta = augmenter.generate_synthetic_data(
        texts, entities, 
        num_augmentations=2,
        use_parallel=True,  # PARALELIZACIÓN ACTIVADA
        num_workers=4       # Ajustar según CPU
    )
    
    tiempo = time.time() - inicio
    logger.info("Tiempo total: %.2f segundos", tiempo)
    
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

