import os
import json
import traceback
import sys
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from logging_config import get_logger

logger = get_logger(__name__)

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

def validate_and_fix_alignment(text: str, entities: List[Tuple[int, int, str]]) -> Tuple[str, List[Tuple[int, int, str]]]:
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
            logger.debug("Removidas %d entidades inválidas en validación inicial", removed_before_truncate)

        if len(initial_valid) > 0:
            logger.debug("Entidades válidas encontradas: %d", len(initial_valid))
        
        
        return text, entities

def load_data(data_dir: str) -> List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]]:
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
                    logger.info("Entidad encontrada: '%s' (tipo=%s, start=%d, end=%d)", value_stripped, entity_type, pos_key[0], pos_key[1])
                else:
                    logger.info("Entidad no encontrada en texto: '%s' (tipo=%s, archivo=%s)",
                                value_stripped[:50], entity_type, text_file)
            logger.info("Cargado archivo %s con %d entidades válidas", text_file, len(entities))
            # Validar y fijar alineación de entidades antes de agregar
            cleaned_text, valid_entities = validate_and_fix_alignment(text, entities)

            # Asegurar que las entidades sean tuplas (List[Tuple[int,int,str]])
            if valid_entities:
                valid_entities = [tuple(ent) for ent in valid_entities]
                spacy_data.append((cleaned_text, {"entities": valid_entities}))
                logger.info("Cargado archivo %s con %d entidades válidas", text_file, len(valid_entities))
            else:
                logger.info("Archivo %s no tiene entidades válidas tras limpieza", text_file)
    
    return spacy_data


def main():
    try:
        data_dir = os.path.join(os.getcwd(), 'Data', 'sroie', 'completo')
        out_file = os.path.join(os.getcwd(), 'output', 'spacy_loaded.json')
        logger.info('Data dir: %s', data_dir)

        spacy_data = load_data(data_dir)
        logger.info('Loaded records: %d', len(spacy_data))

        serial = []
        for text, ann in spacy_data:
            try:
                # spacy_data already has structure: (text, {"entities": [(start, end, label), ...]})
                ents = ann.get('entities', []) if isinstance(ann, dict) else []
                # Convert to list of arrays for JSON serialization
                ent_arrays = []
                for e in ents:
                    if e and len(e) >= 3:
                        s, epos, lab = int(e[0]), int(e[1]), str(e[2])
                        ent_arrays.append([s, epos, lab])
                # Serialize as [text, {"entities": [[start, end, label], ...]}]
                serial.append([text, {"entities": ent_arrays}])
            except Exception as item_err:
                logger.warning("  Skipping item: %s", item_err)
                continue

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(serial, f, ensure_ascii=False, indent=2)

        logger.info('Wrote: %s', out_file)
        logger.info('Total records written: %d', len(serial))
        if len(serial) > 0:
            logger.info('First record sample:')
            sample_text = json.dumps(serial[0], ensure_ascii=False, indent=2)
            logger.info(sample_text[:1500])

    except Exception as ex:
        logger.error('Error during generation: %s', ex)
        traceback.print_exc()


if __name__ == '__main__':
    main()
