import os
import json
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd


def _convert_to_bio_tags(text: str, annotations: Dict) -> List[str]:
    """
        Convierte anotaciones en formato JSON a etiquetas BIO.
        
        Args:
            text: Texto original.
            annotations: Anotaciones en formato JSON.
            
        Returns:
            Lista de etiquetas BIO.
        """

        # Maneja casos donde las anotaciones pueden ser strings o listas.
    tokens = text.split()
    tags = ['O'] * len(tokens)

    # Precomputar posiciones de inicio de cada token en el texto original
    token_positions = []
    pos = 0
    for i, token in enumerate(tokens):
        token_positions.append(pos)
        pos += len(token) + (1 if i < len(tokens) - 1 else 0)

    for entity_type, entities in annotations.items():
        # Aceptar tanto string como lista de strings
        if isinstance(entities, str):
            entity_list = [entities]
        elif isinstance(entities, (list, tuple)):
            entity_list = list(entities)
        else:
            # Si el formato es inesperado, convertir a string y continuar
            entity_list = [str(entities)]

        for entity in entity_list:
            if not entity:
                continue

            # Buscar todas las apariciones de la entidad en el texto
            start_search = 0
            while True:
                start_idx = text.find(entity, start_search)
                if start_idx == -1:
                    break
                end_idx = start_idx + len(entity)

                # Determinar los índices de tokens que se solapan con la entidad
                first_token = None
                last_token = None
                for ti, tstart in enumerate(token_positions):
                    tend = tstart + len(tokens[ti])
                    # Si hay solapamiento entre [tstart,tend) y [start_idx,end_idx)
                    if not (tend <= start_idx or tstart >= end_idx):
                        if first_token is None:
                            first_token = ti
                        last_token = ti

                if first_token is not None:
                    # Asignar etiquetas BIO a los tokens correspondientes
                    label_b = f'B-{entity_type.upper()}'
                    label_i = f'I-{entity_type.upper()}'
                    tags[first_token] = label_b
                    for j in range(first_token + 1, last_token + 1):
                        if j < len(tags):
                            tags[j] = label_i

                # Continuar búsqueda después de esta ocurrencia
                start_search = end_idx

    return tags

def load_data(data_dir: str) -> Tuple[List[str], List[List[str]]]:
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
        """
        Carga los datos del dataset SROIE.
        
        Args:
            data_dir: Directorio con los archivos del dataset.
            
        Returns:
            Tuple con listas de textos y etiquetas.
        """
        print("Inicio carga datos para DistilBERT...")
        texts = []
        all_tags = []
        
        # Implementar la carga de datos según el formato específico de SROIE
        # Esta es una implementación genérica que debe adaptarse al formato real
        
        # Ejemplo de carga de datos (adaptar según el formato real)
        data_dir_texto = data_dir+"\\box"
        data_dir_tag = data_dir+"\\entities"
        text_files = [f for f in os.listdir(data_dir_texto) if f.endswith('.txt')]
        for text_file in text_files:
            # Cargar texto
            with open(os.path.join(data_dir_texto, text_file), 'r', encoding='utf-8', errors='ignore') as f:
                text = f.readlines()
            data = pd.DataFrame(list(map(parse, text)), columns=[*(f"coor{i}" for i in range(8)), "text"])
            data = data.dropna()
            #print("data",data)
            texto = build_text(data)
            
            # Cargar etiquetas (adaptar según el formato real)
            tag_file = text_file
            if os.path.exists(os.path.join(data_dir_tag, tag_file)):
                with open(os.path.join(data_dir_tag, tag_file), 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                # Convertir anotaciones a formato BIO
                tags = _convert_to_bio_tags(texto, annotations)
                
                texts.append(texto)
                all_tags.append(tags)
        print("Fin carga datos para DistilBERT...,%d textos cargados, tags cargados %d", len(texts), len(all_tags))
if __name__ == "__main__":
     load_data (".\\Data\\sroie\\train")