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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support
from typing import List, Dict, Tuple, Any, Optional
import json
import re

# Importar el aumentador de datos
from sroie_data_augmentation import SROIEDataAugmenter, Entity, Entities
from logging_config import get_logger

logger = get_logger(__name__)

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
    
    def _align_labels(self, tags, text, offset_mapping):
        """Alinea las etiquetas con los tokens de DistilBERT."""
        # Inicializar con -100 (ignorado por la función de pérdida)
        label_ids = [-100] * len(offset_mapping)
        
        # Tokenizar el texto original para alinear con las etiquetas
        tokens = text.split()
        
        if len(tokens) != len(tags):
            # Si hay desalineación, usar una estrategia simple
            return [-100] * len(offset_mapping)
        
        # Calcular posiciones de inicio de cada token
        token_positions = []
        pos = 0
        for i, token in enumerate(tokens):
            token_positions.append(pos)
            pos += len(token) + (1 if i < len(tokens) - 1 else 0)
        
        # Asignar etiquetas a tokens de DistilBERT
        for i, (start, end) in enumerate(offset_mapping):
            # Ignorar tokens especiales
            if start == 0 and end == 0:
                continue
            
            # Encontrar el token original que corresponde a este token de DistilBERT
            for j, token_start in enumerate(token_positions):
                token_end = token_start + len(tokens[j])
                
                # Si hay solapamiento, asignar la etiqueta
                if start <= token_end and end >= token_start:
                    label_ids[i] = self.tag2id[tags[j]]
                    break
        
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
        self.model = DistilBertForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)
       
    def _convert_to_bio_tags(self, text: str, annotations: Dict) -> List[str]:
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
        #text_files = text_files[:10] #para realizar pruebas con pocos archivos
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
                tags = self._convert_to_bio_tags(texto, annotations)
                
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
        tokens = text.split()
        
        if len(tokens) != len(tags):
            return []
        
        # Calcular posiciones de inicio de cada token
        token_positions = []
        pos = 0
        for i, token in enumerate(tokens):
            token_positions.append(pos)
            pos += len(token) + (1 if i < len(tokens) - 1 else 0)
        
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
        Convierte entidades a etiquetas BIO.
        
        Args:
            text: Texto original.
            entities: Lista de entidades.
            
        Returns:
            Lista de etiquetas BIO.
        """
        tokens = text.split()
        tags = ['O'] * len(tokens)
        
        # Calcular posiciones de inicio de cada token
        token_positions = []
        pos = 0
        for i, token in enumerate(tokens):
            token_positions.append(pos)
            pos += len(token) + (1 if i < len(tokens) - 1 else 0)
        
        # Asignar etiquetas a tokens
        for entity_text, start, end, entity_type in entities:
            entity_tokens = entity_text.split()
            
            # Buscar los tokens que corresponden a esta entidad
            for i, token_start in enumerate(token_positions):
                if token_start == start:
                    # Encontramos el inicio de la entidad
                    tags[i] = f'B-{entity_type}'
                    
                    # Marcar los tokens restantes de la entidad
                    for j in range(1, len(entity_tokens)):
                        if i + j < len(tags):
                            tags[i + j] = f'I-{entity_type}'
                    
                    break
        
        return tags
    
    def augment_data(self, texts: List[str], all_tags: List[List[str]], 
                    num_augmentations: int = 2) -> Tuple[List[str], List[List[str]]]:
        """
        Aumenta los datos aplicando técnicas de aumentación.
        
        Args:
            texts: Lista de textos originales.
            all_tags: Lista de listas de etiquetas.
            num_augmentations: Número de versiones aumentadas a generar por texto.
            
        Returns:
            Tuple con listas aumentadas de textos y etiquetas.
        """
        # Convertir etiquetas a formato de entidades
        all_entities = [
            self.convert_tags_to_entities(text, tags)
            for text, tags in zip(texts, all_tags)
        ]
        
        # Generar datos sintéticos
        synthetic_texts, synthetic_entities = self.data_augmenter.generate_synthetic_data(
            texts, all_entities, num_augmentations=num_augmentations,
            use_parallel=True,use_threads=True, num_workers=6
        )
        
        # Filtrar datos sintéticos de baja calidad
        filtered_texts, filtered_entities = self.data_augmenter.filter_synthetic_data(
            texts, all_entities, synthetic_texts, synthetic_entities
        )
        
        # Convertir entidades a etiquetas
        synthetic_tags = [
            self.convert_entities_to_tags(text, entities)
            for text, entities in zip(filtered_texts, filtered_entities)
        ]
        
        # Combinar datos originales y sintéticos
        augmented_texts = texts + filtered_texts
        augmented_tags = all_tags + synthetic_tags
        
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
        
        # Configurar optimizador
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Configurar learning rate scheduler
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
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
        
        # Variables para early stopping
        best_val_f1 = 0
        patience = 3
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
            # Modo entrenamiento
            self.model.train()
            total_train_loss = 0
            
            for batch in train_dataloader:
                # Mover batch al dispositivo
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**{k: v for k, v in batch.items() if k != 'labels'})
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
                
                # Actualizar parámetros
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_train_loss += loss.item()
            
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
                    outputs = self.model(**{k: v for k, v in batch.items() if k != 'labels'})
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
                    
                    total_val_loss += loss.item()
                    
                    # Obtener predicciones
                    predictions = torch.argmax(logits, dim=2)
                    
                    # Recopilar predicciones y etiquetas verdaderas
                    for i in range(predictions.shape[0]):
                        for j in range(predictions.shape[1]):
                            if batch['labels'][i, j] != -100:
                                val_predictions.append(predictions[i, j].item())
                                val_true_labels.append(batch['labels'][i, j].item())
            
            # Calcular métricas
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                val_true_labels, val_predictions, average='weighted'
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
        if self.model is None or self.tokenizer is None:
            raise ValueError("El modelo o tokenizador no están cargados.")
        
        # Crear dataset temporal para predicción
        # Usamos etiquetas dummy que serán ignoradas
        dummy_tags = [['O'] * len(text.split()) for text in texts]
        dataset = SROIEDataset(texts, dummy_tags, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        # Modo evaluación
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Mover batch al dispositivo
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**{k: v for k, v in batch.items() if k != 'labels'})
                logits = outputs.logits
                
                # Obtener predicciones
                predictions = torch.argmax(logits, dim=2)
                
                # Convertir predicciones a etiquetas
                for i in range(predictions.shape[0]):
                    pred_labels = []
                    for j in range(predictions.shape[1]):
                        if batch['attention_mask'][i, j] == 1 and j > 0:  # Ignorar [CLS]
                            pred_idx = predictions[i, j].item()
                            if pred_idx in dataset.id2tag:
                                pred_labels.append(dataset.id2tag[pred_idx])
                    
                    # Recortar a la longitud del texto original
                    text_tokens = texts[len(all_predictions)].split()
                    pred_labels = pred_labels[:len(text_tokens)]
                    
                    # Rellenar si es necesario
                    if len(pred_labels) < len(text_tokens):
                        pred_labels.extend(['O'] * (len(text_tokens) - len(pred_labels)))
                    
                    all_predictions.append(pred_labels)
        
        return all_predictions


# Ejemplo de uso
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

