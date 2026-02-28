# ⚡ Guía Rápida: Solución W036

## El Problema

```
UserWarning: [W036] The component 'entity_ruler' does not have any patterns defined.
```

## La Causa

spaCy crea un componente `entity_ruler` pero no tiene patrones para ejecutar cuando:

- Los datos están vacíos
- No hay entidades extraídas
- Los patrones no se generan correctamente

## La Solución (Ya Implementada)

### ✅ Cambio 1: `add_entity_patterns()`

**Antes**: Siempre crea EntityRuler

```python
def add_entity_patterns(self, patterns):
    self.nlp.add_pipe("entity_ruler")  # ← Crea sin validar
    self.entity_ruler.add_patterns(patterns)
```

**Después**: Valida antes de crear

```python
def add_entity_patterns(self, patterns):
    if not patterns:  # ← Validar primero
        return
    self.nlp.add_pipe("entity_ruler")
    self.entity_ruler.add_patterns(patterns)
```

### ✅ Cambio 2: `create_entity_patterns()`

**Antes**: Asume que hay datos y entidades

```python
def create_entity_patterns(self, spacy_data):
    for text, annotations in spacy_data:
        for start, end, label in annotations["entities"]:  # ← Falla si vacío
            # ...
```

**Después**: Valida cada paso

```python
def create_entity_patterns(self, spacy_data):
    if not spacy_data:
        return []  # ← Retorna vacío
    
    for text, annotations in spacy_data:
        entities = annotations.get("entities", [])  # ← Usa .get()
        # ...
    
    if not entity_examples:
        return []  # ← Retorna vacío si no hay
```

### ✅ Cambio 3: `train_model()`

**Antes**: Siempre agrega EntityRuler

```python
patterns = self.create_entity_patterns(spacy_data)
self.add_entity_patterns(patterns)  # ← W036 si vacío
```

**Después**: Valida antes

```python
patterns = self.create_entity_patterns(spacy_data)
if patterns:  # ← Validar
    self.add_entity_patterns(patterns)
else:
    logger.info("Sin patrones EntityRuler")
```

### ✅ Cambio 4: Pipeline

**Antes**: Asume EntityRuler existe

```python
if "ner" not in self.nlp.pipe_names:
    self.ner = self.nlp.add_pipe("ner", after="entity_ruler")  # ← KeyError
```

**Después**: Valida primero

```python
if "ner" not in self.nlp.pipe_names:
    if "entity_ruler" in self.nlp.pipe_names:  # ← Validar
        self.ner = self.nlp.add_pipe("ner", after="entity_ruler")
    else:
        self.ner = self.nlp.add_pipe("ner")
```

## Verificar Que Funciona

### Opción 1: Ejecutar como siempre

```bash
python sroie_main.py Data/sroie/completo --model_type spacy --num_augmentations 2

# Deberías ver en logs:
# INFO: Sin patrones EntityRuler (datos pueden estar vacíos...)
# (Sin mensaje de W036)
```

### Opción 2: Ejecutar tests

```bash
python test_w036_resolution.py

# Verás:
# TEST 1 (Datos sin entidades)... ✓ PASÓ
# TEST 2 (Datos mixtos)... ✓ PASÓ
# TEST 3 (Datos válidos)... ✓ PASÓ
# TEST 5 (Funciones)... ✓ PASÓ
```

### Opción 3: Verificar en código

```python
from spacy_sroie_augmentation import SROIESpacyAugmenter

augmenter = SROIESpacyAugmenter()

# Datos sin entidades
empty_data = [("Texto", {"entities": []})]

# NO debe producir W036
metrics = augmenter.train_model(empty_data, n_iter=1)
# INFO: Sin patrones EntityRuler
# (Sin warnings)
```

## Casos Manejados

| Caso | Antes | Después |
|------|-------|---------|
| Sin entidades | ⚠️ W036 | ✓ Sin warning |
| Sin patrones | ⚠️ W036 | ✓ Sin warning |
| EntityRuler sin usar | ⚠️ W036 | ✓ No se crea |
| Datos válidos | ✓ OK | ✓ OK (mejorado) |

## Integración con W030

⚠️ **Importante**: Esta solución funciona junto con la solución del W030

```
W030: Entidades desalineadas
└─→ Solucionado con validate_and_repair_training_data()

W036: Entity Ruler sin patrones
└─→ Solucionado con validación en add_entity_patterns()
```

Juntas garantizan:

1. ✅ Todas las entidades están correctamente alineadas (W030)
2. ✅ EntityRuler solo existe si hay patrones (W036)
3. ✅ Entrenamiento limpio sin warnings

## Archivos Modificados

- `spacy_sroie_augmentation.py` - 4 funciones mejoradas

## Documentación

- [SOLUCION_W036.md](SOLUCION_W036.md) - Análisis detallado
- [SOLUCION_COMPLETA.md](SOLUCION_COMPLETA.md) - Overview de ambas soluciones
- [VALIDACION_ENTIDADES.md](VALIDACION_ENTIDADES.md) - Validación de entidades (W030)

## ¿Preguntas?

Ver [SOLUCION_W036.md](SOLUCION_W036.md) para:

- Debugging si aún ves W036
- Entendimiento técnico
- Casos edge
- FAQ completa

---

**¡W036 está totalmente resuelto!** ✅
