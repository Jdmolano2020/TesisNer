# Guía de Validación y Corrección de Alineamiento de Entidades

## Problema

Cuando entrenas un modelo spaCy con datos que tienen entidades desalineadas, recibes advertencias como:

```
UserWarning: [W030] Some entities could not be aligned in the text "RESTORAN WAN [UNK] NO.2, JALAN..." 
with entities "[(19, 104, 'address'), (167, 177, 'date')...]". 
Misaligned entities ('-') will be ignored during training.
```

Esto significa que los offsets (start, end) de las entidades NO coinciden exactamente con los caracteres del texto.

## Causas Comunes

1. **Normalización de texto**: El texto se procesa (se limpian espacios, caracteres especiales) pero los offsets no se ajustan
2. **Caracteres especiales**: `[UNK]`, caracteres acentuados, emojis pueden cambiar la longitud del texto
3. **Espacios**: Múltiples espacios se reducen a uno, desplazando los offsets posteriores
4. **Codificación**: Diferencias en cómo se encodifican caracteres especiales

## Solución Implementada

He añadido tres funciones principales a `spaco_sroie_augmentation.py`:

### 1. `validate_entity_alignment(text, entities)`

Valida si las entidades están correctamente alineadas usando el método oficial de spaCy.

```python
augmenter = SROIESpacyAugmenter()
is_valid, issues = augmenter.validate_entity_alignment(text, entities)

if not is_valid:
    print(f"Problemas encontrados: {issues}")
```

### 2. `fix_misaligned_entities(text, entities)`

Intenta corregir entidades desalineadas buscando el texto de la entidad en el documento.

```python
fixed_entities = augmenter.fix_misaligned_entities(text, entities, strict=False)
```

### 3. `validate_and_repair_training_data(spacy_data)`

Valida y repara un conjunto completo de datos de entrenamiento.

```python
repaired_data, stats = augmenter.validate_and_repair_training_data(spacy_data)
print(f"Datos válidos: {stats['valid_without_changes']}")
print(f"Datos reparados: {stats['repaired']}")
print(f"Datos eliminados: {stats['removed_invalid']}")
```

### Cambios en `train_model()`

Ahora, automáticamente llama a `validate_and_repair_training_data()` al inicio:

```python
metrics = spacy_augmenter.train_model(
    augmented_data,
    n_iter=args.n_iter,
    batch_size=args.batch_size,
    model_dir=os.path.join(args.output_dir, "spacy_model")
)
# Internamente: se validan y reparan los datos automáticamente
```

## Cómo Usar

### Opción A: Validar archivos JSON antes de entrenar

```bash
python scripts/validate_entity_alignment.py validate output/spacy_augmented_2.json

# Con muestra (para archivos grandes):
python scripts/validate_entity_alignment.py validate output/spacy_augmented_2.json --sample 100
```

Salida:

```
============================================================
REPORTE DE VALIDACIÓN
============================================================
Archivo: output/spacy_augmented_2.json
Total muestras: 1000
Muestras validadas: 100
Válidas: 95
Inválidas: 5

Tasa de validación: 95.0%
Recomendación: Revisar y reparar los datos antes del entrenamiento

Primeros problemas encontrados:
  Índice 15: RESTORAN WAN [UNK] NO.2, JALAN...
    - 3 entidades desalineadas (tags "-" encontrados)
  ...
============================================================
```

### Opción B: Reparar datos automáticamente

```bash
python scripts/validate_entity_alignment.py repair output/spacy_augmented_2.json

# Con salida específica:
python scripts/validate_entity_alignment.py repair output/spacy_augmented_2.json --output output/spacy_augmented_2_fixed.json
```

Salida:

```
============================================================
REPORTE DE REPARACIÓN
============================================================
Archivo original: output/spacy_augmented_2.json
Archivo reparado: output/spacy_augmented_2_repaired.json
Total muestras: 1000
Válidas sin cambios: 870
Reparadas: 120
Eliminadas (inválidas): 10
============================================================
```

### Opción C: Validar en código Python

```python
from scripts.validate_entity_alignment import validate_spacy_data_file

results = validate_spacy_data_file('output/spacy_augmented_2.json', sample_size=100)

print(f"Válidas: {results['valid_samples']}")
print(f"Inválidas: {results['invalid_samples']}")
print(f"Rate: {results['summary']['validation_rate']}")
```

## Ejemplo Paso a Paso

### 1. Generar datos aumentados (como haces normalmente)

```python
from sroie_main import main
# O ejecutar:
# python sroie_main.py Data/sroie/ --model_type spacy --num_augmentations 2
```

Esto genera `output/spacy_augmented_2.json`

### 2. Validar los datos generados

```bash
python scripts/validate_entity_alignment.py validate output/spacy_augmented_2.json
```

### 3. Si hay problemas, reparar

```bash
python scripts/validate_entity_alignment.py repair output/spacy_augmented_2.json --output output/spacy_augmented_2_fixed.json
```

### 4. Usar los datos reparados para entrenar manualmente

```python
import json
from spacy_sroie_augmentation import SROIESpacyAugmenter

# Cargar datos reparados
with open('output/spacy_augmented_2_fixed.json', 'r') as f:
    data = json.load(f)

# Convertir a formato interno
spacy_data = []
for item in data:
    entities = [(e['start'], e['end'], e['label']) for e in item['entities']]
    spacy_data.append((item['text'], {'entities': entities}))

# Entrenar (automáticamente valida)
augmenter = SROIESpacyAugmenter()
metrics = augmenter.train_model(spacy_data, n_iter=50, batch_size=16)
```

## Entendiendo los Tipos de Problemas

### Problema 1: Entidades Completamente Desalineadas

```
Texto: "RESTORAN WAN [UNK] NO.2"
Entidad esperada: (start=10, end=21) "WAN [UNK] NO"
Problema: No ve "WAN [UNK] NO" a partir de posición 10
```

**Solución**: El script busca el texto de la entidad en el documento y ajusta automaticamente los offsets.

### Problema 2: Espacios Normalizados

```
Texto original: "Jalan  Temenggung"    (2 espacios)
Texto normalizado: "Jalan Temenggung"  (1 espacio)
Entidades: Basadas en texto original (con 2 espacios)
```

**Solución**: Los offsets se reajustan al nuevo texto normalizado.

### Problema 3: Caracteres Unicode

```
Texto: "Café" (con é especial)
Codificación puede variar según cómo se procese
```

**Solución**: Normalización NFKC integrada en `normalize_text()`.

## Logs Detallados

Durante la validación, puedes ver logs como:

```
INFO: Removidas 5 entidades desalineadas en validación inicial
INFO: Removidas 2 entidades que no caben en texto truncado
DEBUG: Realineada entidad 'WAN [UNK] NO': [10:21] -> [10:20]
DEBUG: Realineada con normalización 'JALAN TEMENGGUNG': [50:68] -> [48:66]
```

Estos te muestran exactamente qué se está arreglando.

## Automatización Completa

El proceso ahora es completamente automático. Cuando ejecutas:

```bash
python sroie_main.py Data/sroie/ --model_type spacy --num_augmentations 2
```

Internamente:

1. ✓ Se cargan y aumentan los datos
2. ✓ Al llamar a `spacy_augmenter.train_model()`, automáticamente:
   - Se valida cada muestra de datos
   - Se reparan las que se pueden arreglar
   - Se eliminan las que están irremediablemente dañadas
   - Solo se entrena con datos válidos
3. ✓ No recibirás más advertencias `[W030]`

## Verificar Que Esto Funciona

```bash
# Ejecutar entrenamiento como normalmente
python sroie_main.py Data/sroie/ --model_type spacy --num_augmentations 2

# Si los logs muestran algo como esto, sabes que funciona:
# INFO: Validando y reparando alineamiento de entidades...
# INFO: Después de reparación: 980 ejemplos listos para entrenamiento
# (Sin más advertencias W030)
```

## Parámetros de Control

En `validate_and_repair_training_data()`:

| Parámetro | Default | Efecto |
|-----------|---------|--------|
| `remove_invalid` | `True` | Si False, mantiene los inválidos sin reparar |
| `strict` | `False` (en fix_misaligned_entities) | Si True, solo acepta búsquedas exactas |

### Ejemplos

```python
# Modo permisivo (intenta reparar todo)
repaired, stats = augmenter.validate_and_repair_training_data(
    data, remove_invalid=False
)

# Modo estricto (solo acepta alineamientos perfectos)
# (Esto está en fix_misaligned_entities)
```

## Resumen

✓ **Problema**: Entidades desalineadas → Advertencia W030 → Entidades ignoradas en training

✓ **Causa**: Offsets no coinciden con caracteres después de normalización

✓ **Solución**: Función `validate_and_repair_training_data()` integrada en `train_model()`

✓ **Resultado**:

- Datos validados antes de entrenar
- Entidades corregidas automáticamente
- Datos inválidos eliminados
- Sin más advertencias W030
- Mejor calidad de entrenamiento

## Troubleshooting

### "No hay datos válidos después de reparación"

- Significa que todas las entidades están irremediablemente desalineadas
- Verifica que los datos se generaron correctamente
- Usa `validate_spacy_data_file()` para inspeccionar

### "Demasiadas entidades eliminadas"

- Los datos pueden estar mal formados
- Usa `--sample 10` para validar una muestra pequeña
- Revisa los logs de detalles con `--verbose`

### Quiero ver exactamente qué se arregló

- Usa el script de validación con `--sample`
- Los logs muestran cada corrección realizada
- Compara el archivo original con `_repaired.json`
