#!/usr/bin/env python3
"""
Script para diagnosticar problemas de rutas de archivo
"""
import os
import sys
import json

# Configurar UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def check_paths():
    """Verificar todas las rutas críticas"""
    print("[*] Verificando rutas de archivo...\n")
    
    paths_to_check = {
        'Directorio actual': os.getcwd(),
        'Directorio Data': 'Data',
        'Directorio Data/sroie': 'Data/sroie',
        'Directorio Data/sroie/completo': 'Data/sroie/completo',
        'Directorio output': 'output',
        'Archivo spacy_loaded.json': 'output/spacy_loaded.json',
    }
    
    print("[-] Estado de las rutas:")
    print("-" * 60)
    
    for name, path in paths_to_check.items():
        full_path = os.path.abspath(path)
        exists = os.path.exists(path)
        is_file = os.path.isfile(path) if exists else False
        is_dir = os.path.isdir(path) if exists else False
        
        status = "[OK]" if exists else "[ERROR]"
        type_str = "ARCHIVO" if is_file else ("CARPETA" if is_dir else "NO EXISTE")
        
        print(f"{status} {name:<35} {type_str:<10}")
        print(f"     Ruta: {full_path}")
        
        if exists and is_dir:
            try:
                items = os.listdir(path)
                print(f"     Contenido: {len(items)} items")
                if len(items) <= 5:
                    for item in items[:5]:
                        print(f"       - {item}")
            except PermissionError:
                print(f"     [PERMISO DENEGADO]")
        print()
    
    # Verificar si spacy_loaded.json es accesible
    print("\n[-] Verificando acceso a spacy_loaded.json:")
    print("-" * 60)
    json_path = 'output/spacy_loaded.json'
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"[OK] Archivo accesible")
            print(f"     Registros: {len(data)}")
            print(f"     Tamaño: {os.path.getsize(json_path) / (1024*1024):.2f} MB")
        except Exception as e:
            print(f"[ERROR] No se puede leer el archivo: {e}")
    else:
        print(f"[ERROR] Archivo no encontrado")
    
    print()

if __name__ == '__main__':
    check_paths()
