#!/usr/bin/env python3
"""
Erweiterte Edge Detection Pipeline mit:
- Einheitlicher Auflösung (höchste Auflösung wird beibehalten)
- Invertierten Ausgaben (weiße Hintergründe, dunkle Linien)
- Alle Ergebnisse in einem einzigen Ordner
- Viele verschiedene Edge Detection Methoden
"""

import argparse
import os
import cv2
import glob
from pathlib import Path
from detectors import get_all_methods, get_max_resolution

def create_output_structure(output_dir: str) -> str:
    """Erstellt die Ausgabestruktur und gibt den Hauptordner zurück"""
    main_output_dir = os.path.join(output_dir, "edge_detection_results")
    os.makedirs(main_output_dir, exist_ok=True)
    return main_output_dir

def get_image_files(input_dir: str) -> list:
    """Sammelt alle Bilddateien aus dem Eingabeordner"""
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(input_dir, ext)
        image_files.extend(glob.glob(pattern))
        # Auch case-insensitive
        pattern_upper = os.path.join(input_dir, ext.upper())
        image_files.extend(glob.glob(pattern_upper))
    
    return list(set(image_files))  # Duplikate entfernen

def process_images(input_dir: str, output_dir: str, selected_methods: list = None) -> None:
    """
    Hauptverarbeitungsfunktion:
    - Findet maximale Auflösung aller Eingabebilder
    - Verarbeitet alle Bilder mit allen Methoden
    - Speichert invertierte Ergebnisse in einheitlicher Auflösung
    """
    
    # Erstelle Ausgabeordner
    main_output_dir = create_output_structure(output_dir)
    
    # Sammle alle Bilddateien
    image_files = get_image_files(input_dir)
    if not image_files:
        print(f"❌ Keine Bilddateien in {input_dir} gefunden!")
        return
    
    print(f"📁 Gefunden: {len(image_files)} Bilddateien")
    
    # Bestimme maximale Auflösung
    max_resolution = get_max_resolution(image_files)
    print(f"🎯 Ziel-Auflösung: {max_resolution[0]}x{max_resolution[1]}")
    
    # Lade alle verfügbaren Methoden
    all_methods = get_all_methods()
    
    # Filtere Methoden falls spezifiziert
    if selected_methods:
        methods_to_use = [(name, func) for name, func in all_methods 
                         if name in selected_methods]
        if not methods_to_use:
            print(f"❌ Keine der angegebenen Methoden gefunden: {selected_methods}")
            return
    else:
        methods_to_use = all_methods
    
    print(f"🔧 Verwende {len(methods_to_use)} Edge Detection Methoden:")
    for name, _ in methods_to_use:
        print(f"   • {name}")
    
    # Verarbeite jedes Bild mit jeder Methode
    total_operations = len(image_files) * len(methods_to_use)
    current_operation = 0
    
    for image_path in image_files:
        image_name = Path(image_path).stem
        print(f"\n📷 Verarbeite: {image_name}")
        
        for method_name, method_func in methods_to_use:
            current_operation += 1
            progress = (current_operation / total_operations) * 100
            
            try:
                print(f"   [{progress:5.1f}%] {method_name}...", end=" ")
                
                # Verarbeite Bild mit aktueller Methode
                # target_size wird automatisch in standardize_output verwendet
                result = method_func(image_path, target_size=max_resolution)
                
                # Speichere Ergebnis mit beschreibendem Namen
                output_filename = f"{image_name}_{method_name}.png"
                output_path = os.path.join(main_output_dir, output_filename)
                
                # Speichere als PNG für beste Qualität
                cv2.imwrite(output_path, result)
                print("✅ OK")
                
            except Exception as e:
                print(f"❌ Fehler: {e}")
                continue
    
    print(f"\n🎉 Verarbeitung abgeschlossen!")
    print(f"📁 Ergebnisse gespeichert in: {main_output_dir}")
    print(f"📊 {current_operation} Operationen durchgeführt")
    
    # Erstelle Übersichtsdatei
    create_summary_file(main_output_dir, image_files, methods_to_use, max_resolution)

def create_summary_file(output_dir: str, image_files: list, methods: list, resolution: tuple) -> None:
    """Erstellt eine Übersichtsdatei mit allen Informationen"""
    summary_path = os.path.join(output_dir, "processing_summary.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("EDGE DETECTION PROCESSING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Ziel-Auflösung: {resolution[0]}x{resolution[1]}\n")
        f.write(f"Verarbeitete Bilder: {len(image_files)}\n")
        f.write(f"Verwendete Methoden: {len(methods)}\n")
        f.write(f"Gesamt-Ausgaben: {len(image_files) * len(methods)}\n\n")
        
        f.write("EINGABEBILDER:\n")
        f.write("-" * 20 + "\n")
        for img_path in image_files:
            img_name = Path(img_path).name
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                f.write(f"• {img_name} ({w}x{h})\n")
        
        f.write("\nVERWENDETE METHODEN:\n")
        f.write("-" * 25 + "\n")
        for i, (method_name, _) in enumerate(methods, 1):
            f.write(f"{i:2d}. {method_name}\n")
        
        f.write("\nAUSGABE-FORMAT:\n")
        f.write("-" * 20 + "\n")
        f.write("• Format: PNG (verlustfrei)\n")
        f.write("• Farben: Invertiert (weiße Hintergründe, dunkle Kanten)\n")
        f.write("• Auflösung: Einheitlich auf höchste Eingabe-Auflösung skaliert\n")
        f.write("• Namensschema: {originalname}_{methode}.png\n")
    
    print(f"📄 Zusammenfassung erstellt: {summary_path}")

def list_available_methods():
    """Zeigt alle verfügbaren Methoden an"""
    methods = get_all_methods()
    print("\n🔧 VERFÜGBARE EDGE DETECTION METHODEN:")
    print("=" * 50)
    
    categories = {
        "Klassische Methoden": [
            "Laplacian", "Prewitt", "Roberts", "Scharr", 
            "GradientMagnitude", "MorphologicalGradient"
        ],
        "Canny Varianten": [
            "Kornia_Canny", "MultiScaleCanny", "AdaptiveCanny"
        ],
        "Deep Learning": [
            "HED_OpenCV", "HED_PyTorch", "StructuredForests", 
            "BDCN", "FixedCNN"
        ],
        "GPU-Beschleunigt": [
            "Kornia_Canny", "Kornia_Sobel"
        ]
    }
    
    for category, method_names in categories.items():
        print(f"\n📂 {category}:")
        for name, _ in methods:
            if name in method_names:
                print(f"   • {name}")

def main():
    parser = argparse.ArgumentParser(
        description="Erweiterte Edge Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python run_edge_detectors.py --input_dir images --output_dir results
  python run_edge_detectors.py --input_dir images --output_dir results --methods HED_PyTorch Kornia_Canny
  python run_edge_detectors.py --list-methods
        """
    )
    
    parser.add_argument('--input_dir', type=str, help='Eingabeordner mit Bildern')
    parser.add_argument('--output_dir', type=str, help='Ausgabeordner für Ergebnisse')
    parser.add_argument('--methods', nargs='+', help='Spezifische Methoden auswählen (optional)')
    parser.add_argument('--list-methods', action='store_true', help='Alle verfügbaren Methoden auflisten')
    
    args = parser.parse_args()
    
    if args.list_methods:
        list_available_methods()
        return
    
    if not args.input_dir or not args.output_dir:
        print("❌ Fehler: --input_dir und --output_dir sind erforderlich")
        parser.print_help()
        return
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Fehler: Eingabeordner '{args.input_dir}' existiert nicht")
        return
    
    print("🚀 ERWEITERTE EDGE DETECTION PIPELINE")
    print("=" * 50)
    print(f"📁 Eingabe: {args.input_dir}")
    print(f"📁 Ausgabe: {args.output_dir}")
    
    if args.methods:
        print(f"🎯 Ausgewählte Methoden: {', '.join(args.methods)}")
    else:
        print("🎯 Verwende alle verfügbaren Methoden")
    
    process_images(args.input_dir, args.output_dir, args.methods)

if __name__ == '__main__':
    main()
