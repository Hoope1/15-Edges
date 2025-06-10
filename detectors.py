import argparse
import gzip
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import requests
import torch

BASE_DIR   = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
HED_DIR    = os.path.join(MODEL_DIR, 'hed')
STRUCT_DIR = os.path.join(MODEL_DIR, 'structured')

# Alternative URLs für HED (fallback wenn erste nicht funktioniert)
HED_PROTO_URLS = [
    'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/hed_edge_detection/deploy.prototxt',
    'https://raw.githubusercontent.com/ashukid/hed-edge-detector/master/deploy.prototxt'
]
HED_WEIGHTS_URLS = [
    'https://github.com/ashukid/hed-edge-detector/raw/master/hed_pretrained_bsds.caffemodel',
    'https://drive.google.com/uc?id=1zc-tSjrZ1Q1q6hzYNDaLdgBCcCRFjYLF'  # Backup Google Drive
]
STRUCT_URL = 'https://github.com/opencv/opencv_extra/raw/master/testdata/cv/ximgproc/model.yml.gz'

# ------------------------------------------------------
# Helper‑Download
# ------------------------------------------------------

def _download_with_fallback(urls: list, dst: str) -> bool:
    """Download mit mehreren Fallback-URLs"""
    if os.path.exists(dst):
        return True
    
    for i, url in enumerate(urls):
        try:
            print(f"[download] {os.path.basename(dst)} (URL {i+1}/{len(urls)})")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(dst, 'wb') as fh:
                fh.write(r.content)
            return True
        except Exception as e:
            print(f"[failed] URL {i+1} fehlgeschlagen: {e}")
            if i < len(urls) - 1:
                print(f"[retry] Versuche nächste URL...")
    
    print(f"[error] Alle URLs für {os.path.basename(dst)} fehlgeschlagen")
    return False

def _download(url: str, dst: str) -> None:
    """Einzelner Download (für rückwärtskompatibilität)"""
    _download_with_fallback([url], dst)

# ------------------------------------------------------
# Helper für einheitliche Bildverarbeitung
# ------------------------------------------------------

def standardize_output(edge_map: np.ndarray, target_size: tuple = None, invert: bool = True) -> np.ndarray:
    """
    Standardisiert die Ausgabe aller Edge Detection Methoden:
    - Invertiert Farben (weiße Hintergründe, schwarze Linien)
    - Skaliert auf einheitliche Auflösung
    - Konvertiert zu uint8 Format
    """
    # Normalisiere auf 0-255 Range
    if edge_map.dtype != np.uint8:
        if edge_map.max() <= 1.0:
            edge_map = (edge_map * 255).astype(np.uint8)
        else:
            edge_map = edge_map.astype(np.uint8)
    
    # Invertiere wenn gewünscht (dunkle Linien auf hellem Hintergrund)
    if invert:
        edge_map = 255 - edge_map
    
    # Skaliere auf Zielgröße falls angegeben
    if target_size is not None:
        edge_map = cv2.resize(edge_map, target_size, interpolation=cv2.INTER_CUBIC)
    
    return edge_map

def get_max_resolution(image_paths: list) -> tuple:
    """Findet die maximale Auflösung aller Eingabebilder"""
    max_width, max_height = 0, 0
    
    for path in image_paths:
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                max_width = max(max_width, w)
                max_height = max(max_height, h)
    
    return (max_width, max_height) if max_width > 0 else (1920, 1080)

# ------------------------------------------------------
# Modelle initialisieren (HED, Structured, optional BDCN, pytorch-hed)
# ------------------------------------------------------

def init_models() -> None:
    os.makedirs(HED_DIR, exist_ok=True)
    os.makedirs(STRUCT_DIR, exist_ok=True)

    # HED Modell mit Fallbacks
    proto_path = os.path.join(HED_DIR, 'deploy.prototxt')
    weights_path = os.path.join(HED_DIR, 'hed.caffemodel')
    
    proto_success = _download_with_fallback(HED_PROTO_URLS, proto_path)
    weights_success = _download_with_fallback(HED_WEIGHTS_URLS, weights_path)
    
    if not proto_success or not weights_success:
        print("[warning] HED Modell Download fehlgeschlagen - HED wird nicht verfügbar sein")

    # Structured Forests
    gz_path = os.path.join(STRUCT_DIR, 'model.yml.gz')
    yml_path = os.path.join(STRUCT_DIR, 'model.yml')
    _download(STRUCT_URL, gz_path)
    if not os.path.exists(yml_path) and os.path.exists(gz_path):
        try:
            with gzip.open(gz_path, 'rb') as fi, open(yml_path, 'wb') as fo:
                fo.write(fi.read())
            print('[unpack] Structured Forests Modell entpackt')
        except Exception as e:
            print(f'[error] Entpacken fehlgeschlagen: {e}')

    # Versuche pytorch-hed zu installieren
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pytorch-hed'], 
                      check=True, capture_output=True)
        print('[success] pytorch-hed erfolgreich installiert')
    except subprocess.CalledProcessError:
        print('[warning] pytorch-hed Installation fehlgeschlagen')

    # Optional: BDCN Repo (Klon + Installation) - VERBESSERT
    bdcn_repo = os.path.join(BASE_DIR, 'bdcn_repo')
    if not os.path.isdir(bdcn_repo):
        print('[clone] BDCN‑Repo …')
        try:
            subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/YacobBY/bdcn.git', bdcn_repo], check=True)
            
            # Erstelle eine modifizierte requirements.txt ohne problematische Versionen
            original_req = os.path.join(bdcn_repo, 'requirements.txt')
            temp_req = os.path.join(bdcn_repo, 'requirements_fixed.txt')
            
            if os.path.exists(original_req):
                with open(original_req, 'r') as f:
                    lines = f.readlines()
                
                # Filtere problematische Pakete und Versionen
                fixed_lines = []
                skip_packages = {'numpy', 'torch', 'torchvision', 'opencv-python', 'opencv-contrib-python'}
                
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    package_name = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                    
                    if package_name.lower() not in skip_packages:
                        if 'matplotlib' in package_name.lower():
                            fixed_lines.append('matplotlib>=3.1.0')
                        elif 'pillow' in package_name.lower():
                            fixed_lines.append('pillow')
                        else:
                            fixed_lines.append(line)
                
                with open(temp_req, 'w') as f:
                    f.write('\n'.join(fixed_lines))
                
                print('[fix] Verwende modifizierte requirements.txt für BDCN')
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', temp_req], check=True)
            
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', bdcn_repo, '--no-deps'], check=True)
                print('[success] BDCN erfolgreich installiert')
            except subprocess.CalledProcessError:
                print('[warning] BDCN Paket-Installation fehlgeschlagen, aber Repository verfügbar')
                
        except subprocess.CalledProcessError as e:
            print(f"[warning] BDCN Installation fehlgeschlagen: {e}")
        except FileNotFoundError:
            print("[warning] Git nicht gefunden - BDCN wird übersprungen")

# ------------------------------------------------------
# Edge‑Methoden - Erweitert mit neuen Algorithmen
# ------------------------------------------------------

def run_hed(path: str, target_size: tuple = None) -> np.ndarray:
    """Original HED mit OpenCV DNN"""
    proto  = os.path.join(HED_DIR, 'deploy.prototxt')
    weight = os.path.join(HED_DIR, 'hed.caffemodel')
    
    if not os.path.exists(proto) or not os.path.exists(weight):
        raise RuntimeError("HED Modell nicht verfügbar - Download fehlgeschlagen")
    
    net = cv2.dnn.readNetFromCaffe(proto, weight)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    H, W = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (W, H), (104.00699, 116.66877, 122.67891), False, False)
    net.setInput(blob)
    out = net.forward()[0, 0]
    out = cv2.resize(out, (W, H))
    result = (out * 255).astype('uint8')
    return standardize_output(result, target_size)

def run_pytorch_hed(path: str, target_size: tuple = None) -> np.ndarray:
    """PyTorch HED Implementation - bessere Qualität"""
    try:
        import torchHED
        from PIL import Image
        
        # Lade Bild mit PIL
        pil_img = Image.open(path).convert('RGB')
        
        # Verarbeite mit pytorch-hed
        edge_pil = torchHED.process_img(pil_img)
        
        # Konvertiere zurück zu OpenCV Format
        edge_array = np.array(edge_pil)
        if len(edge_array.shape) == 3:
            edge_array = cv2.cvtColor(edge_array, cv2.COLOR_RGB2GRAY)
        
        return standardize_output(edge_array, target_size)
        
    except ImportError:
        print("[fallback] pytorch-hed nicht verfügbar, verwende Standard HED")
        return run_hed(path, target_size)

def run_structured(path: str, target_size: tuple = None) -> np.ndarray:
    """Structured Forests Edge Detection"""
    mdl = os.path.join(STRUCT_DIR, 'model.yml')
    if not os.path.exists(mdl):
        raise RuntimeError("Structured Forests Modell nicht verfügbar")
    
    ed  = cv2.ximgproc.createStructuredEdgeDetection(mdl)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    img = img.astype('float32') / 255.0
    edge = ed.detectEdges(img)
    result = (edge * 255).astype('uint8')
    return standardize_output(result, target_size)

def run_kornia_canny(path: str, target_size: tuple = None) -> np.ndarray:
    """Kornia GPU-beschleunigter Canny"""
    import kornia
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    t = torch.tensor(gray / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    edge = kornia.filters.canny(t)[0][0]
    result = (edge.numpy() * 255).astype('uint8')
    return standardize_output(result, target_size)

def run_kornia_sobel(path: str, target_size: tuple = None) -> np.ndarray:
    """Kornia Sobel Edge Detection"""
    import kornia
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    
    t = torch.tensor(gray / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Sobel X und Y
    sobel_x = kornia.filters.sobel(t, normalized=True)
    sobel_y = kornia.filters.sobel(t, normalized=True)
    
    # Magnitude berechnen
    magnitude = torch.sqrt(sobel_x**2 + sobel_y**2)
    result = (magnitude[0][0].numpy() * 255).astype('uint8')
    return standardize_output(result, target_size)

def run_laplacian(path: str, target_size: tuple = None) -> np.ndarray:
    """OpenCV Laplacian Edge Detection"""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    
    # Gaußscher Blur zur Rauschreduzierung
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    laplacian = np.absolute(laplacian)
    result = laplacian.astype(np.uint8)
    return standardize_output(result, target_size)

def run_prewitt(path: str, target_size: tuple = None) -> np.ndarray:
    """Prewitt Edge Detection"""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    
    # Prewitt Kernel
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    
    # Konvolution
    edge_x = cv2.filter2D(gray, cv2.CV_32F, prewitt_x)
    edge_y = cv2.filter2D(gray, cv2.CV_32F, prewitt_y)
    
    # Magnitude
    magnitude = np.sqrt(edge_x**2 + edge_y**2)
    result = np.clip(magnitude, 0, 255).astype(np.uint8)
    return standardize_output(result, target_size)

def run_roberts(path: str, target_size: tuple = None) -> np.ndarray:
    """Roberts Cross Edge Detection"""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    
    # Roberts Cross Kernel
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    
    # Konvolution
    edge_x = cv2.filter2D(gray, cv2.CV_32F, roberts_x)
    edge_y = cv2.filter2D(gray, cv2.CV_32F, roberts_y)
    
    # Magnitude
    magnitude = np.sqrt(edge_x**2 + edge_y**2)
    result = np.clip(magnitude, 0, 255).astype(np.uint8)
    return standardize_output(result, target_size)

def run_scharr(path: str, target_size: tuple = None) -> np.ndarray:
    """Scharr Edge Detection - verbesserte Sobel Variante"""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    
    # Scharr X und Y
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    
    # Magnitude
    magnitude = np.sqrt(scharr_x**2 + scharr_y**2)
    result = np.clip(magnitude, 0, 255).astype(np.uint8)
    return standardize_output(result, target_size)

def run_gradient_magnitude(path: str, target_size: tuple = None) -> np.ndarray:
    """Gradient Magnitude mit mehreren Methoden kombiniert"""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    
    # Verschiedene Gradienten berechnen
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    
    # Kombiniere Magnitudes
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    scharr_mag = np.sqrt(scharr_x**2 + scharr_y**2)
    
    # Gewichteter Durchschnitt
    combined = 0.6 * sobel_mag + 0.4 * scharr_mag
    result = np.clip(combined, 0, 255).astype(np.uint8)
    return standardize_output(result, target_size)

def run_bdcn(path: str, target_size: tuple = None) -> np.ndarray:
    """BDCN Edge Detection mit Fallback"""
    try:
        from bdcn_edge import BDCNEdgeDetector
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Bild konnte nicht geladen werden: {path}")
        edge = BDCNEdgeDetector().detect(img)
        result = (edge * 255).astype('uint8')
        return standardize_output(result, target_size)
    except ImportError:
        # Fallback: Verwende eine Kombination aus Canny und morphologischen Operationen
        print("[fallback] BDCN nicht verfügbar, verwende Canny-basierte Alternative")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Bild konnte nicht geladen werden: {path}")
        
        # Gaussianisches Rauschen reduzieren
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Canny Edge Detection mit optimierten Parametern
        edges = cv2.Canny(img_blur, 50, 150)
        
        # Morphologische Operationen für bessere Kantenkontinuität
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)
        
        return standardize_output(edges, target_size)

def run_fixed_cnn(path: str, target_size: tuple = None) -> np.ndarray:
    """Fixed CNN Edge Detector (Sobel-basiert)"""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    k = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
    cx = torch.nn.Conv2d(1,1,3,padding=1,bias=False)
    cx.weight.data = k.unsqueeze(0).unsqueeze(0)
    cy = torch.nn.Conv2d(1,1,3,padding=1,bias=False)
    cy.weight.data = k.t().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        t = torch.tensor(gray / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        e = torch.sqrt(cx(t) ** 2 + cy(t) ** 2).squeeze().numpy()
    result = (e * 255).astype('uint8')
    return standardize_output(result, target_size)

def run_multi_scale_canny(path: str, target_size: tuple = None) -> np.ndarray:
    """Multi-Scale Canny - kombiniert verschiedene Skalen"""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    
    # Verschiedene Blur-Level für Multi-Scale
    edges_combined = np.zeros_like(gray, dtype=np.float32)
    
    # Scale 1: Feine Details
    blur1 = cv2.GaussianBlur(gray, (3, 3), 0.5)
    edges1 = cv2.Canny(blur1, 50, 150)
    edges_combined += edges1.astype(np.float32) * 0.4
    
    # Scale 2: Mittlere Details
    blur2 = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges2 = cv2.Canny(blur2, 30, 100)
    edges_combined += edges2.astype(np.float32) * 0.4
    
    # Scale 3: Grobe Strukturen
    blur3 = cv2.GaussianBlur(gray, (7, 7), 1.5)
    edges3 = cv2.Canny(blur3, 20, 80)
    edges_combined += edges3.astype(np.float32) * 0.2
    
    result = np.clip(edges_combined, 0, 255).astype(np.uint8)
    return standardize_output(result, target_size)

def run_adaptive_canny(path: str, target_size: tuple = None) -> np.ndarray:
    """Adaptive Canny - passt Schwellwerte automatisch an"""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    
    # Berechne adaptive Schwellwerte basierend auf Histogramm
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Median als Basis für adaptive Schwellwerte
    median = np.median(blur)
    sigma = 0.33  # Faktor für Schwellwert-Berechnung
    
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    
    edges = cv2.Canny(blur, lower, upper)
    return standardize_output(edges, target_size)

def run_morphological_gradient(path: str, target_size: tuple = None) -> np.ndarray:
    """Morphological Gradient Edge Detection"""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    
    # Verschiedene Kernel-Größen für Multi-Scale
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Morphological Gradient
    gradient_small = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_small)
    gradient_medium = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_medium)
    
    # Kombiniere verschiedene Skalen
    combined = cv2.addWeighted(gradient_small, 0.7, gradient_medium, 0.3, 0)
    return standardize_output(combined, target_size)

# ------------------------------------------------------
# Alle verfügbaren Methoden definieren
# ------------------------------------------------------

def get_all_methods():
    """Gibt alle verfügbaren Edge Detection Methoden zurück"""
    return [
        ('HED_OpenCV', run_hed),
        ('HED_PyTorch', run_pytorch_hed),
        ('StructuredForests', run_structured),
        ('Kornia_Canny', run_kornia_canny),
        ('Kornia_Sobel', run_kornia_sobel),
        ('Laplacian', run_laplacian),
        ('Prewitt', run_prewitt),
        ('Roberts', run_roberts),
        ('Scharr', run_scharr),
        ('GradientMagnitude', run_gradient_magnitude),
        ('BDCN', run_bdcn),
        ('FixedCNN', run_fixed_cnn),
        ('MultiScaleCanny', run_multi_scale_canny),
        ('AdaptiveCanny', run_adaptive_canny),
        ('MorphologicalGradient', run_morphological_gradient),
    ]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--init-models', action='store_true')
    p.add_argument('--list-methods', action='store_true', help='Liste alle verfügbaren Methoden auf')
    args = p.parse_args()
    
    if args.init_models:
        init_models()
    elif args.list_methods:
        print("Verfügbare Edge Detection Methoden:")
        methods = get_all_methods()
        for i, (name, _) in enumerate(methods, 1):
            print(f"{i:2d}. {name}")
    else:
        print("Verwende --init-models zum Initialisieren oder --list-methods zum Auflisten aller Methoden")
