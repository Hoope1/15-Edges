# AGENTS.md

## Projektübersicht

Dies ist ein professionelles Edge Detection Tool mit 15+ Algorithmen und moderner Streamlit GUI. Das Tool unterstützt Batch-Verarbeitung, Live-Vorschau und automatische Modell-Downloads. Alle Ergebnisse werden in einheitlicher Auflösung mit invertierten Farben (weiße Hintergründe, dunkle Kanten) ausgegeben.

**Kern-Features:**
- 15+ Edge Detection Algorithmen (Canny, Sobel, Prewitt, Roberts, Laplacian, HED, StructuredForests, etc.)
- Streamlit GUI mit 5 Tabs: Bildauswahl → Methoden → Einstellungen → Verarbeitung → Vorschau
- Automatische Modell-Downloads und venv-Setup über run.bat
- Batch-Verarbeitung mit Progress-Tracking und ETA
- ZIP-Download aller Ergebnisse
- Drag & Drop Datei-Upload
- Live-Vorschau für Algorithmus-Tests

**Technologie-Stack:**
- Python 3.9+
- Streamlit für GUI
- OpenCV 4.5+ für Bildverarbeitung
- PyTorch für Deep Learning Modelle
- Kornia für GPU-beschleunigte Algorithmen

## Code-Stil

### Python Standards
- **Formatter**: Black mit 88 Zeichen Zeilenlänge: `black --line-length 88 .`
- **Import-Sortierung**: isort mit Black-Profil: `isort --profile black .`
- **Linting**: flake8 mit max-line-length 88
- **Naming**: snake_case für Funktionen/Variablen, PascalCase für Klassen
- **Encoding**: UTF-8 für alle Dateien, deutsche Kommentare erlaubt

### Dokumentation und Kommentare
- **Docstrings**: Google-Style für alle öffentlichen Funktionen
- **Deutsche Kommentare**: Erlaubt und erwünscht für bessere Verständlichkeit
- **Inline-Kommentare**: Sparsam verwenden, nur bei komplexer Logik
- **README**: Immer aktuell halten, deutsche Sprache verwenden

### Type Hints für Computer Vision
```python
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from numpy.typing import NDArray
import cv2

# Bild-Typen
ImageArray = NDArray[np.uint8]
GrayscaleImage = NDArray[np.uint8]
EdgeMap = NDArray[np.uint8]

# Beispiel-Funktion mit korrekten Type Hints
def detect_edges(
    image: ImageArray, 
    method: str, 
    params: Dict[str, Union[int, float]] = None
) -> EdgeMap:
    """
    Führt Edge Detection mit der angegebenen Methode durch.
    
    Args:
        image: Eingabebild als uint8 numpy array
        method: Name des Edge Detection Algorithmus
        params: Parameter-Dictionary für den Algorithmus
        
    Returns:
        Binäres Edge-Map als uint8 numpy array (0 oder 255)
    """
```

### Dateistruktur-Konventionen
- **Modulare Aufteilung**: Jede Hauptfunktion in eigenem Modul
- **Klare Namensgebung**: Dateinamen beschreiben Funktionalität
- **Keine zirkulären Imports**: Abhängigkeiten klar strukturieren
- **Config-Trennung**: Konfiguration separat von Implementierung

## Testing

### Testabdeckung und -struktur
```python
# Teststruktur für Edge Detection Algorithmen
import pytest
import numpy as np
import cv2
from detectors import run_canny, run_sobel, run_hed_pytorch

class TestEdgeDetection:
    @pytest.fixture
    def sample_image(self):
        """Erstellt Testbild mit bekannten Kanten."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[30:70, 30:70] = 255  # Weißes Quadrat
        return image
    
    @pytest.fixture
    def sample_color_image(self):
        """Erstellt Farbtestbild."""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_canny_basic_functionality(self, sample_image):
        """Testet grundlegende Canny-Funktionalität."""
        result = run_canny(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.shape == sample_image.shape
        assert result.max() == 255  # Weiße Kanten nach Invertierung
        assert result.min() == 0    # Schwarzer Hintergrund
    
    @pytest.mark.parametrize("method_name", [
        "Canny", "Sobel", "Prewitt", "Roberts", "Laplacian"
    ])
    def test_all_classical_methods(self, sample_image, method_name):
        """Testet alle klassischen Edge Detection Methoden."""
        # Dynamischer Import basierend auf Methodenname
        func_name = f"run_{method_name.lower()}"
        from detectors import globals
        method_func = globals().get(func_name)
        
        if method_func:
            result = method_func(sample_image)
            assert isinstance(result, np.ndarray)
            assert result.shape == sample_image.shape
    
    def test_output_inversion(self, sample_image):
        """Testet, dass Ausgabe korrekt invertiert ist."""
        result = run_canny(sample_image)
        
        # Prüfe, dass Hintergrund weiß und Kanten dunkel sind
        center_value = result[50, 50]  # Zentrum (sollte Hintergrund sein)
        edge_value = result[30, 30]    # Kantenbereich
        
        assert center_value == 255     # Weißer Hintergrund
        assert edge_value < center_value  # Dunkle Kanten
```

### Performance Tests
```python
import time
import psutil

def test_processing_speed():
    """Testet Verarbeitungsgeschwindigkeit aller Methoden."""
    large_image = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
    
    speed_results = {}
    
    for method in ["canny", "sobel", "prewitt"]:
        start_time = time.time()
        func = globals()[f"run_{method}"]
        result = func(large_image)
        processing_time = time.time() - start_time
        
        speed_results[method] = processing_time
        
        # Performance-Anforderung: Unter 2 Sekunden für 1K Bild
        assert processing_time < 2.0, f"{method} zu langsam: {processing_time:.2f}s"
    
    print(f"Performance-Ergebnisse: {speed_results}")

def test_memory_usage():
    """Testet Speicherverbrauch während Verarbeitung."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Verarbeite mehrere große Bilder
    for i in range(5):
        large_image = np.random.randint(0, 255, (2048, 2048), dtype=np.uint8)
        result = run_canny(large_image)
        del large_image, result
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Speicher-Anforderung: Weniger als 500MB Anstieg
    assert memory_increase < 500 * 1024 * 1024, f"Memory leak: {memory_increase / 1024**2:.1f}MB"
```

### GUI Tests für Streamlit
```python
from streamlit.testing.v1 import AppTest
import tempfile
import os

def test_streamlit_app_startup():
    """Testet, dass Streamlit App korrekt startet."""
    app = AppTest.from_file("streamlit_app.py")
    app.run()
    
    assert not app.exception
    assert "Edge Detection Studio" in app.title[0].value

def test_file_upload_workflow():
    """Testet Datei-Upload und Verarbeitung."""
    app = AppTest.from_file("streamlit_app.py")
    app.run()
    
    # Erstelle Testbild
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, test_image)
        
        # Upload in App
        app.file_uploader("upload").upload_from_path(tmp.name).run()
        
        # Prüfe, dass Upload erfolgreich
        assert not app.error
        
        os.unlink(tmp.name)

def test_method_selection():
    """Testet Methodenauswahl in GUI."""
    app = AppTest.from_file("streamlit_app.py")
    app.run()
    
    # Wähle Canny-Methode
    app.selectbox("method").select("Canny").run()
    assert app.selectbox("method").value == "Canny"
    
    # Prüfe, dass Parameter-Controls erscheinen
    assert len(app.slider) >= 2  # Mindestens zwei Threshold-Slider
```

## Build und Entwicklung

### Entwicklungsumgebung Setup
```batch
REM run.bat - Automatisches Setup und Start
@echo off
echo 🎨 Edge Detection Studio Setup
echo ================================

REM Python-Check
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python nicht gefunden! Bitte von https://python.org installieren
    pause & exit /b 1
)

REM Virtuelle Umgebung
if not exist venv (
    echo 📦 Erstelle virtuelle Umgebung...
    python -m venv venv
)

echo 🔄 Aktiviere virtuelle Umgebung...
call venv\Scripts\activate

REM Dependencies installieren
echo 📚 Installiere Abhängigkeiten...
pip install --upgrade pip --quiet
pip install -r requirements.txt

REM Modelle herunterladen
echo 🤖 Lade Edge Detection Modelle...
python detectors.py --init-models

REM Ordnerstruktur erstellen
if not exist images mkdir images
if not exist results mkdir results
if not exist models mkdir models

echo ✅ Setup abgeschlossen!
echo 🚀 Starte Edge Detection Studio...
streamlit run streamlit_app.py --server.headless false
```

### Build-Kommandos
```makefile
# Makefile für Entwicklung

.PHONY: install dev test format lint clean docker

install:
	python -m venv venv
	venv/Scripts/activate && pip install -r requirements.txt

dev:
	venv/Scripts/activate && streamlit run streamlit_app.py

test:
	venv/Scripts/activate && pytest tests/ -v

test-coverage:
	venv/Scripts/activate && pytest tests/ --cov=. --cov-report=html

format:
	venv/Scripts/activate && black --line-length 88 .
	venv/Scripts/activate && isort --profile black .

lint:
	venv/Scripts/activate && flake8 --max-line-length=88 .
	venv/Scripts/activate && mypy *.py

clean:
	rmdir /s /q venv
	rmdir /s /q __pycache__
	rmdir /s /q .pytest_cache

docker:
	docker build -t edge-detection-tool .
	docker run -p 8501:8501 edge-detection-tool
```

### Dependencies Management
```txt
# requirements.txt - Produktions-Dependencies

# GUI Framework
streamlit>=1.28.0
plotly>=5.15.0                # Für interaktive Diagramme

# Computer Vision Basis
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0  # Für StructuredForests

# Deep Learning
torch>=2.0.0                  # CPU-Version, GPU-User sollten CUDA-Version wählen
torchvision>=0.15.0

# GPU-beschleunigte Verarbeitung
kornia>=0.7.0                 # GPU-beschleunigte Edge Detection

# Edge Detection Spezialist-Pakete
pytorch-hed                   # HED Implementation

# Basis-Libraries
numpy>=1.24.0
pillow>=10.0.0
requests>=2.31.0

# Performance und Monitoring
psutil>=5.9.0                 # Für Memory/CPU Monitoring
tqdm>=4.65.0                  # Progress Bars

# Entwicklung (nur in requirements-dev.txt)
# pytest>=7.4.0
# black>=23.7.0
# isort>=5.12.0
# flake8>=6.0.0
# mypy>=1.5.0
```

## Streamlit-spezifische Richtlinien

### App-Struktur und State Management
```python
import streamlit as st
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class AppState:
    """Zentraler App-Zustand."""
    selected_images: List[str]
    selected_methods: List[str]
    processing_status: str
    results: Dict[str, Any]
    settings: Dict[str, Any]

def init_session_state():
    """Initialisiert Session State mit Standardwerten."""
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState(
            selected_images=[],
            selected_methods=["Canny", "Sobel"],
            processing_status="idle",
            results={},
            settings={
                "output_dir": "./results",
                "invert_colors": True,
                "uniform_resolution": True
            }
        )

def main():
    """Haupt-App-Funktion."""
    st.set_page_config(
        page_title="🎨 Edge Detection Studio",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    render_app()

def render_app():
    """Rendert die Haupt-App-Oberfläche."""
    st.markdown("# 🎨 Edge Detection Studio")
    st.markdown("### Professionelle Kantenerkennung mit 15+ Algorithmen")
    
    # Sidebar für Hauptsteuerung
    with st.sidebar:
        render_sidebar_controls()
    
    # Hauptbereich mit Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📷 Bildauswahl", "🔧 Methoden", "⚙️ Einstellungen", 
        "🚀 Verarbeitung", "👁️ Vorschau"
    ])
    
    with tab1:
        render_image_selection()
    with tab2:
        render_method_selection()
    with tab3:
        render_settings()
    with tab4:
        render_processing()
    with tab5:
        render_preview()
```

### Performance-Optimierung für Streamlit
```python
@st.cache_data(ttl=3600)  # Cache für 1 Stunde
def load_and_process_image(image_path: str, method: str) -> np.ndarray:
    """Lädt und verarbeitet Bild mit Caching."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {image_path}")
    
    # Edge Detection anwenden
    if method == "Canny":
        return run_canny(image)
    elif method == "Sobel":
        return run_sobel(image)
    # ... weitere Methoden

@st.cache_resource
def load_models():
    """Lädt alle ML-Modelle einmalig."""
    models = {}
    
    # HED Model laden
    try:
        models['hed'] = load_hed_model()
    except Exception as e:
        st.warning(f"HED Model konnte nicht geladen werden: {e}")
    
    # Structured Forests Model laden
    try:
        models['structured'] = load_structured_forests_model()
    except Exception as e:
        st.warning(f"Structured Forests Model konnte nicht geladen werden: {e}")
    
    return models

# Progress Tracking für lange Operationen
def process_batch_with_progress(images: List[str], methods: List[str]):
    """Batch-Verarbeitung mit Progress Bar."""
    total_operations = len(images) * len(methods)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    current_operation = 0
    
    for image_path in images:
        for method in methods:
            current_operation += 1
            progress = current_operation / total_operations
            
            progress_bar.progress(progress)
            status_text.text(f"Verarbeite {os.path.basename(image_path)} mit {method}...")
            
            try:
                result = load_and_process_image(image_path, method)
                # Speichere Ergebnis...
                
            except Exception as e:
                st.error(f"Fehler bei {image_path} mit {method}: {e}")
    
    progress_bar.progress(1.0)
    status_text.text("✅ Verarbeitung abgeschlossen!")
```

### Error Handling und Benutzer-Feedback
```python
def safe_execute_with_feedback(operation_name: str, operation_func, *args, **kwargs):
    """Führt Operation sicher aus mit Benutzer-Feedback."""
    try:
        with st.spinner(f"{operation_name} läuft..."):
            result = operation_func(*args, **kwargs)
        
        st.success(f"✅ {operation_name} erfolgreich abgeschlossen!")
        return result
        
    except FileNotFoundError as e:
        st.error(f"❌ Datei nicht gefunden: {e}")
    except ValueError as e:
        st.error(f"❌ Ungültiger Wert: {e}")
    except MemoryError:
        st.error("❌ Nicht genügend Speicher! Versuchen Sie kleinere Bilder oder weniger Methoden.")
    except Exception as e:
        st.error(f"❌ Unerwarteter Fehler: {e}")
        st.exception(e)  # Für Debug-Info
    
    return None

# Beispiel-Verwendung
def handle_file_upload():
    """Behandelt Datei-Upload mit Validierung."""
    uploaded_files = st.file_uploader(
        "Bilder hochladen",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        accept_multiple_files=True,
        help="Unterstützte Formate: PNG, JPG, JPEG, BMP"
    )
    
    if uploaded_files:
        valid_files = []
        
        for uploaded_file in uploaded_files:
            # Datei-Validierung
            if uploaded_file.size > 50 * 1024 * 1024:  # 50MB Limit
                st.warning(f"⚠️ {uploaded_file.name} ist zu groß (>50MB)")
                continue
            
            # Speichere temporär
            temp_path = save_uploaded_file(uploaded_file)
            if temp_path:
                valid_files.append(temp_path)
        
        if valid_files:
            st.success(f"✅ {len(valid_files)} Bilder erfolgreich hochgeladen")
            st.session_state.app_state.selected_images = valid_files
        
        return valid_files
    
    return []
```

## PR-Anweisungen

### Pull Request Format
```markdown
## 📋 Beschreibung
Kurze Beschreibung der Änderungen auf Deutsch.

## 🔧 Art der Änderung
- [ ] Bug Fix (nicht-breaking change, der ein Problem behebt)
- [ ] Neues Feature (nicht-breaking change, der Funktionalität hinzufügt)
- [ ] Breaking Change (fix oder feature, der bestehende Funktionalität beeinflusst)
- [ ] Dokumentation Update
- [ ] Performance Verbesserung
- [ ] Code Refactoring

## 🧪 Tests
- [ ] Alle bestehenden Tests laufen durch
- [ ] Neue Tests für neue Funktionalität hinzugefügt
- [ ] Manuelle Tests durchgeführt

## 📝 Checklist
- [ ] Code folgt dem Style Guide (Black, isort)
- [ ] Selbst-Review des Codes durchgeführt
- [ ] Code ist kommentiert, besonders in schwer verständlichen Bereichen
- [ ] Entsprechende Änderungen an der Dokumentation vorgenommen
- [ ] Keine neuen Warnungen generiert
- [ ] Tests hinzugefügt, die die Änderung abdecken
- [ ] Neue und bestehende Unit Tests laufen lokal durch

## 🖼️ Screenshots (falls UI-Änderungen)
[Screenshots hier einfügen]
```

### Code Review Kriterien
1. **Funktionalität**: Code macht was er soll, Edge Detection funktioniert korrekt
2. **Performance**: Keine Verschlechterung der Verarbeitungszeit
3. **Speicher**: Keine Memory Leaks, effiziente Bildverarbeitung
4. **GUI**: Streamlit Interface bleibt responsiv und benutzerfreundlich
5. **Tests**: Ausreichende Testabdeckung für neue Features
6. **Dokumentation**: Deutsche Kommentare und Docstrings aktualisiert

### Merge-Anforderungen
- Mindestens 1 Approval von Code Owner
- Alle CI/CD Checks müssen grün sein
- Branch ist auf neuestem Stand mit main
- Keine Merge-Konflikte
- Performance-Tests bestanden

## Programmatische Validierung

### Pre-commit Hooks Setup
```bash
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        args: [--line-length=88]
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black]
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88]
  
  - repo: local
    hooks:
      - id: edge-detection-test
        name: Edge Detection Algorithm Test
        entry: python -m pytest tests/test_algorithms.py -v
        language: system
        pass_filenames: false
```

### Automatische Qualitätschecks
```python
# scripts/quality_check.py
import subprocess
import sys
import os

def run_check(command: str, description: str) -> bool:
    """Führt Qualitätscheck aus und gibt Ergebnis zurück."""
    print(f"🔍 {description}...")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} bestanden")
        return True
    else:
        print(f"❌ {description} fehlgeschlagen:")
        print(result.stdout)
        print(result.stderr)
        return False

def main():
    """Führt alle Qualitätschecks aus."""
    checks = [
        ("black --check --line-length 88 .", "Code Formatierung"),
        ("isort --check-only --profile black .", "Import Sortierung"),
        ("flake8 --max-line-length=88 .", "Linting"),
        ("python -m pytest tests/ -v", "Unit Tests"),
        ("python detectors.py --test-all-methods", "Edge Detection Tests")
    ]
    
    all_passed = True
    
    for command, description in checks:
        if not run_check(command, description):
            all_passed = False
    
    if all_passed:
        print("\n🎉 Alle Qualitätschecks bestanden!")
        sys.exit(0)
    else:
        print("\n💥 Einige Checks fehlgeschlagen!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Edge Detection Validierung
```python
# tests/validation.py
def validate_edge_detection_output(result: np.ndarray, input_image: np.ndarray) -> bool:
    """Validiert Edge Detection Ausgabe."""
    
    # Basis-Validierungen
    if result is None:
        return False
    
    if not isinstance(result, np.ndarray):
        return False
    
    if result.shape != input_image.shape[:2]:  # Nur H,W vergleichen
        return False
    
    if result.dtype != np.uint8:
        return False
    
    # Werte-Validierung (sollte nur 0 und 255 enthalten nach Invertierung)
    unique_values = np.unique(result)
    valid_values = np.array([0, 255])
    
    if not all(val in valid_values for val in unique_values):
        return False
    
    # Mindestens einige Kanten sollten detektiert werden
    edge_pixels = np.sum(result == 0)  # Dunkle Pixel (Kanten)
    total_pixels = result.size
    edge_ratio = edge_pixels / total_pixels
    
    if edge_ratio < 0.01 or edge_ratio > 0.5:  # 1% bis 50% Kanten
        return False
    
    return True

def run_validation_suite():
    """Führt komplette Validierung aller Methoden aus."""
    test_images = [
        "tests/fixtures/simple_square.png",
        "tests/fixtures/complex_scene.jpg",
        "tests/fixtures/noisy_image.png"
    ]
    
    methods = ["canny", "sobel", "prewitt", "roberts", "laplacian"]
    
    results = {}
    
    for test_image_path in test_images:
        if not os.path.exists(test_image_path):
            continue
            
        image = cv2.imread(test_image_path)
        image_name = os.path.basename(test_image_path)
        
        results[image_name] = {}
        
        for method in methods:
            try:
                func = globals()[f"run_{method}"]
                result = func(image)
                
                is_valid = validate_edge_detection_output(result, image)
                results[image_name][method] = is_valid
                
                if is_valid:
                    print(f"✅ {method} auf {image_name}: Bestanden")
                else:
                    print(f"❌ {method} auf {image_name}: Fehlgeschlagen")
                    
            except Exception as e:
                print(f"💥 {method} auf {image_name}: Fehler - {e}")
                results[image_name][method] = False
    
    return results
```

---

**🎯 Diese AGENTS.md ist speziell für dein "15-Edges" Repository optimiert und gibt OpenAI Codex präzise Anweisungen für:**

- ✅ Korrekte Edge Detection Implementierung 
- ✅ Deutsche Kommentare und Dokumentation
- ✅ Streamlit GUI Best Practices
- ✅ Robuste Fehlerbehandlung
- ✅ Performance-Optimierung
- ✅ Umfassende Tests
- ✅ Produktions-Ready Deployment

**Kopiere diese Datei als `AGENTS.md` in dein Repository-Root und Codex wird genau verstehen, was du möchtest! 🚀**
