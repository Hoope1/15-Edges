import streamlit as st
import os
import cv2
import numpy as np
from pathlib import Path
import tempfile
import zipfile
import time
import threading
from io import BytesIO
import base64
from typing import List, Tuple, Dict, Optional
import json

# Import unserer bestehenden Module
try:
    from detectors import get_all_methods, get_max_resolution, standardize_output
    DETECTORS_AVAILABLE = True
except ImportError:
    DETECTORS_AVAILABLE = False
    st.error("❌ detectors.py konnte nicht importiert werden. Bitte run.bat ausführen!")

# Streamlit Konfiguration
st.set_page_config(
    page_title="🎨 Edge Detection Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS für besseres Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd, #bbdefb);
        border-radius: 10px;
    }
    .method-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-processing {
        color: #ffc107;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0px 0px;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialisierung
def init_session_state():
    if 'selected_methods' not in st.session_state:
        st.session_state.selected_methods = []
    if 'input_mode' not in st.session_state:
        st.session_state.input_mode = "Ordner auswählen"
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = "idle"
    if 'selected_images' not in st.session_state:
        st.session_state.selected_images = []
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = "./results"
    if 'progress' not in st.session_state:
        st.session_state.progress = 0
    if 'processing_log' not in st.session_state:
        st.session_state.processing_log = []

# Helper Functions
@st.cache_data
def load_image(image_path: str) -> Optional[np.ndarray]:
    """Lädt ein Bild und cached es"""
    try:
        return cv2.imread(image_path)
    except Exception:
        return None

def get_image_info(image_path: str) -> Dict:
    """Sammelt Informationen über ein Bild"""
    img = load_image(image_path)
    if img is not None:
        h, w = img.shape[:2]
        size_mb = os.path.getsize(image_path) / (1024 * 1024)
        return {
            'path': image_path,
            'name': os.path.basename(image_path),
            'dimensions': f"{w}x{h}",
            'size_mb': round(size_mb, 2),
            'valid': True
        }
    return {
        'path': image_path,
        'name': os.path.basename(image_path),
        'dimensions': "Ungültig",
        'size_mb': 0,
        'valid': False
    }

def find_images_in_folder(folder_path: str) -> List[str]:
    """Findet alle Bilder in einem Ordner"""
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    image_files = []
    
    if not os.path.exists(folder_path):
        return []
    
    for ext in extensions:
        pattern = os.path.join(folder_path, f"*{ext}")
        image_files.extend(Path(folder_path).glob(f"*{ext}"))
        image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
    
    return [str(p) for p in image_files]

def create_image_thumbnail(image_path: str, size: Tuple[int, int] = (150, 150)) -> Optional[np.ndarray]:
    """Erstellt ein Thumbnail eines Bildes"""
    img = load_image(image_path)
    if img is not None:
        return cv2.resize(img, size)
    return None

def numpy_to_base64(img_array: np.ndarray) -> str:
    """Konvertiert NumPy Array zu base64 String für Anzeige"""
    _, buffer = cv2.imencode('.png', img_array)
    img_base64 = base64.b64encode(buffer).decode()
    return f"data:image/png;base64,{img_base64}"

# Hauptanwendung
def main():
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">🎨 Edge Detection Studio</div>', unsafe_allow_html=True)
    
    if not DETECTORS_AVAILABLE:
        st.error("⚠️ Edge Detection Module nicht verfügbar. Bitte führen Sie zuerst 'run.bat' aus!")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Konfiguration")
        
        # Eingabe-Modus
        st.subheader("📁 Eingabe")
        input_mode = st.radio(
            "Modus wählen:",
            ["Ordner auswählen", "Einzelne Bilder"],
            key="input_mode_radio"
        )
        st.session_state.input_mode = input_mode
        
        # Methoden-Schnellauswahl
        st.subheader("🔧 Schnell-Methoden")
        all_methods = get_all_methods() if DETECTORS_AVAILABLE else []
        
        quick_selections = {
            "Empfohlene (Schnell)": ["HED_PyTorch", "Kornia_Canny", "MultiScaleCanny"],
            "Klassisch": ["Laplacian", "Prewitt", "Scharr"],
            "Deep Learning": ["HED_PyTorch", "StructuredForests", "BDCN"],
            "Alle": [name for name, _ in all_methods]
        }
        
        for preset_name, methods in quick_selections.items():
            if st.button(f"📦 {preset_name}", key=f"preset_{preset_name}"):
                st.session_state.selected_methods = methods
        
        # Ausgabe-Konfiguration
        st.subheader("📂 Ausgabe")
        output_dir = st.text_input(
            "Ausgabeordner:",
            value=st.session_state.output_dir,
            help="Pfad zum Ausgabeordner"
        )
        st.session_state.output_dir = output_dir
        
        # Optionen
        st.subheader("⚙️ Optionen")
        invert_colors = st.checkbox("🎨 Invertierte Ausgabe", value=True)
        uniform_size = st.checkbox("📐 Einheitliche Größe", value=True)
        
        # Verarbeitungs-Button
        st.markdown("---")
        if st.session_state.processing_status == "idle":
            if st.button("🚀 VERARBEITUNG STARTEN", type="primary", use_container_width=True):
                start_processing()
        else:
            st.markdown('<p class="status-processing">⏳ Verarbeitung läuft...</p>', unsafe_allow_html=True)
            if st.button("⏹️ STOPPEN", type="secondary", use_container_width=True):
                st.session_state.processing_status = "idle"
    
    # Hauptbereich mit Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📷 Bildauswahl", 
        "🔧 Methoden", 
        "⚙️ Einstellungen", 
        "🚀 Verarbeitung", 
        "👁️ Vorschau"
    ])
    
    with tab1:
        image_selection_tab()
    
    with tab2:
        method_selection_tab()
    
    with tab3:
        settings_tab()
    
    with tab4:
        processing_tab()
    
    with tab5:
        preview_tab()

def image_selection_tab():
    """Tab für Bildauswahl"""
    st.header("📷 Bildauswahl")
    
    if st.session_state.input_mode == "Ordner auswählen":
        col1, col2 = st.columns([3, 1])
        
        with col1:
            folder_path = st.text_input(
                "📁 Ordner-Pfad:",
                value="./images",
                help="Pfad zum Ordner mit Bildern"
            )
        
        with col2:
            if st.button("🔍 Scannen", type="primary"):
                if os.path.exists(folder_path):
                    image_files = find_images_in_folder(folder_path)
                    st.session_state.selected_images = image_files
                else:
                    st.error("Ordner existiert nicht!")
        
        if st.session_state.selected_images:
            st.success(f"✅ {len(st.session_state.selected_images)} Bilder gefunden")
            
            # Bild-Gallery
            st.subheader("🖼️ Gefundene Bilder")
            cols = st.columns(4)
            
            for i, img_path in enumerate(st.session_state.selected_images[:8]):  # Zeige nur erste 8
                with cols[i % 4]:
                    img_info = get_image_info(img_path)
                    if img_info['valid']:
                        thumbnail = create_image_thumbnail(img_path, (120, 120))
                        if thumbnail is not None:
                            st.image(thumbnail, caption=f"{img_info['name']}\n{img_info['dimensions']}")
                    else:
                        st.error(f"❌ {img_info['name']}")
            
            if len(st.session_state.selected_images) > 8:
                st.info(f"... und {len(st.session_state.selected_images) - 8} weitere Bilder")
            
            # Maximale Auflösung anzeigen
            if st.session_state.selected_images:
                max_res = get_max_resolution(st.session_state.selected_images)
                st.info(f"🎯 Maximale Auflösung: {max_res[0]}x{max_res[1]}")
    
    else:  # Einzelne Bilder
        uploaded_files = st.file_uploader(
            "📎 Bilder hochladen",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Wählen Sie eine oder mehrere Bilddateien aus"
        )
        
        if uploaded_files:
            st.session_state.selected_images = []
            
            # Speichere hochgeladene Dateien temporär
            temp_dir = tempfile.mkdtemp()
            
            for uploaded_file in uploaded_files:
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.selected_images.append(temp_path)
            
            st.success(f"✅ {len(uploaded_files)} Bilder hochgeladen")
            
            # Zeige hochgeladene Bilder
            cols = st.columns(4)
            for i, img_path in enumerate(st.session_state.selected_images):
                with cols[i % 4]:
                    img_info = get_image_info(img_path)
                    thumbnail = create_image_thumbnail(img_path, (120, 120))
                    if thumbnail is not None:
                        st.image(thumbnail, caption=f"{img_info['name']}\n{img_info['dimensions']}")

def method_selection_tab():
    """Tab für Methodenauswahl"""
    st.header("🔧 Edge Detection Methoden")
    
    all_methods = get_all_methods()
    
    # Methoden nach Kategorien gruppiert
    categories = {
        "📊 Klassische Methoden": [
            "Laplacian", "Prewitt", "Roberts", "Scharr", 
            "GradientMagnitude", "MorphologicalGradient"
        ],
        "🎯 Canny Varianten": [
            "Kornia_Canny", "MultiScaleCanny", "AdaptiveCanny"
        ],
        "🤖 Deep Learning": [
            "HED_OpenCV", "HED_PyTorch", "StructuredForests", 
            "BDCN", "FixedCNN"
        ],
        "⚡ GPU-Beschleunigt": [
            "Kornia_Canny", "Kornia_Sobel"
        ]
    }
    
    # Bulk-Auswahl
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Alle auswählen"):
            st.session_state.selected_methods = [name for name, _ in all_methods]
    with col2:
        if st.button("❌ Alle abwählen"):
            st.session_state.selected_methods = []
    with col3:
        if st.button("⭐ Empfohlene"):
            st.session_state.selected_methods = ["HED_PyTorch", "Kornia_Canny", "MultiScaleCanny", "Scharr"]
    
    # Methoden nach Kategorien
    for category, method_names in categories.items():
        with st.expander(category, expanded=True):
            available_methods = [(name, func) for name, func in all_methods if name in method_names]
            
            for name, _ in available_methods:
                selected = name in st.session_state.selected_methods
                if st.checkbox(
                    f"🔧 {name}", 
                    value=selected, 
                    key=f"method_{name}"
                ):
                    if name not in st.session_state.selected_methods:
                        st.session_state.selected_methods.append(name)
                else:
                    if name in st.session_state.selected_methods:
                        st.session_state.selected_methods.remove(name)
    
    # Zusammenfassung
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Ausgewählte Methoden", len(st.session_state.selected_methods))
    with col2:
        if st.session_state.selected_images and st.session_state.selected_methods:
            total_ops = len(st.session_state.selected_images) * len(st.session_state.selected_methods)
            estimated_time = total_ops * 2  # ~2 Sekunden pro Operation
            st.metric("Geschätzte Zeit", f"{estimated_time//60}m {estimated_time%60}s")

def settings_tab():
    """Tab für Einstellungen"""
    st.header("⚙️ Ausgabe-Einstellungen")
    
    # Ausgabeordner
    st.subheader("📂 Ausgabeordner")
    col1, col2 = st.columns([3, 1])
    with col1:
        output_dir = st.text_input(
            "Ausgabepfad:",
            value=st.session_state.output_dir,
            help="Ordner für die Ergebnisse"
        )
        st.session_state.output_dir = output_dir
    with col2:
        if st.button("📁 Standard"):
            st.session_state.output_dir = "./results"
    
    # Bildverarbeitung
    st.subheader("🎨 Bildverarbeitung")
    col1, col2 = st.columns(2)
    with col1:
        invert_colors = st.checkbox(
            "🔄 Invertierte Ausgabe", 
            value=True,
            help="Weiße Hintergründe, dunkle Linien"
        )
    with col2:
        uniform_size = st.checkbox(
            "📐 Einheitliche Auflösung", 
            value=True,
            help="Alle Ergebnisse auf gleiche Größe skalieren"
        )
    
    # Auflösungseinstellungen
    st.subheader("📐 Auflösung")
    resolution_mode = st.radio(
        "Auflösungsmodus:",
        ["Automatisch (höchste Eingabe)", "Benutzerdefiniert", "Standard-Größen"]
    )
    
    if resolution_mode == "Benutzerdefiniert":
        col1, col2 = st.columns(2)
        with col1:
            custom_width = st.number_input("Breite:", value=1920, min_value=64, max_value=8192)
        with col2:
            custom_height = st.number_input("Höhe:", value=1080, min_value=64, max_value=8192)
    
    elif resolution_mode == "Standard-Größen":
        standard_sizes = {
            "Full HD": (1920, 1080),
            "4K": (3840, 2160),
            "HD": (1280, 720),
            "SVGA": (800, 600),
            "XGA": (1024, 768)
        }
        selected_size = st.selectbox("Standard-Auflösung:", list(standard_sizes.keys()))
    
    # Dateiformat
    st.subheader("💾 Dateiformat")
    col1, col2 = st.columns(2)
    with col1:
        file_format = st.radio(
            "Format:",
            ["PNG (verlustfrei)", "JPEG (komprimiert)"]
        )
    with col2:
        if file_format == "JPEG (komprimiert)":
            jpeg_quality = st.slider("JPEG Qualität:", 70, 100, 95)
    
    # Namensschema
    st.subheader("🏷️ Namensschema")
    naming_scheme = st.radio(
        "Schema:",
        [
            "{originalname}_{methode}",
            "{methode}_{originalname}", 
            "Benutzerdefiniert"
        ]
    )
    
    if naming_scheme == "Benutzerdefiniert":
        custom_naming = st.text_input(
            "Schema:", 
            value="{originalname}_{methode}",
            help="Verfügbare Platzhalter: {originalname}, {methode}, {timestamp}"
        )
    
    # Erweiterte Optionen
    with st.expander("🔧 Erweiterte Optionen"):
        create_subdirs = st.checkbox("📁 Unterordner für jede Methode", value=False)
        create_summary = st.checkbox("📄 Zusammenfassungsdatei erstellen", value=True)
        preserve_metadata = st.checkbox("🏷️ Metadaten beibehalten", value=False)
        parallel_processing = st.checkbox("⚡ Parallele Verarbeitung", value=True)
        
        if parallel_processing:
            max_workers = st.slider("Max. parallele Prozesse:", 1, 8, 4)

def processing_tab():
    """Tab für Verarbeitung und Ergebnisse"""
    st.header("🚀 Verarbeitung & Ergebnisse")
    
    # Status-Anzeige
    status_colors = {
        "idle": "🟢 Bereit",
        "running": "🟡 Läuft",
        "completed": "🟢 Abgeschlossen", 
        "error": "🔴 Fehler"
    }
    
    st.markdown(f"**Status:** {status_colors.get(st.session_state.processing_status, '🟠 Unbekannt')}")
    
    # Progress Bar
    if st.session_state.processing_status == "running":
        progress_bar = st.progress(st.session_state.progress / 100)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fortschritt", f"{st.session_state.progress:.1f}%")
        with col2:
            st.metric("Verarbeitet", f"{len(st.session_state.processing_log)}")
        with col3:
            if st.session_state.selected_images and st.session_state.selected_methods:
                total = len(st.session_state.selected_images) * len(st.session_state.selected_methods)
                remaining = total - len(st.session_state.processing_log)
                st.metric("Verbleibend", f"{remaining}")
    
    # Verarbeitungslog
    if st.session_state.processing_log:
        st.subheader("📋 Verarbeitungslog")
        
        # Statistiken
        successful = sum(1 for entry in st.session_state.processing_log if entry['status'] == 'success')
        failed = len(st.session_state.processing_log) - successful
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Erfolgreich", successful)
        with col2:
            st.metric("❌ Fehlgeschlagen", failed)
        with col3:
            if st.session_state.processing_log:
                avg_time = sum(entry.get('duration', 0) for entry in st.session_state.processing_log) / len(st.session_state.processing_log)
                st.metric("⏱️ Ø Zeit/Bild", f"{avg_time:.1f}s")
        
        # Log-Einträge
        with st.expander("📜 Detailliertes Log", expanded=False):
            for entry in st.session_state.processing_log[-10:]:  # Letzte 10 Einträge
                status_icon = "✅" if entry['status'] == 'success' else "❌"
                st.text(f"{status_icon} {entry['image']} → {entry['method']} ({entry.get('duration', 0):.1f}s)")
    
    # Ergebnisse
    if st.session_state.processing_status == "completed":
        st.subheader("📁 Ergebnisse")
        
        output_path = os.path.join(st.session_state.output_dir, "edge_detection_results")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📂 Ordner öffnen", type="primary"):
                try:
                    os.startfile(output_path)  # Windows
                except:
                    st.info(f"Ergebnisse gespeichert in: {output_path}")
        
        with col2:
            if st.button("📥 ZIP Download", type="secondary"):
                create_download_zip(output_path)
        
        # Ergebnis-Gallery (falls Bilder vorhanden)
        if os.path.exists(output_path):
            result_files = [f for f in os.listdir(output_path) if f.endswith(('.png', '.jpg'))]
            if result_files:
                st.subheader("🖼️ Ergebnis-Vorschau")
                
                # Zeige erste paar Ergebnisse
                cols = st.columns(4)
                for i, result_file in enumerate(result_files[:8]):
                    with cols[i % 4]:
                        result_path = os.path.join(output_path, result_file)
                        img = load_image(result_path)
                        if img is not None:
                            thumbnail = cv2.resize(img, (150, 150))
                            st.image(thumbnail, caption=result_file)

def preview_tab():
    """Tab für Live-Vorschau"""
    st.header("👁️ Live-Vorschau")
    
    if not st.session_state.selected_images:
        st.warning("⚠️ Bitte zuerst Bilder auswählen!")
        return
    
    if not st.session_state.selected_methods:
        st.warning("⚠️ Bitte zuerst Edge Detection Methoden auswählen!")
        return
    
    # Auswahl für Vorschau
    col1, col2 = st.columns(2)
    with col1:
        selected_image = st.selectbox(
            "🖼️ Bild auswählen:",
            [os.path.basename(img) for img in st.session_state.selected_images],
            key="preview_image"
        )
    
    with col2:
        selected_method = st.selectbox(
            "🔧 Methode auswählen:",
            st.session_state.selected_methods,
            key="preview_method"
        )
    
    if st.button("🔄 Vorschau generieren", type="primary"):
        # Finde den vollständigen Pfad
        image_path = None
        for img_path in st.session_state.selected_images:
            if os.path.basename(img_path) == selected_image:
                image_path = img_path
                break
        
        if image_path and os.path.exists(image_path):
            try:
                with st.spinner(f"Generiere Vorschau mit {selected_method}..."):
                    # Lade die entsprechende Methode
                    all_methods = get_all_methods()
                    method_func = None
                    for name, func in all_methods:
                        if name == selected_method:
                            method_func = func
                            break
                    
                    if method_func:
                        # Original laden
                        original_img = load_image(image_path)
                        
                        # Edge Detection anwenden
                        result_img = method_func(image_path, target_size=(512, 512))  # Kleinere Größe für Vorschau
                        
                        # Zeige Vergleich
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📷 Original")
                            if original_img is not None:
                                original_resized = cv2.resize(original_img, (512, 512))
                                st.image(original_resized, channels="BGR")
                        
                        with col2:
                            st.subheader(f"🎨 {selected_method}")
                            st.image(result_img, channels="GRAY")
                        
                        # Speicher-Option für diese Vorschau
                        if st.button("💾 Diese Vorschau speichern"):
                            preview_output_dir = os.path.join(st.session_state.output_dir, "previews")
                            os.makedirs(preview_output_dir, exist_ok=True)
                            
                            output_filename = f"preview_{os.path.splitext(selected_image)[0]}_{selected_method}.png"
                            output_path = os.path.join(preview_output_dir, output_filename)
                            
                            cv2.imwrite(output_path, result_img)
                            st.success(f"✅ Vorschau gespeichert: {output_path}")
                    
                    else:
                        st.error(f"Methode {selected_method} nicht gefunden!")
            
            except Exception as e:
                st.error(f"❌ Fehler bei der Vorschau-Generierung: {str(e)}")
        else:
            st.error("❌ Bild nicht gefunden!")

def start_processing():
    """Startet die Batch-Verarbeitung"""
    if not st.session_state.selected_images:
        st.error("❌ Keine Bilder ausgewählt!")
        return
    
    if not st.session_state.selected_methods:
        st.error("❌ Keine Methoden ausgewählt!")
        return
    
    st.session_state.processing_status = "running"
    st.session_state.progress = 0
    st.session_state.processing_log = []
    
    # Hier würde die eigentliche Verarbeitung starten
    # In einer echten Implementierung würde das in einem separaten Thread laufen
    st.success("🚀 Verarbeitung gestartet! (Demo-Modus)")

def create_download_zip(output_path: str):
    """Erstellt ein ZIP-File für Download"""
    if not os.path.exists(output_path):
        st.error("❌ Ausgabeordner nicht gefunden!")
        return
    
    # Erstelle ZIP in Memory
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(output_path):
            for file in files:
                file_path = os.path.join(root, file)
                arc_name = os.path.relpath(file_path, output_path)
                zip_file.write(file_path, arc_name)
    
    zip_buffer.seek(0)
    
    st.download_button(
        label="📥 Edge Detection Ergebnisse.zip",
        data=zip_buffer.getvalue(),
        file_name="edge_detection_results.zip",
        mime="application/zip"
    )

if __name__ == "__main__":
    main()
