"""
GUI-Komponenten für erweiterte Streamlit-Funktionalität
"""

import streamlit as st
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def folder_picker(label: str, default_path: str = "./") -> Optional[str]:
    """
    Erweiterte Ordner-Auswahl mit Browser-ähnlicher Navigation
    """
    st.subheader(label)
    
    # Aktueller Pfad
    if 'current_path' not in st.session_state:
        st.session_state.current_path = os.path.abspath(default_path)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("⬆️ Hoch"):
            parent = os.path.dirname(st.session_state.current_path)
            if os.path.exists(parent):
                st.session_state.current_path = parent
    
    with col2:
        st.text_input(
            "Aktueller Pfad:", 
            value=st.session_state.current_path,
            key="path_input",
            on_change=lambda: setattr(st.session_state, 'current_path', st.session_state.path_input)
        )
    
    with col3:
        if st.button("🏠 Home"):
            st.session_state.current_path = os.path.expanduser("~")
    
    # Ordner-Inhalt anzeigen
    try:
        if os.path.exists(st.session_state.current_path):
            items = os.listdir(st.session_state.current_path)
            folders = [item for item in items if os.path.isdir(os.path.join(st.session_state.current_path, item))]
            folders.sort()
            
            if folders:
                st.write("📁 Verfügbare Ordner:")
                cols = st.columns(3)
                
                for i, folder in enumerate(folders):
                    with cols[i % 3]:
                        if st.button(f"📁 {folder}", key=f"folder_{i}"):
                            new_path = os.path.join(st.session_state.current_path, folder)
                            st.session_state.current_path = new_path
                            st.rerun()
            else:
                st.info("Keine Unterordner gefunden")
        else:
            st.error("Pfad existiert nicht!")
            
    except PermissionError:
        st.error("Keine Berechtigung für diesen Ordner!")
    
    # Aktuellen Ordner auswählen
    if st.button("✅ Diesen Ordner auswählen", type="primary"):
        return st.session_state.current_path
    
    return None

def image_gallery(images: List[str], max_images: int = 12) -> List[str]:
    """
    Interaktive Bildgalerie mit Auswahl-Funktionalität
    """
    if not images:
        st.warning("Keine Bilder gefunden")
        return []
    
    st.subheader(f"🖼️ Bildgalerie ({len(images)} Bilder)")
    
    # Auswahl-Optionen
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Alle auswählen"):
            return images
    with col2:
        if st.button("❌ Alle abwählen"):
            return []
    with col3:
        show_details = st.checkbox("📊 Details anzeigen", value=False)
    
    # Paginierung
    items_per_page = max_images
    total_pages = (len(images) - 1) // items_per_page + 1
    
    if total_pages > 1:
        page = st.selectbox("Seite:", range(1, total_pages + 1)) - 1
    else:
        page = 0
    
    start_idx = page * items_per_page
    end_idx = min(start_idx + items_per_page, len(images))
    page_images = images[start_idx:end_idx]
    
    # Bildgalerie
    selected_images = []
    cols = st.columns(4)
    
    for i, img_path in enumerate(page_images):
        with cols[i % 4]:
            try:
                # Thumbnail laden
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    thumbnail = cv2.resize(img, (150, 150))
                    
                    # Bild anzeigen
                    st.image(thumbnail, channels="BGR")
                    
                    # Bildname
                    img_name = os.path.basename(img_path)
                    st.caption(img_name)
                    
                    # Details
                    if show_details:
                        file_size = os.path.getsize(img_path) / 1024  # KB
                        st.caption(f"{w}×{h} • {file_size:.1f}KB")
                    
                    # Auswahl-Checkbox
                    if st.checkbox("Auswählen", key=f"select_{start_idx + i}"):
                        selected_images.append(img_path)
                
                else:
                    st.error("❌ Ungültiges Bild")
                    
            except Exception as e:
                st.error(f"❌ Fehler: {str(e)}")
    
    if total_pages > 1:
        st.info(f"Seite {page + 1} von {total_pages}")
    
    return selected_images

def progress_tracker(total_operations: int, current_operation: int, 
                    current_image: str = "", current_method: str = "",
                    start_time: float = None) -> Dict:
    """
    Erweiterte Progress-Anzeige mit ETA und Statistiken
    """
    progress_percent = (current_operation / total_operations) * 100 if total_operations > 0 else 0
    
    # Progress Bar
    progress_bar = st.progress(progress_percent / 100)
    
    # Statistiken
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Fortschritt", f"{progress_percent:.1f}%")
    
    with col2:
        st.metric("Operationen", f"{current_operation}/{total_operations}")
    
    with col3:
        remaining = total_operations - current_operation
        st.metric("Verbleibend", remaining)
    
    with col4:
        if start_time and current_operation > 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / current_operation
            eta_seconds = avg_time * remaining
            
            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds/60:.0f}m {eta_seconds%60:.0f}s"
            else:
                eta_str = f"{eta_seconds/3600:.0f}h {(eta_seconds%3600)/60:.0f}m"
            
            st.metric("ETA", eta_str)
        else:
            st.metric("ETA", "Berechnung...")
    
    # Aktuelle Operation
    if current_image and current_method:
        st.info(f"🔄 Verarbeite: **{os.path.basename(current_image)}** mit **{current_method}**")
    
    return {
        "progress_percent": progress_percent,
        "current_operation": current_operation,
        "total_operations": total_operations,
        "remaining": remaining
    }

def method_selector_advanced(all_methods: List[Tuple[str, callable]]) -> List[str]:
    """
    Erweiterte Methoden-Auswahl mit Kategorisierung und Empfehlungen
    """
    st.subheader("🔧 Edge Detection Methoden")
    
    # Methoden-Kategorien mit Beschreibungen
    categories = {
        "⭐ Empfohlene (Beste Qualität)": {
            "methods": ["HED_PyTorch", "Kornia_Canny", "MultiScaleCanny"],
            "description": "Beste Balance aus Qualität und Geschwindigkeit",
            "color": "#28a745"
        },
        "📊 Klassische Methoden": {
            "methods": ["Laplacian", "Prewitt", "Roberts", "Scharr", "GradientMagnitude"],
            "description": "Bewährte mathematische Ansätze",
            "color": "#6c757d"
        },
        "🎯 Canny Varianten": {
            "methods": ["Kornia_Canny", "MultiScaleCanny", "AdaptiveCanny"],
            "description": "Verschiedene Canny-Implementierungen",
            "color": "#17a2b8"
        },
        "🤖 Deep Learning": {
            "methods": ["HED_OpenCV", "HED_PyTorch", "StructuredForests", "BDCN"],
            "description": "KI-basierte Kantenerkennung",
            "color": "#e83e8c"
        },
        "⚡ GPU-Beschleunigt": {
            "methods": ["Kornia_Canny", "Kornia_Sobel"],
            "description": "Schnelle GPU-Implementierungen",
            "color": "#fd7e14"
        }
    }
    
    # Initialisiere ausgewählte Methoden
    if 'selected_methods_advanced' not in st.session_state:
        st.session_state.selected_methods_advanced = ["HED_PyTorch", "Kornia_Canny"]
    
    # Schnell-Auswahl
    st.markdown("**🚀 Schnell-Auswahl:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("⭐ Empfohlene", key="quick_recommended"):
            st.session_state.selected_methods_advanced = ["HED_PyTorch", "Kornia_Canny", "MultiScaleCanny"]
    
    with col2:
        if st.button("🚀 Schnell", key="quick_fast"):
            st.session_state.selected_methods_advanced = ["Kornia_Canny", "Laplacian"]
    
    with col3:
        if st.button("🎯 Qualität", key="quick_quality"):
            st.session_state.selected_methods_advanced = ["HED_PyTorch", "StructuredForests", "BDCN"]
    
    with col4:
        if st.button("🔄 Alle", key="quick_all"):
            st.session_state.selected_methods_advanced = [name for name, _ in all_methods]
    
    st.markdown("---")
    
    # Kategorien durchgehen
    for category_name, category_info in categories.items():
        with st.expander(category_name, expanded=category_name.startswith("⭐")):
            st.markdown(f"*{category_info['description']}*")
            
            # Verfügbare Methoden in dieser Kategorie
            available_methods = [(name, func) for name, func in all_methods 
                               if name in category_info['methods']]
            
            if available_methods:
                cols = st.columns(min(3, len(available_methods)))
                
                for i, (method_name, _) in enumerate(available_methods):
                    with cols[i % 3]:
                        # Methoden-Info
                        method_info = get_method_info(method_name)
                        
                        # Checkbox mit Info
                        selected = method_name in st.session_state.selected_methods_advanced
                        if st.checkbox(
                            f"**{method_name}**", 
                            value=selected,
                            key=f"method_checkbox_{method_name}"
                        ):
                            if method_name not in st.session_state.selected_methods_advanced:
                                st.session_state.selected_methods_advanced.append(method_name)
                        else:
                            if method_name in st.session_state.selected_methods_advanced:
                                st.session_state.selected_methods_advanced.remove(method_name)
                        
                        # Zusätzliche Info
                        st.caption(f"⚡ {method_info['speed']} • 🎯 {method_info['quality']}")
            else:
                st.info("Keine Methoden in dieser Kategorie verfügbar")
    
    # Zusammenfassung
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Ausgewählte Methoden", len(st.session_state.selected_methods_advanced))
    
    with col2:
        if st.session_state.selected_methods_advanced:
            # Geschätzte Zeit berechnen
            total_time_estimate = sum(
                get_method_info(method)['time_per_image'] 
                for method in st.session_state.selected_methods_advanced
            )
            st.metric("Zeit/Bild (ca.)", f"{total_time_estimate:.1f}s")
    
    # Ausgewählte Methoden anzeigen
    if st.session_state.selected_methods_advanced:
        st.markdown("**📋 Ausgewählte Methoden:**")
        methods_text = " • ".join(st.session_state.selected_methods_advanced)
        st.info(methods_text)
    
    return st.session_state.selected_methods_advanced

def get_method_info(method_name: str) -> Dict:
    """
    Gibt Informationen über eine Edge Detection Methode zurück
    """
    method_infos = {
        "HED_PyTorch": {
            "speed": "Mittel",
            "quality": "Sehr hoch",
            "time_per_image": 3.5,
            "description": "Deep Learning basierte Kantenerkennung"
        },
        "Kornia_Canny": {
            "speed": "Sehr schnell",
            "quality": "Hoch",
            "time_per_image": 0.5,
            "description": "GPU-beschleunigter Canny Edge Detector"
        },
        "MultiScaleCanny": {
            "speed": "Schnell",
            "quality": "Sehr hoch",
            "time_per_image": 1.2,
            "description": "Canny mit mehreren Skalen"
        },
        "StructuredForests": {
            "speed": "Mittel",
            "quality": "Sehr hoch",
            "time_per_image": 2.8,
            "description": "Structured Edge Detection"
        },
        "Laplacian": {
            "speed": "Sehr schnell",
            "quality": "Mittel",
            "time_per_image": 0.2,
            "description": "Klassischer Laplacian Filter"
        },
        "BDCN": {
            "speed": "Langsam",
            "quality": "Höchste",
            "time_per_image": 5.0,
            "description": "State-of-the-art Deep Learning"
        }
    }
    
    return method_infos.get(method_name, {
        "speed": "Unbekannt",
        "quality": "Unbekannt", 
        "time_per_image": 2.0,
        "description": "Keine Informationen verfügbar"
    })

def batch_processor(images: List[str], methods: List[str], 
                   output_dir: str, settings: Dict) -> None:
    """
    Batch-Prozessor mit Threading und Progress-Updates
    """
    if not images or not methods:
        st.error("❌ Keine Bilder oder Methoden ausgewählt!")
        return
    
    # Ausgabeordner erstellen
    os.makedirs(output_dir, exist_ok=True)
    
    total_operations = len(images) * len(methods)
    current_operation = 0
    start_time = time.time()
    
    # Progress Container
    progress_container = st.container()
    log_container = st.container()
    
    # Processing Log
    processing_log = []
    
    with st.spinner("🚀 Verarbeitung läuft..."):
        # Lade alle verfügbaren Methoden
        from detectors import get_all_methods
        all_methods_dict = {name: func for name, func in get_all_methods()}
        
        # Verarbeite jedes Bild mit jeder Methode
        for image_path in images:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            
            for method_name in methods:
                if method_name in all_methods_dict:
                    try:
                        # Progress Update
                        current_operation += 1
                        
                        with progress_container:
                            progress_info = progress_tracker(
                                total_operations, current_operation,
                                image_path, method_name, start_time
                            )
                        
                        # Methode ausführen
                        method_func = all_methods_dict[method_name]
                        result = method_func(image_path, target_size=settings.get('target_size'))
                        
                        # Ergebnis speichern
                        output_filename = f"{image_name}_{method_name}.png"
                        output_path = os.path.join(output_dir, output_filename)
                        cv2.imwrite(output_path, result)
                        
                        # Log-Eintrag
                        log_entry = {
                            'status': 'success',
                            'image': os.path.basename(image_path),
                            'method': method_name,
                            'output': output_filename,
                            'timestamp': time.time()
                        }
                        processing_log.append(log_entry)
                        
                    except Exception as e:
                        # Fehler-Log
                        log_entry = {
                            'status': 'error',
                            'image': os.path.basename(image_path),
                            'method': method_name,
                            'error': str(e),
                            'timestamp': time.time()
                        }
                        processing_log.append(log_entry)
                        
                        with log_container:
                            st.error(f"❌ Fehler bei {image_name} → {method_name}: {str(e)}")
    
    # Verarbeitung abgeschlossen
    st.success("🎉 Verarbeitung abgeschlossen!")
    
    # Statistiken
    successful = sum(1 for log in processing_log if log['status'] == 'success')
    failed = len(processing_log) - successful
    total_time = time.time() - start_time
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Erfolgreich", successful)
    with col2:
        st.metric("❌ Fehlgeschlagen", failed)
    with col3:
        st.metric("⏱️ Gesamtzeit", f"{total_time/60:.1f}m")
    with col4:
        if successful > 0:
            avg_time = total_time / successful
            st.metric("Ø Zeit/Bild", f"{avg_time:.1f}s")
    
    return processing_log
