# 🎨 Edge Detection Studio

Eine professionelle, benutzerfreundliche GUI für Edge Detection mit über 15 verschiedenen Algorithmen - von klassischen Methoden bis hin zu modernen Deep Learning-Ansätzen.

## ✨ Features

### 🎯 **Edge Detection Methoden**
- **Klassische Methoden**: Laplacian, Prewitt, Roberts, Scharr
- **Canny Varianten**: Standard, Multi-Scale, Adaptive
- **Deep Learning**: HED (OpenCV & PyTorch), Structured Forests, BDCN
- **GPU-Beschleunigt**: Kornia Canny & Sobel
- **Kombinierte Ansätze**: Gradient Magnitude, Morphological Gradient

### 🖥️ **Streamlit GUI**
- **Intuitive Tabs**: Bildauswahl → Methoden → Einstellungen → Verarbeitung → Vorschau
- **Drag & Drop**: Einfaches Hochladen von Bildern
- **Live-Vorschau**: Testen Sie Methoden vor der Batch-Verarbeitung
- **Progress-Tracking**: Echtzeitfortschritt mit ETA
- **Batch-Download**: ZIP-Download aller Ergebnisse

### ⚙️ **Erweiterte Funktionen**
- **Einheitliche Auflösung**: Automatisches Upscaling auf höchste Eingabe-Auflösung
- **Invertierte Ausgabe**: Weiße Hintergründe, dunkle Kantenjobs
- **Flexible Eingabe**: Ordner-Batch oder einzelne Bilder
- **Konfigurierbare Ausgabe**: Pfad, Format, Namensschema anpassbar

---

## 🚀 **Installation & Start**

### **Einfacher Start (Empfohlen)**
```bash
# 1. Repository klonen oder Dateien herunterladen
# 2. In den Projektordner wechseln
cd edge_detection_tool

# 3. Einfach run.bat doppelklicken oder ausführen:
run.bat
```

Das war's! Die `run.bat` macht alles automatisch:
- ✅ Virtuelle Umgebung erstellen
- ✅ Alle Dependencies installieren  
- ✅ Edge Detection Modelle herunterladen
- ✅ Streamlit GUI starten
- ✅ Browser automatisch öffnen

### **Manuelle Installation (für Experten)**
```bash
# Virtuelle Umgebung
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Dependencies
pip install -r requirements.txt

# Modelle initialisieren
python detectors.py --init-models

# GUI starten
streamlit run streamlit_app.py
```

---

## 📁 **Projektstruktur**

```
edge_detection_tool/
├── 🚀 run.bat                 # Haupt-Startskript
├── 🎨 streamlit_app.py        # Streamlit GUI Hauptanwendung
├── 🔧 detectors.py           # Edge Detection Algorithmen
├── 🎛️ gui_components.py      # Erweiterte GUI-Komponenten
├── 📦 requirements.txt       # Python-Abhängigkeiten
├── 📖 README.md              # Diese Anleitung
├── 📁 images/                # Eingabebilder (Beispiele)
├── 📁 results/               # Ausgabeergebnisse
├── 📁 models/                # Heruntergeladene ML-Modelle
└── 📁 venv/                  # Virtuelle Umgebung (automatisch erstellt)
```

---

## 🎯 **Verwendung**

### **1. 🚀 Start**
- Führen Sie `run.bat` aus
- Browser öffnet sich automatisch mit der GUI
- URL: `http://localhost:8501`

### **2. 📷 Bilder auswählen**
**Option A: Ordner-Batch**
- Tab "📷 Bildauswahl" öffnen
- "Ordner auswählen" wählen
- Pfad eingeben oder durchsuchen
- "🔍 Scannen" klicken

**Option B: Einzelne Bilder**
- "Einzelne Bilder" wählen
- Dateien per Drag & Drop hochladen
- Oder "Browse files" verwenden

### **3. 🔧 Methoden auswählen**
- Tab "🔧 Methoden" öffnen
- Schnell-Auswahl verwenden:
  - ⭐ **Empfohlene**: Beste Balance (HED_PyTorch, Kornia_Canny, MultiScaleCanny)
  - 🚀 **Schnell**: Für schnelle Ergebnisse
  - 🎯 **Qualität**: Höchste Qualität (dauert länger)
- Oder manuell aus Kategorien auswählen

### **4. ⚙️ Einstellungen konfigurieren** (Optional)
- Ausgabeordner festlegen
- Auflösungseinstellungen
- Dateiformat (PNG/JPEG)
- Namensschema anpassen

### **5. 👁️ Vorschau testen** (Optional)
- Tab "👁️ Vorschau" öffnen
- Bild und Methode auswählen
- "🔄 Vorschau generieren" klicken
- Original vs. Ergebnis vergleichen

### **6. 🚀 Batch-Verarbeitung starten**
- Sidebar: "🚀 VERARBEITUNG STARTEN" klicken
- Tab "🚀 Verarbeitung" für Live-Progress
- Fortschritt, ETA und Log verfolgen

### **7. 📥 Ergebnisse herunterladen**
- "📂 Ordner öffnen" für lokalen Zugriff
- "📥 ZIP Download" für alle Ergebnisse
- Einzelne Ergebnisse in der Vorschau-Gallery

---

## 🎨 **Ausgabe-Beispiele**

### **Dateinamen-Schema**
```
Originalname: photo001.jpg
Ergebnisse:
├── photo001_HED_PyTorch.png
├── photo001_Kornia_Canny.png
├── photo001_MultiScaleCanny.png
├── photo001_Scharr.png
└── ...
```

### **Ausgabe-Eigenschaften**
- ✅ **Einheitliche Auflösung**: Alle Ergebnisse in höchster Eingabe-Auflösung
- ✅ **Invertierte Farben**: Weiße Hintergründe, dunkle Kantenjobs
- ✅ **PNG-Format**: Verlustfreie Qualität
- ✅ **Beschreibende Namen**: Methode im Dateinamen

---

## 🔧 **Erweiterte Funktionen**

### **Sidebar-Schnellzugriff**
- 🎛️ **Konfiguration**: Alle wichtigen Einstellungen
- 📦 **Methoden-Presets**: Empfohlene, Schnell, Qualität, Alle
- 📂 **Ausgabe-Pfad**: Schnell konfigurierbar
- 🚀 **Start/Stop**: Verarbeitung kontrollieren

### **Method Performance Guide**
| Methode | Geschwindigkeit | Qualität | Empfehlung |
|---------|----------------|----------|------------|
| **HED_PyTorch** | 🟡 Mittel | 🟢 Sehr hoch | ⭐ Beste Balance |
| **Kornia_Canny** | 🟢 Sehr schnell | 🟢 Hoch | ⚡ Für Speed |
| **MultiScaleCanny** | 🟢 Schnell | 🟢 Sehr hoch | 🎯 Vielseitig |
| **StructuredForests** | 🟡 Mittel | 🟢 Sehr hoch | 🏆 Qualität |
| **BDCN** | 🔴 Langsam | 🟢 Höchste | 🥇 Top-Qualität |
| **Laplacian** | 🟢 Sehr schnell | 🟡 Mittel | ⚡ Quick Test |

### **Empfohlene Workflows**

**🚀 Schneller Test (1-2 Minuten)**
```
Methoden: Kornia_Canny, Laplacian
Bilder: 5-10 Testbilder
Zweck: Erste Einschätzung
```

**⭐ Standard-Workflow (5-10 Minuten)**
```
Methoden: HED_PyTorch, Kornia_Canny, MultiScaleCanny
Bilder: 10-50 Bilder
Zweck: Ausgewogene Ergebnisse
```

**🏆 Qualitäts-Workflow (15-30 Minuten)**
```
Methoden: HED_PyTorch, StructuredForests, BDCN, MultiScaleCanny
Bilder: Wichtige/finale Bilder
Zweck: Beste Ergebnisse
```

---

## 🛠️ **Fehlerbehebung**

### **Häufige Probleme**

**❌ "Python nicht gefunden"**
```bash
# Lösung: Python installieren
https://python.org/downloads
# Bei Installation "Add to PATH" aktivieren!
```

**❌ "Requirements Installation fehlgeschlagen"**
```bash
# Lösung: Manuelle Installation
pip install --upgrade pip
pip install streamlit opencv-python torch kornia
```

**❌ "Streamlit startet nicht"**
```bash
# Lösung: Manueller Start
cd edge_detection_tool
venv\Scripts\activate
streamlit run streamlit_app.py
```

**❌ "Modelle laden fehlgeschlagen"**
```bash
# Lösung: Modelle manuell initialisieren
python detectors.py --init-models
```

**❌ "GUI lädt nicht richtig"**
- Browser-Cache leeren (Ctrl+F5)
- Anderen Browser versuchen
- Firewall/Antivirus prüfen

### **Performance-Optimierung**

**🚀 Für schnellere Verarbeitung:**
- Weniger Methoden auswählen
- Kleinere Eingabe-Auflösung verwenden
- GPU-beschleunigte Methoden bevorzugen
- Parallele Verarbeitung aktivieren

**💾 Für weniger Speicherverbrauch:**
- Bilder einzeln statt Batch verarbeiten
- Kleinere Ziel-Auflösung wählen
- JPEG statt PNG für Ausgabe

---

## 🎓 **Tipps & Tricks**

### **Für beste Ergebnisse:**
1. **Bildqualität**: Hochauflösende, scharfe Eingabebilder verwenden
2. **Methodenwahl**: Mit Vorschau verschiedene Methoden testen
3. **Parametertuning**: Für spezielle Anwendungen Methoden anpassen
4. **Batch-Größe**: Bei großen Mengen in kleinere Batches aufteilen

### **Workflow-Optimierung:**
1. **Vorschau nutzen**: Methoden an 1-2 Testbildern evaluieren
2. **Presets verwenden**: Schnell-Auswahl für typische Anwendungsfälle
3. **Parallele Verarbeitung**: Bei vielen Bildern aktivieren
4. **Ergebnis-Management**: Aussagekräftige Ausgabeordner verwenden

### **Keyboard Shortcuts:**
- `Ctrl+R`: Seite neu laden
- `Ctrl+Shift+R`: Browser-Cache leeren
- `Ctrl+C`: Streamlit beenden (im Terminal)

---

## 📞 **Support & Updates**

### **Bei Problemen:**
1. README erneut durchlesen
2. run.bat erneut ausführen
3. Browser-Cache leeren
4. Neueste Python-Version verwenden

### **Feature-Wünsche:**
Das Tool ist erweiterbar! Neue Edge Detection Methoden können einfach in `detectors.py` hinzugefügt werden.

---

## 🏆 **Fazit**

Das Edge Detection Studio kombiniert:
- ✅ **15+ Edge Detection Algorithmen** in einer Anwendung
- ✅ **Professionelle GUI** mit Streamlit
- ✅ **Zero-Config Installation** über run.bat
- ✅ **Batch-Verarbeitung** mit Progress-Tracking
- ✅ **Konsistente Ausgabe** in einheitlicher Qualität

**Perfekt für:** Bildverarbeitung, Computer Vision, Forschung, Preprocessing, Analyse

---

**🎨 Viel Spaß beim Edge Detection! 🎨**
