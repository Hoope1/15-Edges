@echo off
echo ========================================
echo    🎨 Edge Detection Studio Setup
echo ========================================
echo.

REM Prüfe ob Python verfügbar ist
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python ist nicht installiert oder nicht im PATH!
    echo Bitte installieren Sie Python 3.8+ von https://python.org
    pause
    exit /b 1
)

echo ✅ Python gefunden
python --version

REM 1) Virtuelle Umgebung anlegen (falls nicht vorhanden)
if not exist venv (
    echo.
    echo 📦 Erstelle virtuelle Umgebung...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ❌ Fehler beim Erstellen der virtuellen Umgebung!
        pause
        exit /b 1
    )
    echo ✅ Virtuelle Umgebung erstellt
) else (
    echo ✅ Virtuelle Umgebung bereits vorhanden
)

REM 2) Aktivieren der virtuellen Umgebung
echo.
echo 🔄 Aktiviere virtuelle Umgebung...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo ❌ Fehler beim Aktivieren der virtuellen Umgebung!
    pause
    exit /b 1
)

REM 3) Pip upgraden (ohne Fehlerabbruch)
echo.
echo 📈 Aktualisiere pip...
python -m pip install --upgrade pip --quiet

REM 4) Requirements installieren
echo.
echo 📚 Installiere Abhängigkeiten...
echo    - OpenCV für Computer Vision
echo    - PyTorch für Deep Learning  
echo    - Kornia für GPU-beschleunigte Filter
echo    - Streamlit für GUI
echo    - Weitere Pakete...
echo.

python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Fehler bei der Installation der Requirements!
    echo Versuche alternative Installation...
    
    REM Fallback: Installiere Pakete einzeln
    echo 📦 Installiere Kern-Pakete einzeln...
    python -m pip install opencv-python opencv-contrib-python
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    python -m pip install kornia
    python -m pip install streamlit
    python -m pip install requests pillow numpy
    
    if %errorlevel% neq 0 (
        echo ❌ Installation fehlgeschlagen!
        pause
        exit /b 1
    )
)

echo ✅ Abhängigkeiten installiert

REM 5) Modelle herunterladen und initialisieren
echo.
echo 🤖 Lade Edge Detection Modelle...
python detectors.py --init-models
if %errorlevel% neq 0 (
    echo ⚠️ Warnung: Einige Modelle konnten nicht geladen werden
    echo Das Tool funktioniert trotzdem mit verfügbaren Methoden
)

REM 6) Erstelle notwendige Ordner
echo.
echo 📁 Erstelle Ordnerstruktur...
if not exist images mkdir images
if not exist results mkdir results
if not exist models mkdir models

REM 7) Prüfe ob streamlit_app.py existiert
if not exist streamlit_app.py (
    echo ❌ streamlit_app.py nicht gefunden!
    echo Bitte stellen Sie sicher, dass alle Dateien vorhanden sind.
    pause
    exit /b 1
)

REM 8) Zeige Systeminformationen
echo.
echo 💻 System-Check:
echo ================
python -c "import sys; print(f'Python: {sys.version.split()[0]}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')" 2>nul || echo "OpenCV: Nicht verfügbar"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>nul || echo "PyTorch: Nicht verfügbar"
python -c "import kornia; print(f'Kornia: {kornia.__version__}')" 2>nul || echo "Kornia: Nicht verfügbar"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')" 2>nul || echo "Streamlit: Nicht verfügbar"

echo.
echo 🎉 Setup abgeschlossen!
echo.
echo 🚀 Starte Edge Detection Studio GUI...
echo    ➜ Browser öffnet sich automatisch
echo    ➜ Zum Beenden: Ctrl+C drücken
echo.

REM 9) Streamlit GUI starten
streamlit run streamlit_app.py --server.headless false --server.port 8501
if %errorlevel% neq 0 (
    echo.
    echo ❌ Fehler beim Starten der GUI!
    echo.
    echo 🔧 Manuelle Fehlerbehebung:
    echo 1. Prüfen Sie ob alle Dateien vorhanden sind
    echo 2. Führen Sie 'pip install streamlit' manuell aus
    echo 3. Starten Sie 'streamlit run streamlit_app.py' manuell
    echo.
    pause
    exit /b 1
)

REM Falls Streamlit beendet wird
echo.
echo 👋 Edge Detection Studio beendet
echo Vielen Dank für die Nutzung!
pause
