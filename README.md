# PDF Viewer

A fast, GPU-accelerated PDF viewer built with Python, PyQt6, and PyOpenGL. Designed for efficient rendering and smooth navigation of PDF documents with high-resolution support.

## Features

*   **GPU-Accelerated Rendering**: OpenGL-powered fast page rendering with texture caching
*   **Dual View Modes**: Single-page for detailed reading, grid view for document overview
*   **Smooth Navigation**: Zoom, pan, scroll with responsive controls
*   **Loading Indicators**: Minimal progress feedback for all operations (PDF loading, grid switching, page navigation)
*   **Auto-Refresh Fix**: Automatic click simulation to resolve fullscreen white screen issues
*   **Asynchronous Processing**: Background thumbnail and page loading for responsive UI
*   **File Association**: Double-click PDF files to open directly
*   **Slideshow Mode**: Automatic page progression with customizable timing intervals

## Controls

*   **Mouse**: Scroll pages, drag to pan, click thumbnails to navigate, middle-click for next page
*   **Keyboard**: `G` for grid view, `F11` for fullscreen, `Ctrl+O` to open files, `S` for slideshow
*   **Grid Sizes**: `1-4` keys or Tab to cycle through 2x2, 3x1, 3x2, 5x1 layouts
*   **Slideshow**: `S` to start/stop, dropdown menu to select timing (5s, 10s, 30s, 60s, 120s)

## Requirements

*   Python 3.6+
*   PyQt6
*   PyOpenGL
*   PyMuPDF

## Installation

1.  **Set up a Python environment**
    It is recommended to use a virtual environment to manage dependencies.

2.  **Install the required packages**
    Open your terminal or command prompt and run the following command:
    ```bash
    pip install PyQt6 PyOpenGL PyMuPDF
    ```

## Usage

1.  **Launch the application**
    ```bash
    python main.py
    ```

2.  **Load a PDF**
    *   Drag & drop a PDF file onto the window
    *   Use `Ctrl+O` to open file dialog
    *   Double-click PDF files if associated with the executable

3.  **Navigate**
    *   **Single View**: Mouse wheel to scroll pages, drag to pan when zoomed
    *   **Grid View**: Press `G` to toggle, use `1-4` keys for different layouts
    *   **Thumbnails**: Click any thumbnail to jump to that page
    *   **Fullscreen**: Press `F11` (auto-fixes white screen issues)
    *   **Slideshow**: Press `S` to start automatic page progression, select timing from dropdown (defaults to 30 seconds)

---

## Build Executable

Compile to standalone executable using PyInstaller:

```bash
pip install pyinstaller
python -m PyInstaller --onefile --windowed --name="GPU PDF Viewer" --clean main.py
```

Binary will be in `dist/` folder. Build on target OS for best compatibility.

---

