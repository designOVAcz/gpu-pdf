# PDF Viewer

A fast, GPU-accelerated PDF viewer built with Python, PyQt6, and PyOpenGL. It is designed for efficient rendering and smooth navigation of PDF documents, especially those with high-resolution images.

## Features

*   **GPU-Accelerated Rendering**: Utilizes OpenGL for fast and efficient page rendering.
*   **Dual View Modes**: Switch between a single-page view for detailed reading and a grid view for quick document overview.
*   **Smooth Navigation**: Zoom, pan, and scroll through documents with ease.
*   **Asynchronous Loading**: Thumbnails and pages are loaded in the background to keep the UI responsive.
*   **Modern Interface**: A clean and intuitive user interface built with PyQt6.

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

1.  **Run the application**
    Navigate to the project directory in your terminal and run:
    ```bash
    python main.py
    ```

2.  **Open a File**
    The application will start, and you can open a PDF file using the "Open" button or by dragging and dropping a file onto the window.

3.  **Navigate the Viewer**
    *   Use the mouse wheel to scroll through pages in single-page view.
    *   Click and drag to pan the document when zoomed in.
    *   Use the buttons on the toolbar to switch between single-page and grid views, and to navigate pages.
