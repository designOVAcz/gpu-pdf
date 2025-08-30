#!/usr/bin/env python3
"""
GPU-Accelerated PDF Viewer
Fast PDF rendering using PyMuPDF + OpenGL acceleration

Features:
- GPU-accelerated rendering with OpenGL textures
- Multiple grid layout modes for optimal viewing:
  * Adaptive: Balances uniform sizing with aspect ratio proportionality
  * Uniform: All grid cells have identical dimensions
  * Proportional: Row heights strictly follow page aspect ratios
- Responsive gap sizing based on widget dimensions
- Priority rendering queue for improved performance
- Keyboard shortcuts: 'L' to cycle layout modes, 'Shift+L' for info
- Zoom-adaptive image quality: Higher quality at higher zoom levels
- Selective thumbnail loading: Only loads thumbnails around current page for large documents
- Smart memory management: Automatically cleans up distant textures

Performance Optimizations:
- Selective loading: For documents >50 pages, only loads thumbnails in ±15 page radius
- Zoom-based quality: Base quality 2.0, scales up to 4.0 at high zoom levels  
- Memory cleanup: Removes textures >20 pages away from current view
- Priority queues: Grid pages rendered at lower quality for speed, single pages at full quality

Grid Layout System:
The grid layout dynamically adjusts row heights based on page aspect ratios,
providing better visual balance than traditional uniform grids. Users can
switch between modes using keyboard shortcuts or the View menu.
"""

import sys
import os
import time
import traceback
from typing import Optional, List, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor
import math

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Please install PyMuPDF: pip install PyMuPDF")
    sys.exit(1)

try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                                QWidget, QLabel, QPushButton, QFileDialog, QSlider,
                                QSpinBox, QProgressBar, QTextEdit, QSplitter, QFrame,
                                QListWidget, QListWidgetItem, QCheckBox, QComboBox,
                                QGroupBox, QGridLayout, QScrollArea, QStatusBar,
                                QToolBar, QMenuBar, QMenu, QMessageBox, QStyle)
    from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QRect, QPointF
    from PyQt6.QtGui import (QPixmap, QImage, QPainter, QFont, QIcon, QKeySequence,
                            QShortcut, QAction, QPalette, QColor, QActionGroup)
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    from PyQt6.QtOpenGL import QOpenGLBuffer, QOpenGLShaderProgram, QOpenGLTexture
except ImportError:
    print("Please install PyQt6: pip install PyQt6 PyQt6-Qt6")
    sys.exit(1)

try:
    from OpenGL.GL import *
    from OpenGL import GL
    import numpy as np
except ImportError:
    print("Please install OpenGL: pip install PyOpenGL PyOpenGL_accelerate numpy")
    sys.exit(1)

# Increase Qt image allocation limit for large PDF pages (default is 256MB)
os.environ['QT_IMAGEIO_MAXALLOC'] = '1024'  # Set to 1GB


class GPUTextureCache:
    """Manages GPU textures for PDF pages"""
    
    def __init__(self, max_textures=20):
        self.textures = {}  # (page_num, quality) -> QOpenGLTexture
        self.page_textures = {}  # page_num -> best_quality_texture (for backwards compatibility)
        self.dimensions = {}  # page_num -> (width, height)
        self.qualities = {}  # (page_num, quality) -> quality_value
        self.max_textures = max_textures
        self.access_order = []  # LRU tracking for (page_num, quality) keys
    
    def get_texture(self, page_num: int, preferred_quality: float = None) -> Optional[QOpenGLTexture]:
        """Get texture for page, optionally preferring a specific quality"""
        if preferred_quality is not None:
            # Look for exact quality match first
            key = (page_num, preferred_quality)
            if key in self.textures:
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.textures[key]
        
        # Fallback to backwards compatibility - get best available texture for page
        if page_num in self.page_textures:
            # Find the key for this texture to update access order
            best_texture = self.page_textures[page_num]
            for key, texture in self.textures.items():
                if key[0] == page_num and texture == best_texture:
                    self.access_order.remove(key)
                    self.access_order.append(key)
                    break
            return best_texture
        return None
    
    def get_dimensions(self, page_num: int) -> Optional[tuple]:
        """Get the dimensions (width, height) for a page"""
        return self.dimensions.get(page_num)
    
    def add_texture(self, page_num: int, image: QImage, width: float = None, height: float = None, quality: float = 2.0) -> QOpenGLTexture:
        """Add texture with quality tracking"""
        key = (page_num, quality)
        
        # Remove oldest if at capacity
        if len(self.textures) >= self.max_textures and key not in self.textures:
            oldest_key = self.access_order.pop(0)
            if oldest_key in self.textures:
                self.textures[oldest_key].destroy()
                del self.textures[oldest_key]
                if oldest_key in self.qualities:
                    del self.qualities[oldest_key]
                # Update page_textures if this was the best quality for this page
                if oldest_key[0] in self.page_textures:
                    page_num_old = oldest_key[0]
                    # Find next best quality texture for this page
                    best_quality = 0
                    best_texture = None
                    for (p, q), texture in self.textures.items():
                        if p == page_num_old and q > best_quality:
                            best_quality = q
                            best_texture = texture
                    if best_texture:
                        self.page_textures[page_num_old] = best_texture
                    else:
                        del self.page_textures[page_num_old]
        
        # Create new texture with proper format for ARGB32
        texture = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
        texture.setData(image)
        texture.setMinificationFilter(QOpenGLTexture.Filter.Linear)
        texture.setMagnificationFilter(QOpenGLTexture.Filter.Linear)
        texture.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
        
        self.textures[key] = texture
        self.qualities[key] = quality
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        # Update best texture for this page
        if page_num not in self.page_textures or quality > self.get_best_quality_for_page(page_num):
            self.page_textures[page_num] = texture
        
        # Store dimensions (use image dimensions if not provided)
        if width is None:
            width = image.width()
        if height is None:
            height = image.height()
        self.dimensions[page_num] = (width, height)
        
        return texture
    
    def get_best_quality_for_page(self, page_num: int) -> float:
        """Get the highest quality available for a page"""
        best_quality = 0.0
        for (p, q) in self.qualities.keys():
            if p == page_num and q > best_quality:
                best_quality = q
        return best_quality
    
    def remove_texture(self, page_num: int):
        """Remove all texture versions for a specific page from cache"""
        keys_to_remove = []
        for key in self.textures.keys():
            if key[0] == page_num:  # key is (page_num, quality)
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in self.textures:
                self.textures[key].destroy()
                del self.textures[key]
            if key in self.access_order:
                self.access_order.remove(key)
            if key in self.qualities:
                del self.qualities[key]
        
        if page_num in self.page_textures:
            del self.page_textures[page_num]
        if page_num in self.dimensions:
            del self.dimensions[page_num]
    
    def clear(self):
        for texture in self.textures.values():
            texture.destroy()
        self.textures.clear()
        self.page_textures.clear()
        self.qualities.clear()
        self.dimensions.clear()
        self.access_order.clear()


class ThumbnailWorker(QThread):
    """QThread worker to generate thumbnails and emit QImage to the main thread."""
    thumbnailReady = pyqtSignal(int, QImage)
    finished = pyqtSignal()

    def __init__(self, pdf_doc, page_list=None, limit=50, parent=None):
        super().__init__(parent)
        self.pdf_doc = pdf_doc
        self.limit = limit
        self.page_list = page_list  # Specific pages to load, or None for sequential
        self._running = True

    def run(self):
        try:
            if self.page_list:
                # Load specific pages
                pages_to_process = self.page_list[:self.limit]
            else:
                # Load sequential pages from start
                total = min(self.pdf_doc.page_count if self.pdf_doc else 0, self.limit)
                pages_to_process = list(range(total))
            
            for page_num in pages_to_process:
                if not self._running:
                    break
                try:
                    if page_num < 0 or page_num >= self.pdf_doc.page_count:
                        continue
                    page = self.pdf_doc[page_num]
                    mat = fitz.Matrix(0.25, 0.25)  # Reduced from 0.3 for faster rendering
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    qimage = QImage.fromData(img_data, "PNG")
                    if not qimage.isNull():
                        # Emit raw QImage to main thread; build GUI objects there
                        self.thumbnailReady.emit(page_num, qimage)
                    # Reduced sleep to minimize freeze
                    time.sleep(0.001)
                except Exception as inner_e:
                    print(f"Thumbnail worker page error ({page_num}): {inner_e}")
        except Exception as e:
            print(f"Thumbnail worker error: {e}")
        finally:
            self.finished.emit()

    def stop(self):
        self._running = False


class PDFPageRenderer(QThread):
    """Background page renderer that converts PDF pages to QImage and emits pageRendered."""
    pageRendered = pyqtSignal(int, QImage, float)  # Added quality parameter

    def __init__(self, pdf_path: str, quality: float = 2.0, parent=None):
        super().__init__(parent)
        self.pdf_path = pdf_path
        self.quality = quality
        self._running = True
        self._queue = []
        self._priority_queue = []  # High priority queue for grid pages
        self._lock = threading.Lock()

    # Delay opening the document until the thread runs to avoid blocking the UI
        # (initialized in __init__ above)

    def add_page_to_queue(self, page_num: int, priority=False, quality=None):
        with self._lock:
            # Check if page is already in either queue
            priority_pages = [p[0] if isinstance(p, tuple) else p for p in self._priority_queue]
            regular_pages = [p[0] if isinstance(p, tuple) else p for p in self._queue]
            
            if page_num in priority_pages or page_num in regular_pages:
                return  # Already queued
                
            if priority:
                # Add to priority queue (for grid pages) with lower quality for speed
                self._priority_queue.append((page_num, quality or 1.0))
            else:
                # Add to regular queue with normal quality
                self._queue.append((page_num, quality or self.quality))

    def stop(self):
        self._running = False

    def run(self):
        try:
            # Open the document inside the thread to avoid UI-thread file I/O
            try:
                self.pdf_doc = fitz.open(self.pdf_path)
            except Exception as e:
                print(f"PDFPageRenderer: failed to open {self.pdf_path}: {e}")
                self.pdf_doc = None
            while self._running:
                page_info = None
                with self._lock:
                    # Process priority queue first (grid pages)
                    if self._priority_queue:
                        page_info = self._priority_queue.pop(0)
                    elif self._queue:
                        page_info = self._queue.pop(0)

                if page_info is None:
                    # Sleep much less when there's nothing to do - check queue more frequently
                    time.sleep(0.005)  # Reduced from 0.02 to 0.005
                    continue

                # Extract page number and quality
                if isinstance(page_info, tuple):
                    page_num, render_quality = page_info
                else:
                    # Backwards compatibility
                    page_num = page_info
                    render_quality = self.quality

                try:
                    if not self.pdf_doc or page_num < 0 or page_num >= self.pdf_doc.page_count:
                        continue

                    page = self.pdf_doc[page_num]
                    mat = fitz.Matrix(render_quality, render_quality)  # Use per-page quality
                    pix = page.get_pixmap(matrix=mat)
                    # Use PNG bytes which QImage understands reliably
                    img_data = pix.tobytes("png")
                    qimage = QImage.fromData(img_data, "PNG")
                    if qimage and not qimage.isNull():
                        self.pageRendered.emit(page_num, qimage, render_quality)

                except Exception as e:
                    # Continue processing remaining pages on error
                    print(f"Renderer error rendering page {page_num}: {e}")

        finally:
            try:
                if self.pdf_doc:
                    self.pdf_doc.close()
            except Exception:
                pass


class GPUPDFWidget(QOpenGLWidget):
    """OpenGL widget for GPU-accelerated PDF rendering"""
    
    def __init__(self):
        super().__init__()
        self.texture_cache = GPUTextureCache()
        self.current_page = 0
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.page_texture = None
        self.page_width = 1.0
        self.page_height = 1.0
        
        # Grid view attributes
        self.grid_mode = False
        self.grid_cols = 1
        self.grid_rows = 1
        self.grid_textures = {}  # page_num -> texture
        # Cached grid layout: list of dicts with pixel positions and inner sizes
        self._grid_cells = []
        self._grid_cached_size = (0, 0, 0, 0)  # (w_px, h_px, cols, rows)
        # Queue of (page_num, QImage, page_width, page_height, quality) to convert to textures
        self._pending_images = []
        
        # Grid layout configuration
        self.grid_layout_mode = 'adaptive'  # 'uniform', 'adaptive', or 'proportional'
        self.grid_gap_ratio = 0.01  # Gap as percentage of widget size (1% instead of 2%)
        self.grid_min_gap = 5.0     # Minimum gap in pixels (reduced from 10)
        self.grid_max_gap = 15.0    # Maximum gap in pixels (reduced from 30)
        self.grid_aspect_weight = 0.7  # How much to weight aspect ratios (0.0-1.0)
        
        # Quality management based on zoom
        self.base_quality = 2.0     # Base rendering quality
        self.quality_zoom_threshold = 1.5  # Zoom level where quality starts increasing
        self.max_quality = 4.0      # Maximum quality at high zoom levels
        self.last_zoom_quality = self.base_quality  # Track quality changes
            
        # Mouse interaction
        self.last_mouse_pos = QPointF()
        self.is_panning = False
        
        # Animation timing for loading state
        self.animation_time = 0.0
        
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    def initializeGL(self):
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # Standard alpha blending
        glEnable(GL_MULTISAMPLE)  # Enable multisampling if available
        
        # Set background color based on theme
        app = QApplication.instance()
        if app and app.palette().color(QPalette.ColorRole.Window).lightness() < 128:
            # Dark theme
            glClearColor(0.2, 0.2, 0.2, 1.0)
        else:
            # Light theme
            glClearColor(0.9, 0.9, 0.9, 1.0)
    
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Update animation time for loading indicators
        import time
        self.animation_time = time.time()
        
        # Process more pending images per frame for faster grid loading
        try:
            self.process_pending_images(max_items=8)  # Increased from 1 to 8
        except Exception:
            pass
        
        if self.grid_mode:
            self.render_grid_view()
        else:
            self.render_single_page()
    
    def render_single_page(self):
        """Render a single page with loading state support"""
        glLoadIdentity()
        
        # If no texture available, show loading state only if a document is loaded
        if not self.page_texture or not self.page_texture.isCreated():
            viewer = self.window()
            # Check if the main window has a document loaded
            if hasattr(viewer, 'pdf_doc') and viewer.pdf_doc is not None:
                self.render_loading_state()
            return
        
        # Calculate aspect ratio correction
        widget_width = self.width()
        widget_height = self.height()
        
        if widget_width > 0 and widget_height > 0 and self.page_width > 0 and self.page_height > 0:
            widget_aspect = widget_width / widget_height
            page_aspect = self.page_width / self.page_height
            
            # Scale to fit the page in the viewport
            if widget_aspect > page_aspect:
                # Widget is wider - fit page height, letterbox sides
                scale_x = page_aspect / widget_aspect
                scale_y = 1.0
            else:
                # Widget is taller - fit page width, letterbox top/bottom
                scale_x = 1.0
                scale_y = widget_aspect / page_aspect
            
            # Apply zoom and pan transformations
            glTranslatef(self.pan_x, self.pan_y, 0.0)
            glScalef(self.zoom_factor * scale_x, self.zoom_factor * scale_y, 1.0)
        else:
            # Fallback to simple scaling
            glTranslatef(self.pan_x, self.pan_y, 0.0)
            glScalef(self.zoom_factor, self.zoom_factor, 1.0)
        
        # Bind texture and draw quad
        self.page_texture.bind()
        
        glBegin(GL_QUADS)
        # Texture coordinates for PPM image (180° rotation fix)
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, -1.0)
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, -1.0)
        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, 1.0)
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, 1.0)
        glEnd()
        
        self.page_texture.release()
    
    def render_loading_state(self):
        """Render a subtle spinner wheel loading indicator"""
        import math
        
        # Small, subtle spinner wheel in the bottom-right corner
        spinner_size = 0.08  # Small size
        spinner_x = 0.8      # Position near right edge
        spinner_y = -0.8     # Position near bottom edge
        
        # Subtle gray color with low opacity
        glColor4f(0.5, 0.5, 0.5, 0.4)  # Light gray, semi-transparent
        
        # Draw spinning wheel with 8 segments
        num_segments = 8
        segment_angle = 2.0 * math.pi / num_segments
        
        for i in range(num_segments):
            # Calculate rotation based on time
            rotation = self.animation_time * 4.0  # Moderate spin speed
            angle = i * segment_angle + rotation
            
            # Fade each segment based on position (creates spinning effect)
            fade = (i / num_segments) * 0.8 + 0.2  # Range from 0.2 to 1.0
            glColor4f(0.4, 0.4, 0.4, fade * 0.3)  # Very subtle
            
            # Calculate segment positions
            inner_radius = spinner_size * 0.3
            outer_radius = spinner_size
            
            x1 = spinner_x + math.cos(angle) * inner_radius
            y1 = spinner_y + math.sin(angle) * inner_radius
            x2 = spinner_x + math.cos(angle) * outer_radius
            y2 = spinner_y + math.sin(angle) * outer_radius
            x3 = spinner_x + math.cos(angle + segment_angle * 0.8) * outer_radius
            y3 = spinner_y + math.sin(angle + segment_angle * 0.8) * outer_radius
            x4 = spinner_x + math.cos(angle + segment_angle * 0.8) * inner_radius
            y4 = spinner_y + math.sin(angle + segment_angle * 0.8) * inner_radius
            
            # Draw segment as a quad
            glBegin(GL_QUADS)
            glVertex2f(x1, y1)
            glVertex2f(x2, y2)
            glVertex2f(x3, y3)
            glVertex2f(x4, y4)
            glEnd()
        
        glColor3f(1.0, 1.0, 1.0)  # Reset to white
        
        # Keep animating by requesting another frame
        self.update()
    
    def render_grid_view(self):
        """Render multiple pages in a grid layout with proper aspect ratios"""
        # Allow rendering even if textures dict is empty; slots are reserved and filled as textures arrive
        
        glLoadIdentity()
        
        # Apply zoom and pan transformations for grid view
        glTranslatef(self.pan_x, self.pan_y, 0.0)
        glScalef(self.zoom_factor, self.zoom_factor, 1.0)
        
        # Prefer the viewer's current_page/total_pages when available
        viewer = self.window()
        start_page = getattr(viewer, 'current_page', self.current_page)
        total_pages = getattr(viewer, 'total_pages', getattr(self, 'total_pages', 0))

        pages_needed = self.grid_cols * self.grid_rows

        # Recompute grid layout only when widget size or grid dims change
        w_px = max(1, self.width())
        h_px = max(1, self.height())
        if self._grid_cached_size != (w_px, h_px, self.grid_cols, self.grid_rows):
            self.compute_grid_layout()

        # Iterate over every grid slot using precomputed cell rectangles
        for idx in range(pages_needed):
            row = idx // self.grid_cols
            col = idx % self.grid_cols

            page_num = start_page + idx
            # If page index is beyond document end, leave cell empty
            if page_num >= total_pages:
                continue
            texture = self.grid_textures.get(page_num)

            # Get cached cell (pixel coords)
            if idx < len(self._grid_cells):
                cell = self._grid_cells[idx]
            else:
                # Shouldn't happen but skip defensively
                continue

            page_x_px = cell['page_x_px']
            page_y_px = cell['page_y_px']
            inner_w_px = cell['inner_w_px']
            inner_h_px = cell['inner_h_px']

            # Skip drawing placeholder - let background show through for cleaner experience
            # Pages will pop in when textures are ready

            # If we have a valid texture, draw it (it will 'pop in')
            try:
                is_texture_valid = bool(texture) and hasattr(texture, 'isCreated') and texture.isCreated()
            except Exception:
                is_texture_valid = False

            if not is_texture_valid:
                continue

            # Compute fitted quad based on page aspect if available
            page_dims = self.texture_cache.get_dimensions(page_num)
            if page_dims:
                page_w, page_h = page_dims
                page_aspect = (page_w / page_h) if page_h > 0 else 1.0
            else:
                page_aspect = 1.0

            # Use the gaps calculated in compute_grid_layout directly
            # No margin needed - uniform grid handles spacing
            inner_effective_w = max(1.0, inner_w_px)
            inner_effective_h = max(1.0, inner_h_px)

            fit_w = inner_effective_w
            fit_h = fit_w / page_aspect
            if fit_h > inner_effective_h:
                fit_h = inner_effective_h
                fit_w = fit_h * page_aspect

            fitted_x_px = page_x_px + (inner_w_px - fit_w) / 2.0
            fitted_y_px = page_y_px + (inner_h_px - fit_h) / 2.0

            px_to_gl_x = lambda px: (px / w_px) * 2.0 - 1.0
            px_to_gl_y = lambda py: 1.0 - (py / h_px) * 2.0

            # Calculate quad position with proper vertical centering
            quad_x = px_to_gl_x(fitted_x_px)
            quad_y = px_to_gl_y(fitted_y_px)  # Top of the fitted area
            quad_width = (fit_w / w_px) * 2.0
            quad_height = (fit_h / h_px) * 2.0

            try:
                glColor3f(1.0, 1.0, 1.0)  # Ensure white color for texture rendering
                texture.bind()
                glBegin(GL_QUADS)
                # Draw quad with proper OpenGL coordinates (Y-axis flipped from pixel coords)
                glTexCoord2f(0.0, 1.0); glVertex2f(quad_x, quad_y - quad_height)  # Bottom-left
                glTexCoord2f(1.0, 1.0); glVertex2f(quad_x + quad_width, quad_y - quad_height)  # Bottom-right  
                glTexCoord2f(1.0, 0.0); glVertex2f(quad_x + quad_width, quad_y)  # Top-right
                glTexCoord2f(0.0, 0.0); glVertex2f(quad_x, quad_y)  # Top-left
                glEnd()
                try:
                    texture.release()
                except Exception:
                    pass
            except Exception:
                # Skip drawing if texture binding fails
                pass
    
    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        # Use simple orthographic projection for grid view
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
            
        glMatrixMode(GL_MODELVIEW)
        # Recompute grid layout on resize
        try:
            self.compute_grid_layout()
        except Exception:
            pass

    def compute_grid_layout(self):
        """Computes a visually balanced grid by making cells proportional to content."""
        w_px = max(1, self.width())
        h_px = max(1, self.height())
        cols = max(1, self.grid_cols)
        rows = max(1, self.grid_rows)
        
        viewer = self.window()
        start_page = getattr(viewer, 'current_page', self.current_page)
        total_pages = getattr(viewer, 'total_pages', getattr(self, 'total_pages', 0))
        
        # Get average aspect ratio of visible pages
        page_aspects = []
        for idx in range(rows * cols):
            page_num = start_page + idx
            if page_num >= total_pages:
                page_aspects.append(0.75) # Default to a portrait-like aspect
                continue
            page_dims = self.texture_cache.get_dimensions(page_num)
            aspect = (page_dims[0] / page_dims[1]) if page_dims and page_dims[1] > 0 else 0.75
            page_aspects.append(aspect)
        avg_aspect = sum(page_aspects) / len(page_aspects) if page_aspects else 0.75

        gap = 20.0
        
        # Calculate cell dimensions based on average aspect ratio
        avail_w = w_px - (cols + 1) * gap
        avail_h = h_px - (rows + 1) * gap
        
        if avail_w <= 0 or avail_h <= 0:
            self._grid_cells = []
            return

        # Determine cell size that fits available space while maintaining aspect ratio
        cell_w = avail_w / cols
        cell_h = cell_w / avg_aspect if avg_aspect > 0 else cell_w

        if cell_h * rows > avail_h:
            cell_h = avail_h / rows
            cell_w = cell_h * avg_aspect

        # Center the final grid
        total_content_w = cell_w * cols
        total_content_h = cell_h * rows
        grid_offset_x = (w_px - (total_content_w + (cols + 1) * gap)) / 2.0
        grid_offset_y = (h_px - (total_content_h + (rows + 1) * gap)) / 2.0

        cells = []
        current_y = grid_offset_y + gap
        for _ in range(rows):
            current_x = grid_offset_x + gap
            for _ in range(cols):
                cells.append({
                    'page_x_px': current_x,
                    'page_y_px': current_y,
                    'inner_w_px': cell_w,
                    'inner_h_px': cell_h,
                })
                current_x += cell_w + gap
            current_y += cell_h + gap

        self._grid_cells = cells
        self._grid_cached_size = (w_px, h_px, cols, rows)

    def cleanup_distant_textures(self, keep_distance=20):
        """Remove textures that are far from current view to save memory"""
        if not hasattr(self, 'window') or not self.window():
            return
            
        viewer = self.window()
        if not hasattr(viewer, 'current_page'):
            return
        
        current_page = viewer.current_page
        
        # Calculate pages to keep in cache
        if self.grid_mode:
            # In grid mode, keep textures around current grid area
            grid_size = self.grid_cols * self.grid_rows
            keep_start = max(0, current_page - keep_distance)
            keep_end = min(getattr(viewer, 'total_pages', 0), current_page + grid_size + keep_distance)
        else:
            # In single page mode, keep textures around current page
            keep_start = max(0, current_page - keep_distance)
            keep_end = min(getattr(viewer, 'total_pages', 0), current_page + keep_distance)
        
        # Remove distant textures
        textures_to_remove = []
        for key in self.texture_cache.textures.keys():
            # Keys are (page_num, quality) tuples
            page_num = key[0] if isinstance(key, tuple) else key
            if page_num < keep_start or page_num >= keep_end:
                textures_to_remove.append(page_num)
        
        # Remove duplicates and clean up textures
        for page_num in set(textures_to_remove):
            self.texture_cache.remove_texture(page_num)
            # Also remove from grid textures if present
            if page_num in self.grid_textures:
                del self.grid_textures[page_num]

    def process_pending_images(self, max_items=1):
        """Convert up to max_items QImage entries to GPU textures per call."""
        layout_needs_update = False
        count = 0
        while self._pending_images and count < max_items:
            try:
                page_num, qimage, page_w, page_h, quality = self._pending_images.pop(0)
                
                # Check if this is new dimension info for grid layout
                had_dimensions = self.texture_cache.get_dimensions(page_num) is not None
                
                # Create texture and store dimensions with quality info
                texture = self.texture_cache.add_texture(page_num, qimage, page_w, page_h, quality)
                
                # If we got new dimensions for a page that might affect grid layout, mark for update
                if self.grid_mode and not had_dimensions:
                    # Check if this page is in the current visible range
                    start_page = getattr(self.window(), 'current_page', self.current_page)
                    pages_needed = self.grid_cols * self.grid_rows
                    if start_page <= page_num < start_page + pages_needed:
                        layout_needs_update = True
                
                # If grid mode and page is visible, add to grid_textures
                if self.grid_mode:
                    # Determine if page is within current visible grid range
                    start_page = getattr(self.window(), 'current_page', self.current_page)
                    pages_needed = self.grid_cols * self.grid_rows
                    if start_page <= page_num < start_page + pages_needed:
                        self.grid_textures[page_num] = texture
                else:
                    # Single page mode: update page texture if it matches current page
                    viewer_current_page = getattr(self.window(), 'current_page', self.current_page)
                    if page_num == viewer_current_page:
                        self.set_page_texture(texture, page_w, page_h)
                count += 1
            except Exception as e:
                print(f"Error processing pending image: {e}")
                # swallow and continue
                count += 1
        
        # If we discovered new dimensions that affect the grid, recalculate layout
        if layout_needs_update:
            try:
                # Force recalculation by clearing cache
                self._grid_cached_size = (0, 0, 0, 0)
                self.compute_grid_layout()
            except Exception as e:
                print(f"Error updating grid layout: {e}")
    
    def set_page_texture(self, texture: QOpenGLTexture, page_width: float = 1.0, page_height: float = 1.0):
        self.page_texture = texture
        if texture is None:
            # Reset to default dimensions when clearing texture
            self.page_width = 1.0
            self.page_height = 1.0
        else:
            self.page_width = page_width
            self.page_height = page_height
        # Only trigger repaint if widget has a valid GL context
        try:
            if self.context() is not None:
                self.update()
        except Exception:
            # Context may not be created yet; skip update
            pass
    
    def set_grid_textures(self, textures_dict):
        """Set textures for grid view"""
        self.grid_textures = textures_dict
        try:
            if self.context() is not None:
                self.update()
        except Exception:
            pass
    
    def update_background_color(self):
        """Update OpenGL background color based on current theme"""
        app = QApplication.instance()
        if app and app.palette().color(QPalette.ColorRole.Window).lightness() < 128:
            # Dark theme
            glClearColor(0.2, 0.2, 0.2, 1.0)
        else:
            # Light theme
            glClearColor(0.9, 0.9, 0.9, 1.0)
        self.update()
    
    def get_zoom_adjusted_quality(self, progressive=False):
        """Calculate rendering quality based on current zoom level
        
        Args:
            progressive: If True, returns a lower quality for immediate display
        """
        if self.zoom_factor <= self.quality_zoom_threshold:
            return self.base_quality
        
        # Scale quality with zoom above threshold, but more conservatively
        zoom_excess = self.zoom_factor - self.quality_zoom_threshold
        quality_boost = zoom_excess * 0.3  # Reduced from 0.5 to 0.3 for faster rendering
        adjusted_quality = min(self.base_quality + quality_boost, self.max_quality)
        
        # For progressive loading, return a lower quality first
        if progressive:
            return max(self.base_quality, adjusted_quality * 0.7)
        
        return adjusted_quality
    
    def get_immediate_quality(self):
        """Get quality for immediate display (faster rendering)"""
        # Always use base quality for immediate display, regardless of zoom
        return self.base_quality
    
    def check_quality_change(self):
        """Check if quality needs updating due to zoom change and re-render if needed"""
        new_quality = self.get_zoom_adjusted_quality()
        if abs(new_quality - self.last_zoom_quality) > 0.2:  # Reduced threshold for more responsive updates
            self.last_zoom_quality = new_quality
            # Re-render current content with progressive quality
            viewer = self.window()
            if viewer and hasattr(viewer, 'render_current_page'):
                if self.grid_mode:
                    # For grid mode, just re-render without clearing cache initially
                    viewer.render_current_page()
                else:
                    # For single page mode, implement progressive rendering
                    current_page = getattr(viewer, 'current_page', 0)
                    
                    # Don't clear existing texture if we're zooming in - keep it for immediate display
                    existing_texture = self.texture_cache.get_texture(current_page)
                    if existing_texture and self.zoom_factor > self.quality_zoom_threshold:
                        # Keep current texture visible while rendering higher quality in background
                        pass  # Don't clear the texture
                    else:
                        # Clear texture for zoom out or first render
                        self.texture_cache.remove_texture(current_page)
                        self.set_page_texture(None)
                
                # Trigger progressive re-render
                viewer.render_current_page_progressive()
    
    def render_current_page_progressive(self):
        """Render current page with progressive quality loading"""
        viewer = self.window()
        if not viewer or not hasattr(viewer, 'current_page'):
            return
            
        current_page = viewer.current_page
        
        # First, render at immediate quality if no texture exists
        existing_texture = self.texture_cache.get_texture(current_page)
        if not existing_texture:
            immediate_quality = self.get_immediate_quality()
            if viewer.renderer:
                viewer.renderer.add_page_to_queue(current_page, priority=True, quality=immediate_quality)
        
        # Then, render at full zoom-adjusted quality in background
        if self.zoom_factor > self.quality_zoom_threshold:
            final_quality = self.get_zoom_adjusted_quality()
            if viewer.renderer:
                # Use lower priority for high-quality render so immediate render goes first
                viewer.renderer.add_page_to_queue(current_page, priority=False, quality=final_quality)
    
    def zoom_in(self, cursor_pos=None):
        old_zoom = self.zoom_factor
        self.zoom_factor = min(self.zoom_factor * 1.2, 10.0)
        
        # If cursor position is provided, adjust pan to zoom towards cursor
        if cursor_pos is not None:
            zoom_ratio = self.zoom_factor / old_zoom
            # Convert cursor position to normalized coordinates (-1 to 1)
            cursor_x = (cursor_pos.x() / self.width()) * 2.0 - 1.0
            cursor_y = 1.0 - (cursor_pos.y() / self.height()) * 2.0
            
            # Adjust pan to keep cursor point stable
            self.pan_x = cursor_x + (self.pan_x - cursor_x) * zoom_ratio
            self.pan_y = cursor_y + (self.pan_y - cursor_y) * zoom_ratio
        
        self.check_quality_change()
        self.update()
    
    def zoom_out(self, cursor_pos=None):
        old_zoom = self.zoom_factor
        self.zoom_factor = max(self.zoom_factor / 1.2, 0.1)
        
        # If cursor position is provided, adjust pan to zoom towards cursor
        if cursor_pos is not None:
            zoom_ratio = self.zoom_factor / old_zoom
            # Convert cursor position to normalized coordinates (-1 to 1)
            cursor_x = (cursor_pos.x() / self.width()) * 2.0 - 1.0
            cursor_y = 1.0 - (cursor_pos.y() / self.height()) * 2.0
            
            # Adjust pan to keep cursor point stable
            self.pan_x = cursor_x + (self.pan_x - cursor_x) * zoom_ratio
            self.pan_y = cursor_y + (self.pan_y - cursor_y) * zoom_ratio
        
        self.check_quality_change()
        self.update()
    
    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.check_quality_change()
        self.update()
    
    def wheelEvent(self, event):
        # Zoom with mouse wheel towards cursor position
        cursor_pos = event.position()
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_in(cursor_pos)
        else:
            self.zoom_out(cursor_pos)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = True
            self.last_mouse_pos = event.position()
        elif event.button() == Qt.MouseButton.MiddleButton:
            # Middle mouse click goes to next page
            viewer = self.window()
            if viewer and hasattr(viewer, 'next_page'):
                viewer.next_page()
    
    def mouseMoveEvent(self, event):
        if self.is_panning:
            delta = event.position() - self.last_mouse_pos
            self.pan_x += delta.x() / self.width() * 2.0
            self.pan_y -= delta.y() / self.height() * 2.0
            self.last_mouse_pos = event.position()
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False


class PDFViewer(QMainWindow):
    """Main PDF viewer application"""
    
    def __init__(self):
        super().__init__()
        self.pdf_path = ""
        self.pdf_doc = None
        self.current_page = 0
        self.total_pages = 0
        self.render_quality = 2.0
        
        self.renderer = None
        self.thumbnail_worker = None
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
        self.setup_ui()
        self.setup_shortcuts()
    
    def setup_ui(self):
        self.setWindowTitle("GPU-Accelerated PDF Viewer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel (thumbnails, bookmarks)
        left_panel = self.create_left_panel()
        
        # PDF viewer widget
        self.pdf_widget = GPUPDFWidget()
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.pdf_widget)
        splitter.setSizes([250, 1000])
        
        main_layout.addWidget(splitter)
        
        # Create menu and toolbar
        self.create_menu()
        self.create_toolbar()
        
        # Hide menu bar for cleaner look
        self.menuBar().hide()
        
        # Connect zoom methods to update slider
        original_zoom_in = self.pdf_widget.zoom_in
        original_zoom_out = self.pdf_widget.zoom_out
        original_reset_zoom = self.pdf_widget.reset_zoom
        
        def update_zoom_in(cursor_pos=None):
            original_zoom_in(cursor_pos)
            self.zoom_slider.setValue(int(self.pdf_widget.zoom_factor * 100))
        
        def update_zoom_out(cursor_pos=None):
            original_zoom_out(cursor_pos)
            self.zoom_slider.setValue(int(self.pdf_widget.zoom_factor * 100))
            
        def update_reset_zoom():
            original_reset_zoom()
            self.zoom_slider.setValue(100)
        
        self.pdf_widget.zoom_in = update_zoom_in
        self.pdf_widget.zoom_out = update_zoom_out
        self.pdf_widget.reset_zoom = update_reset_zoom
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready - Use File menu or wizard to open a PDF")
    
    def create_left_panel(self):
        panel = QWidget()
        panel.setFixedWidth(200)
        layout = QVBoxLayout(panel)
        
        # Thumbnails without title
        self.thumbnail_list = QListWidget()
        self.thumbnail_list.setStyleSheet("QListWidget { background-color: rgb(51, 51, 51); border: none; }")
        self.thumbnail_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.thumbnail_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.thumbnail_list.setMovement(QListWidget.Movement.Static)
        self.thumbnail_list.setIconSize(QSize(120, 160))  # Thumbnail size
        self.thumbnail_list.setSpacing(20)
        self.thumbnail_list.itemClicked.connect(self.thumbnail_clicked)
        
        # Connect scroll events to detect when user reaches edges
        scroll_bar = self.thumbnail_list.verticalScrollBar()
        scroll_bar.valueChanged.connect(self._on_thumbnail_scroll)
        
        layout.addWidget(self.thumbnail_list)
        
        return panel
    
    def create_menu(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open PDF...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_pdf_direct)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        fullscreen_action = QAction("Fullscreen", self)
        fullscreen_action.setShortcut(QKeySequence.StandardKey.FullScreen)
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        view_menu.addSeparator()
        
        # Grid layout submenu
        layout_menu = view_menu.addMenu("Grid Layout")
        
        adaptive_action = QAction("Adaptive (Balanced)", self)
        adaptive_action.triggered.connect(lambda: self.set_grid_layout_mode('adaptive'))
        adaptive_action.setCheckable(True)
        adaptive_action.setChecked(True)  # Default mode
        layout_menu.addAction(adaptive_action)
        
        uniform_action = QAction("Uniform (Equal Cells)", self)
        uniform_action.triggered.connect(lambda: self.set_grid_layout_mode('uniform'))
        uniform_action.setCheckable(True)
        layout_menu.addAction(uniform_action)
        
        proportional_action = QAction("Proportional (Aspect-based)", self)
        proportional_action.triggered.connect(lambda: self.set_grid_layout_mode('proportional'))
        proportional_action.setCheckable(True)
        layout_menu.addAction(proportional_action)
        
        # Create action group for mutually exclusive selection
        layout_group = QActionGroup(self)
        layout_group.addAction(adaptive_action)
        layout_group.addAction(uniform_action)
        layout_group.addAction(proportional_action)
        
        # Store references for later updates
        self.layout_actions = {
            'adaptive': adaptive_action,
            'uniform': uniform_action,
            'proportional': proportional_action
        }
        
        layout_menu.addSeparator()
        layout_info_action = QAction("Layout Mode Info... (Shift+L)", self)
        layout_info_action.triggered.connect(self.show_layout_mode_info)
        layout_menu.addAction(layout_info_action)
    
    def create_toolbar(self):
        toolbar = self.addToolBar("Main")

        # Open PDF
        open_action = QAction("📁 Open PDF", self)
        open_action.triggered.connect(self.open_pdf_direct)
        toolbar.addAction(open_action)

        toolbar.addSeparator()

        # Page label
        self.page_label = QLabel("Page: 0 / 0")
        toolbar.addWidget(self.page_label)

        toolbar.addSeparator()

        # Navigation
        prev_action = QAction("◀", self)
        prev_action.triggered.connect(self.prev_page)
        toolbar.addAction(prev_action)

        next_action = QAction("▶", self)
        next_action.triggered.connect(self.next_page)
        toolbar.addAction(next_action)

        toolbar.addSeparator()

        # Grid view toggle
        self.grid_view_action = QAction("⊞ Grid View", self)
        self.grid_view_action.setCheckable(True)
        self.grid_view_action.triggered.connect(self.toggle_grid_view)
        toolbar.addAction(self.grid_view_action)

        toolbar.addSeparator()

        # Grid size selector for single-row view
        self.grid_size_combo = QComboBox()
        self.grid_size_combo.addItems(["2x2", "3x3", "4x4", "5x1", "5x2"])
        self.grid_size_combo.currentTextChanged.connect(self.change_grid_size)
        self.grid_size_combo.setEnabled(False)
        toolbar.addWidget(self.grid_size_combo)

        # Fullscreen action with better matching icon
        fullscreen_action = QAction("⛶", self)
        fullscreen_action.setCheckable(True)
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        fullscreen_action.setToolTip("Toggle Fullscreen Mode (F11)")
        toolbar.addAction(fullscreen_action)

        toolbar.addSeparator()

        # Zoom controls
        zoom_in_action = QAction("🔍+", self)
        zoom_in_action.triggered.connect(self.pdf_widget.zoom_in)
        toolbar.addAction(zoom_in_action)

        zoom_out_action = QAction("🔍-", self)
        zoom_out_action.triggered.connect(self.pdf_widget.zoom_out)
        toolbar.addAction(zoom_out_action)

        reset_zoom_action = QAction("⌂", self)
        reset_zoom_action.triggered.connect(self.pdf_widget.reset_zoom)
        toolbar.addAction(reset_zoom_action)

        toolbar.addSeparator()

        # Zoom slider
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 1000)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setMaximumWidth(150)
        self.zoom_slider.valueChanged.connect(self.zoom_changed)
        toolbar.addWidget(self.zoom_slider)
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Page navigation
        QShortcut(QKeySequence("Left"), self, self.prev_page)
        QShortcut(QKeySequence("Right"), self, self.next_page)
        QShortcut(QKeySequence("Up"), self, self.prev_page)
        QShortcut(QKeySequence("Down"), self, self.next_page)
        QShortcut(QKeySequence("Page Up"), self, self.prev_page)
        QShortcut(QKeySequence("Page Down"), self, self.next_page)
        
        # Zoom shortcuts
        QShortcut(QKeySequence("Ctrl++"), self, self.pdf_widget.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self, self.pdf_widget.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, self.pdf_widget.reset_zoom)
        
        # Grid layout mode shortcuts
        QShortcut(QKeySequence("L"), self, self.cycle_grid_layout_mode)
        QShortcut(QKeySequence("Shift+L"), self, self.show_layout_mode_info)
    
    def cycle_grid_layout_mode(self):
        """Cycle through grid layout modes: adaptive -> uniform -> proportional -> adaptive"""
        modes = ['adaptive', 'uniform', 'proportional']
        current_idx = modes.index(self.pdf_widget.grid_layout_mode)
        next_idx = (current_idx + 1) % len(modes)
        new_mode = modes[next_idx]
        
        self.pdf_widget.grid_layout_mode = new_mode
        
        # Force layout recalculation
        self.pdf_widget._grid_cached_size = (0, 0, 0, 0)
        
        # Show current mode
        mode_descriptions = {
            'adaptive': 'Adaptive (balanced uniform + proportional)',
            'uniform': 'Uniform (all cells same size)',
            'proportional': 'Proportional (row heights match aspect ratios)'
        }
        
        self.status_bar.showMessage(f"Grid Layout: {mode_descriptions[new_mode]}", 3000)
        
        # Trigger repaint if in grid mode
        if self.pdf_widget.grid_mode:
            self.pdf_widget.update()
    
    def show_layout_mode_info(self):
        """Show information about current layout mode and available options"""
        current = self.pdf_widget.grid_layout_mode
        info_text = f"""Current Grid Layout Mode: {current.title()}

Available Modes:
• Adaptive: Balances uniform sizing with aspect ratio proportionality
• Uniform: All grid cells have the same dimensions
• Proportional: Row heights strictly follow page aspect ratios

Settings:
• Gap ratio: {self.pdf_widget.grid_gap_ratio:.1%} of widget size
• Aspect weight: {self.pdf_widget.grid_aspect_weight:.0%} (adaptive mode only)
• Gap range: {self.pdf_widget.grid_min_gap:.0f}px - {self.pdf_widget.grid_max_gap:.0f}px

Press 'L' to cycle through modes."""
        
        QMessageBox.information(self, "Grid Layout Modes", info_text)
    
    def set_grid_layout_mode(self, mode):
        """Set specific grid layout mode and update menu checkmarks"""
        if mode in ['adaptive', 'uniform', 'proportional']:
            self.pdf_widget.grid_layout_mode = mode
            
            # Update menu checkmarks if menu exists
            if hasattr(self, 'layout_actions'):
                for mode_name, action in self.layout_actions.items():
                    action.setChecked(mode_name == mode)
            
            # Force layout recalculation
            self.pdf_widget._grid_cached_size = (0, 0, 0, 0)
            
            # Show current mode
            mode_descriptions = {
                'adaptive': 'Adaptive (balanced uniform + proportional)',
                'uniform': 'Uniform (all cells same size)',
                'proportional': 'Proportional (row heights match aspect ratios)'
            }
            
            self.status_bar.showMessage(f"Grid Layout: {mode_descriptions[mode]}", 3000)
            
            # Trigger repaint if in grid mode
            if self.pdf_widget.grid_mode:
                self.pdf_widget.update()
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def dragEnterEvent(self, event):
        """Handle drag enter events"""
        if event.mimeData().hasUrls():
            # Check if any of the dragged files is a PDF
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith('.pdf'):
                    event.acceptProposedAction()
                    return
        event.ignore()
    
    def dropEvent(self, event):
        """Handle drop events"""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if file_path.lower().endswith('.pdf'):
                        self.load_pdf(file_path)
                        event.acceptProposedAction()
                        self.status_bar.showMessage(f"Dropped: {os.path.basename(file_path)}", 3000)
                        return
        event.ignore()
    
    def open_pdf_direct(self):
        """Open PDF file with minimal dialog implementation"""
        try:
            # Process any pending events to ensure clean state
            QApplication.processEvents()
            
            # Try the static method first (most compatible)
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select PDF File",
                "",  # Let system choose default directory
                "PDF Files (*.pdf);;All Files (*.*)",
                options=QFileDialog.Option.DontUseNativeDialog  # Force Qt dialog
            )
            
            if file_path:
                print(f"Selected file: {file_path}")
                self.load_pdf(file_path)
            else:
                print("No file selected")
                
        except Exception as e:
            print(f"Error in file dialog: {e}")
            self.status_bar.showMessage(f"File dialog error: {str(e)}", 5000)
            
            # Fallback: try with native dialog
            try:
                print("Trying fallback native dialog...")
                file_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select PDF File", 
                    "",
                    "PDF Files (*.pdf)"
                )
                if file_path:
                    self.load_pdf(file_path)
            except Exception as e2:
                print(f"Fallback dialog also failed: {e2}")
                self.status_bar.showMessage("File dialog unavailable", 5000)
    
    def load_pdf(self, pdf_path: str, quality: float = 2.0):
        try:
            # Clean up previous document
            if self.pdf_doc:
                self.pdf_doc.close()
            if self.renderer:
                self.renderer.stop()
                self.renderer.wait()
            if self.thumbnail_worker:
                self.thumbnail_worker.stop()
                self.thumbnail_worker.wait()
            
            # Clear texture cache and grid textures when loading new document
            self.pdf_widget.texture_cache.clear()
            self.pdf_widget.grid_textures.clear()
            self.pdf_widget.set_page_texture(None)  # Clear current page texture
            
            # Open new document
            self.pdf_doc = fitz.open(pdf_path)
            self.pdf_path = pdf_path
            self.total_pages = self.pdf_doc.page_count
            self.current_page = 0
            self.render_quality = quality
            
            # Update UI
            self.setWindowTitle(f"PDF Viewer - {os.path.basename(pdf_path)}")
            self.update_page_label()
            
            # Setup renderer
            self.renderer = PDFPageRenderer(pdf_path, quality)
            self.renderer.pageRendered.connect(self.on_page_rendered)
            
            # Render first page
            self.render_current_page()
            self.renderer.start()
            
            # Update background color for current theme
            self.pdf_widget.update_background_color()
            
            # Generate thumbnails
            self.generate_thumbnails()
            
            # Initialize tracking variables for selective loading
            if not hasattr(self, '_current_thumb_range'):
                self._current_thumb_range = (0, min(30, self.total_pages - 1))  # Initial range
            
            # Show final loaded status
            self.status_bar.showMessage(f"Loaded: {os.path.basename(pdf_path)} ({self.total_pages} pages)")
            
            # Clear status to "Ready" after a brief moment
            QTimer.singleShot(2000, lambda: self.status_bar.showMessage("Ready"))
            
        except Exception as e:
            tb = traceback.format_exc()
            # Show the exception and include a copyable traceback for debugging
            QMessageBox.critical(self, "Error", f"Failed to load PDF:\n{str(e)}\n\nTraceback:\n{tb}")
    
    def render_current_page(self):
        if self.pdf_doc and 0 <= self.current_page < self.total_pages:
            if self.pdf_widget.grid_mode:
                self.render_grid_pages()
            else:
                # Single page mode - check for any available texture first
                texture = self.pdf_widget.texture_cache.get_texture(self.current_page)
                if texture:
                    # We have a texture, set it immediately
                    page = self.pdf_doc[self.current_page]
                    page_width = page.rect.width
                    page_height = page.rect.height
                    self.pdf_widget.set_page_texture(texture, page_width, page_height)
                    
                    # If zoomed in significantly, queue a higher quality version in background
                    if self.pdf_widget.zoom_factor > self.pdf_widget.quality_zoom_threshold:
                        final_quality = self.pdf_widget.get_zoom_adjusted_quality()
                        current_best_quality = self.pdf_widget.texture_cache.get_best_quality_for_page(self.current_page)
                        if final_quality > current_best_quality + 0.3:  # Only re-render if significantly better quality needed
                            if self.renderer:
                                self.renderer.add_page_to_queue(self.current_page, priority=False, quality=final_quality)
                else:
                    # No texture exists - only clear if we don't already have the loading state showing
                    if self.pdf_widget.page_texture is not None:
                        self.pdf_widget.set_page_texture(None)
                    
                    if self.renderer:
                        # Use immediate quality for fast display
                        immediate_quality = self.pdf_widget.get_immediate_quality()
                        self.renderer.add_page_to_queue(self.current_page, priority=True, quality=immediate_quality)
                        
                        # If zoomed in, also queue high-quality version
                        if self.pdf_widget.zoom_factor > self.pdf_widget.quality_zoom_threshold:
                            final_quality = self.pdf_widget.get_zoom_adjusted_quality()
                            self.renderer.add_page_to_queue(self.current_page, priority=False, quality=final_quality)
                
                # Always trigger a repaint
                self.pdf_widget.update()
    
    def render_current_page_progressive(self):
        """Render current page with progressive quality - fast first, then high quality"""
        if not self.pdf_doc or not (0 <= self.current_page < self.total_pages):
            return
            
        if self.pdf_widget.grid_mode:
            # Grid mode uses normal rendering
            self.render_grid_pages()
        else:
            # Progressive single page rendering
            texture = self.pdf_widget.texture_cache.get_texture(self.current_page)
            if texture and self.pdf_widget.zoom_factor <= self.pdf_widget.quality_zoom_threshold:
                # Already have suitable texture for current zoom
                page = self.pdf_doc[self.current_page]
                page_width = page.rect.width
                page_height = page.rect.height
                self.pdf_widget.set_page_texture(texture, page_width, page_height)
            else:
                # Need to render - use progressive approach
                if not texture:
                    # No texture at all - render at immediate quality first
                    self.pdf_widget.set_page_texture(None)
                    if self.renderer:
                        immediate_quality = self.pdf_widget.get_immediate_quality()
                        self.renderer.add_page_to_queue(self.current_page, priority=True, quality=immediate_quality)
                
                # If zoomed in, queue high-quality render
                if self.pdf_widget.zoom_factor > self.pdf_widget.quality_zoom_threshold:
                    if self.renderer:
                        final_quality = self.pdf_widget.get_zoom_adjusted_quality()
                        # Add to regular queue with lower priority
                        self.renderer.add_page_to_queue(self.current_page, priority=False, quality=final_quality)
                
                self.pdf_widget.update()
    
    def render_grid_pages(self):
        """Render multiple pages for grid view with zoom-adjusted quality"""
        if not self.pdf_doc:
            return
        
        grid_textures = {}
        pages_needed = self.pdf_widget.grid_cols * self.pdf_widget.grid_rows
        
        # Get zoom-adjusted quality from the widget
        grid_quality = self.pdf_widget.get_zoom_adjusted_quality()
        # For grid view, use slightly lower quality for performance unless zoomed in
        if self.pdf_widget.zoom_factor < 1.5:
            grid_quality = max(1.0, grid_quality * 0.8)  # 80% of calculated quality
        
        # Collect pages starting from current page
        for i in range(pages_needed):
            page_num = self.current_page + i
            if page_num >= self.total_pages:
                break
            
            # Check cache first
            texture = self.pdf_widget.texture_cache.get_texture(page_num)
            if texture:
                grid_textures[page_num] = texture
            else:
                # Queue for high-priority rendering with zoom-adjusted quality
                if self.renderer:
                    self.renderer.add_page_to_queue(page_num, priority=True, quality=grid_quality)
        
        # Set grid textures
        self.pdf_widget.set_grid_textures(grid_textures)
        
        # If we don't have all textures, they will be added when rendering completes
        if len(grid_textures) < pages_needed:
            self.pdf_widget.update()
    
    def generate_thumbnails(self, around_page=None, radius=15, preserve_scroll=True):
        """Generate thumbnails selectively around current page or specified page
        
        Args:
            around_page: Page number to center loading around (uses current_page if None)
            radius: Number of pages to load before/after the center page
            preserve_scroll: Whether to preserve current scroll position in thumbnail list
        """
        if not self.pdf_doc:
            return
        
        # If this is first load or user explicitly requested all pages, load everything
        if not hasattr(self, '_selective_thumbnails_enabled'):
            self._selective_thumbnails_enabled = self.total_pages > 50  # Enable for large docs
        
        # Store current scroll position before modifying thumbnails
        current_scroll_pos = None
        if preserve_scroll and self._selective_thumbnails_enabled and self.thumbnail_list.count() > 0:
            current_scroll_pos = self.thumbnail_list.verticalScrollBar().value()
        
        # Determine center page and calculate page range to load
        center_page = around_page if around_page is not None else self.current_page
        
        if not self._selective_thumbnails_enabled:
            # Load all thumbnails for smaller documents
            start_page = 0
            end_page = self.total_pages - 1
            preserve_scroll = False  # No need to preserve scroll for small docs
        else:
            # Calculate selective page range
            start_page = max(0, center_page - radius)
            end_page = min(self.total_pages - 1, center_page + radius)
        
        # Store range for scroll restoration and loading indicators
        prev_range = getattr(self, '_current_thumb_range', None)
        self._current_thumb_range = (start_page, end_page)
        
        # For selective loading, check if we can extend current range instead of full reload
        if (self._selective_thumbnails_enabled and preserve_scroll and prev_range):
            prev_start, prev_end = prev_range
            
            # If new range overlaps significantly with current range, extend instead of clearing
            overlap_start = max(start_page, prev_start)
            overlap_end = min(end_page, prev_end)
            overlap_size = max(0, overlap_end - overlap_start + 1)
            current_size = prev_end - prev_start + 1
            
            # If 70% or more overlap, extend range instead of full reload
            if overlap_size >= current_size * 0.7:
                # Load additional pages only
                pages_to_add = []
                for p in range(start_page, end_page + 1):
                    if p < prev_start or p > prev_end:
                        pages_to_add.append(p)
                
                if pages_to_add:
                    # Load additional thumbnails without clearing existing ones
                    self._load_additional_thumbnails(pages_to_add)
                    return
            else:
                # Clear existing thumbnails for full reload
                self.thumbnail_list.clear()
        elif self._selective_thumbnails_enabled:
            # Clear existing thumbnails if doing selective loading
            self.thumbnail_list.clear()
        
        try:
            # Start a worker thread to generate thumbnails and emit signals safely
            if self.thumbnail_worker:
                self.thumbnail_worker.stop()
                self.thumbnail_worker.wait()
            
            # Create worker with limited page range
            pages_to_load = list(range(start_page, end_page + 1))
            self.thumbnail_worker = ThumbnailWorker(
                self.pdf_doc, 
                page_list=pages_to_load, 
                limit=len(pages_to_load), 
                parent=self
            )
            self.thumbnail_worker.thumbnailReady.connect(self._on_thumbnail_ready)
            self.thumbnail_worker.finished.connect(lambda: self._on_thumbnails_finished(current_scroll_pos))
            
            if self._selective_thumbnails_enabled:
                self.status_bar.showMessage(f"Loading thumbnails {start_page+1}-{end_page+1}...")
            else:
                self.status_bar.showMessage("Loading thumbnails...")
            
            self.thumbnail_worker.start()
        except Exception as e:
            print(f"Error starting thumbnail generation: {e}")
    
    def _load_additional_thumbnails(self, page_list):
        """Load additional thumbnails without clearing existing ones"""
        try:
            if self.thumbnail_worker:
                self.thumbnail_worker.stop()
                self.thumbnail_worker.wait()
            
            self.thumbnail_worker = ThumbnailWorker(
                self.pdf_doc, 
                page_list=page_list, 
                limit=len(page_list), 
                parent=self
            )
            self.thumbnail_worker.thumbnailReady.connect(self._on_thumbnail_ready)
            self.thumbnail_worker.finished.connect(lambda: self._on_thumbnails_finished(None))
            
            self.status_bar.showMessage(f"Loading {len(page_list)} additional thumbnails...")
            self.thumbnail_worker.start()
        except Exception as e:
            print(f"Error loading additional thumbnails: {e}")
    
    def _on_thumbnail_ready(self, page_num, img_or_icon, item=None):
        # Accept either (page_num, QImage) from the new worker
        # or (page_num, QIcon, QListWidgetItem) from older code paths.
        try:
            if isinstance(img_or_icon, QImage):
                qimage = img_or_icon
                thumb = qimage.scaled(120, 160, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                pixmap = QPixmap.fromImage(thumb)
                qt_icon = QIcon(pixmap)
                item = QListWidgetItem(qt_icon, f"{page_num + 1}")
                item.setData(Qt.ItemDataRole.UserRole, page_num)
                
                # Insert at correct position if we have a range, otherwise just append
                if (hasattr(self, '_current_thumb_range') and 
                    hasattr(self, '_selective_thumbnails_enabled') and self._selective_thumbnails_enabled):
                    # Find correct position to insert
                    inserted = False
                    for i in range(self.thumbnail_list.count()):
                        existing_item = self.thumbnail_list.item(i)
                        existing_page = existing_item.data(Qt.ItemDataRole.UserRole)
                        if existing_page is not None and existing_page > page_num:
                            self.thumbnail_list.insertItem(i, item)
                            inserted = True
                            break
                        elif existing_page == -2:  # "Load More Below" indicator
                            self.thumbnail_list.insertItem(i, item)
                            inserted = True
                            break
                    if not inserted:
                        # Remove "Load More Below" indicator temporarily if it exists
                        for i in range(self.thumbnail_list.count()):
                            test_item = self.thumbnail_list.item(i)
                            if test_item and test_item.data(Qt.ItemDataRole.UserRole) == -2:
                                self.thumbnail_list.takeItem(i)
                                break
                        self.thumbnail_list.addItem(item)
                else:
                    self.thumbnail_list.addItem(item)
                
                # Only update status every few thumbnails to reduce UI chatter
                if page_num % 5 == 0 or page_num < 5:
                    self.status_bar.showMessage(f"Loading thumbnails... ({page_num+1})")
            else:
                # Assume img_or_icon is a QIcon and item may be provided
                try:
                    if item is not None:
                        self.thumbnail_list.addItem(item)
                    else:
                        # Fallback: create an item with the provided icon
                        qt_icon = img_or_icon if isinstance(img_or_icon, QIcon) else QIcon()
                        item = QListWidgetItem(qt_icon, f"{page_num + 1}")
                        item.setData(Qt.ItemDataRole.UserRole, page_num)
                        self.thumbnail_list.addItem(item)
                    # Only update status every few thumbnails to reduce UI chatter
                    if page_num % 5 == 0 or page_num < 5:
                        self.status_bar.showMessage(f"Loading thumbnails... ({page_num+1})")
                except Exception:
                    pass
        except Exception:
            pass
    
    def _on_thumbnails_finished(self, restore_scroll_pos=None):
        """Handle thumbnail generation completion with optional scroll restoration"""
        try:
            # Restore scroll position if provided
            if restore_scroll_pos is not None and self.thumbnail_list.count() > 0:
                QTimer.singleShot(50, lambda: self.thumbnail_list.verticalScrollBar().setValue(restore_scroll_pos))
            
            # Add loading indicators if in selective mode
            if (hasattr(self, '_selective_thumbnails_enabled') and self._selective_thumbnails_enabled and
                hasattr(self, '_current_thumb_range')):
                self._add_loading_indicators()
            
            self.status_bar.showMessage("Ready")
        except Exception:
            self.status_bar.showMessage("Ready")
    
    def _add_loading_indicators(self):
        """Add visual indicators for more thumbnails above/below current range"""
        if not hasattr(self, '_current_thumb_range'):
            return
        
        start_page, end_page = self._current_thumb_range
        
        # Add "Load More Above" indicator if there are pages before current range
        if start_page > 0:
            load_above_item = QListWidgetItem("⬆️ Load Earlier Pages")
            load_above_item.setData(Qt.ItemDataRole.UserRole, -1)  # Special marker
            load_above_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            load_above_item.setBackground(QColor(70, 70, 70))
            load_above_item.setForeground(QColor(200, 200, 200))
            self.thumbnail_list.insertItem(0, load_above_item)
        
        # Add "Load More Below" indicator if there are pages after current range
        if end_page < self.total_pages - 1:
            load_below_item = QListWidgetItem("⬇️ Load Later Pages")
            load_below_item.setData(Qt.ItemDataRole.UserRole, -2)  # Special marker
            load_below_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            load_below_item.setBackground(QColor(70, 70, 70))
            load_below_item.setForeground(QColor(200, 200, 200))
            self.thumbnail_list.addItem(load_below_item)
    
    def thumbnail_clicked(self, item):
        """Handle thumbnail click to navigate to page or trigger loading"""
        if item:
            page_num = item.data(Qt.ItemDataRole.UserRole)
            
            # Handle special loading indicators
            if page_num == -1:  # Load Earlier Pages
                if hasattr(self, '_current_thumb_range'):
                    start_page, end_page = self._current_thumb_range
                    new_start = max(0, start_page - 20)  # Load 20 pages before
                    self.generate_thumbnails(around_page=(new_start + end_page) // 2, radius=25, preserve_scroll=True)
                return
            elif page_num == -2:  # Load Later Pages
                if hasattr(self, '_current_thumb_range'):
                    start_page, end_page = self._current_thumb_range
                    new_end = min(self.total_pages - 1, end_page + 20)  # Load 20 pages after
                    self.generate_thumbnails(around_page=(start_page + new_end) // 2, radius=25, preserve_scroll=True)
                return
            
            # Handle regular page navigation
            if page_num is not None and 0 <= page_num < self.total_pages:
                # Store the requested page
                old_page = self.current_page
                self.current_page = page_num
                
                # Switch to single page mode if in grid mode
                if self.pdf_widget.grid_mode:
                    self.pdf_widget.grid_mode = False
                    self.pdf_widget.grid_textures.clear()
                    # Update UI to reflect mode change
                    self.grid_view_action.setChecked(False)
                    self.grid_size_combo.setEnabled(False)
                    self.pdf_widget.grid_cols = 1
                    self.pdf_widget.grid_rows = 1
                
                # Update page label immediately
                self.update_page_label()
                
                # Check if we already have any texture for this page (any quality)
                existing_texture = self.pdf_widget.texture_cache.get_texture(page_num)
                if existing_texture:
                    # We have the texture, set it immediately
                    if self.pdf_doc:
                        page = self.pdf_doc[page_num]
                        page_width = page.rect.width
                        page_height = page.rect.height
                        self.pdf_widget.set_page_texture(existing_texture, page_width, page_height)
                    
                    # If zoomed in significantly, queue a higher quality version in background
                    if self.pdf_widget.zoom_factor > self.pdf_widget.quality_zoom_threshold:
                        final_quality = self.pdf_widget.get_zoom_adjusted_quality()
                        current_best_quality = self.pdf_widget.texture_cache.get_best_quality_for_page(page_num)
                        if final_quality > current_best_quality + 0.3:  # Only re-render if significantly better quality needed
                            if self.renderer:
                                self.renderer.add_page_to_queue(page_num, priority=False, quality=final_quality)
                else:
                    # No texture exists - keep current display if possible and render in background
                    # Only clear texture if we're switching pages
                    if old_page != page_num:
                        self.pdf_widget.set_page_texture(None)
                    
                    if self.renderer:
                        # Add with high priority for immediate response
                        immediate_quality = self.pdf_widget.get_immediate_quality()
                        self.renderer.add_page_to_queue(page_num, priority=True, quality=immediate_quality)
                        
                        # Also queue high quality if zoomed
                        if self.pdf_widget.zoom_factor > self.pdf_widget.quality_zoom_threshold:
                            final_quality = self.pdf_widget.get_zoom_adjusted_quality()
                            self.renderer.add_page_to_queue(page_num, priority=False, quality=final_quality)
                
                # Trigger repaint to show changes
                self.pdf_widget.update()
                
                # Force immediate processing of pending images if any exist
                if self.pdf_widget._pending_images:
                    try:
                        self.pdf_widget.process_pending_images(max_items=5)
                    except Exception:
                        pass
                            # Check if we need to load more thumbnails around the new page
                if (hasattr(self, '_selective_thumbnails_enabled') and self._selective_thumbnails_enabled and
                    hasattr(self, '_current_thumb_range')):
                    start_page, end_page = self._current_thumb_range
                    # If navigating near the edge of loaded range, preload more
                    if page_num <= start_page + 3 or page_num >= end_page - 3:
                        self.generate_thumbnails(around_page=page_num, radius=20, preserve_scroll=True)
    
    def _on_thumbnail_scroll(self, value):
        """Handle thumbnail list scrolling to trigger loading more thumbnails"""
        if not (hasattr(self, '_selective_thumbnails_enabled') and self._selective_thumbnails_enabled):
            return
        
        if not hasattr(self, '_current_thumb_range'):
            return
        
        scroll_bar = self.thumbnail_list.verticalScrollBar()
        max_value = scroll_bar.maximum()
        
        # Prevent excessive loading by using a timer
        if not hasattr(self, '_scroll_load_timer'):
            self._scroll_load_timer = QTimer()
            self._scroll_load_timer.setSingleShot(True)
            self._scroll_load_timer.timeout.connect(self._handle_edge_scroll)
        
        if self._scroll_load_timer.isActive():
            self._scroll_load_timer.stop()
        
        # Check if scrolled near top or bottom (within 10% of edges)
        near_top = value <= max_value * 0.1 and value < 50
        near_bottom = value >= max_value * 0.9 and value > max_value - 50
        
        if near_top or near_bottom:
            self._edge_scroll_direction = 'up' if near_top else 'down'
            self._scroll_load_timer.start(300)  # Delay to avoid excessive loading
    
    def _handle_edge_scroll(self):
        """Handle loading more thumbnails when scrolled to edge"""
        if not hasattr(self, '_current_thumb_range') or not hasattr(self, '_edge_scroll_direction'):
            return
        
        start_page, end_page = self._current_thumb_range
        
        if self._edge_scroll_direction == 'up' and start_page > 0:
            # Load earlier pages
            new_start = max(0, start_page - 15)
            center_page = (new_start + end_page) // 2
            self.generate_thumbnails(around_page=center_page, radius=20, preserve_scroll=True)
            
        elif self._edge_scroll_direction == 'down' and end_page < self.total_pages - 1:
            # Load later pages
            new_end = min(self.total_pages - 1, end_page + 15)
            center_page = (start_page + new_end) // 2
            self.generate_thumbnails(around_page=center_page, radius=20, preserve_scroll=True)
    
    def on_page_rendered(self, page_num: int, image: QImage, quality: float = 2.0):
        # Get actual page dimensions from PDF document for accurate aspect ratios
        page_width = image.width()
        page_height = image.height()
        if self.pdf_doc and 0 <= page_num < self.total_pages:
            page = self.pdf_doc[page_num]
            page_width = page.rect.width
            page_height = page.rect.height
        # Enqueue the rendered image for asynchronous texture upload on the GL thread
        try:
            self.pdf_widget._pending_images.append((page_num, image, page_width, page_height, quality))
            # Trigger a repaint so the widget can process pending images
            self.pdf_widget.update()
        except Exception:
            # Fallback: synchronous path (if something unexpected)
            texture = self.pdf_widget.texture_cache.add_texture(page_num, image, page_width, page_height, quality)
            if self.pdf_widget.grid_mode:
                pages_needed = self.pdf_widget.grid_cols * self.pdf_widget.grid_rows
                start_page = self.current_page
                end_page = min(start_page + pages_needed, self.total_pages)
                if start_page <= page_num < end_page:
                    self.pdf_widget.grid_textures[page_num] = texture
                    self.pdf_widget.update()
            else:
                if page_num == self.current_page:
                    self.pdf_widget.set_page_texture(texture, image.width(), image.height())
                    self.pdf_widget.update()
        # Update status when pages are rendered in any mode
        try:
            # Show simple progress if many pages, but less frequently
            rendered = len(self.pdf_widget.texture_cache.textures)
            total = max(1, self.total_pages)
            
            # Only show progress for larger documents and at certain intervals
            if total > 10 and rendered < total:
                # Only update status every 5 pages or at certain percentages
                if rendered % 5 == 0 or rendered == 1 or rendered == total - 1:
                    progress_percent = int((rendered / total) * 100)
                    self.status_bar.showMessage(f"Rendering... {progress_percent}% ({rendered}/{total})")
            elif rendered >= total and total > 1:
                self.status_bar.showMessage("Ready")
        except Exception:
            pass
    
    def prev_page(self):
        if self.pdf_widget.grid_mode:
            # In grid mode, move by grid size
            grid_size = self.pdf_widget.grid_cols * self.pdf_widget.grid_rows
            self.current_page = max(0, self.current_page - grid_size)
        else:
            # Single page mode
            if self.current_page > 0:
                self.current_page -= 1
        
        self.update_page_label()
        self.render_current_page()
        
        # Update thumbnails around new current page if selective loading is enabled
        if hasattr(self, '_selective_thumbnails_enabled') and self._selective_thumbnails_enabled:
            # Check if we need to extend thumbnail range
            if hasattr(self, '_current_thumb_range'):
                start_page, end_page = self._current_thumb_range
                # If we're near the beginning of loaded range, load more earlier pages
                if self.current_page <= start_page + 5 and start_page > 0:
                    self.generate_thumbnails(around_page=self.current_page, radius=20, preserve_scroll=True)
        
        # Clean up distant textures to save memory
        self.pdf_widget.cleanup_distant_textures()
        
        # Preload adjacent pages for smoother navigation
        if self.renderer:
            if self.current_page > 0:
                self.renderer.add_page_to_queue(self.current_page - 1)
            if self.current_page < self.total_pages - 1:
                self.renderer.add_page_to_queue(self.current_page + 1)
    
    def next_page(self):
        if self.pdf_widget.grid_mode:
            # In grid mode, move by grid size
            grid_size = self.pdf_widget.grid_cols * self.pdf_widget.grid_rows
            max_page = self.total_pages - grid_size
            self.current_page = min(max_page, self.current_page + grid_size)
        else:
            # Single page mode
            if self.current_page < self.total_pages - 1:
                self.current_page += 1
        
        self.update_page_label()
        self.render_current_page()
        
        # Update thumbnails around new current page if selective loading is enabled
        if hasattr(self, '_selective_thumbnails_enabled') and self._selective_thumbnails_enabled:
            # Check if we need to extend thumbnail range
            if hasattr(self, '_current_thumb_range'):
                start_page, end_page = self._current_thumb_range
                # If we're near the end of loaded range, load more later pages
                if self.current_page >= end_page - 5 and end_page < self.total_pages - 1:
                    self.generate_thumbnails(around_page=self.current_page, radius=20, preserve_scroll=True)
        
        # Clean up distant textures to save memory
        self.pdf_widget.cleanup_distant_textures()
        
        # Preload adjacent pages for smoother navigation
        if self.renderer:
            if self.current_page > 0:
                self.renderer.add_page_to_queue(self.current_page - 1)
            if self.current_page < self.total_pages - 1:
                self.renderer.add_page_to_queue(self.current_page + 1)
    
    def update_page_label(self):
        if self.pdf_widget.grid_mode:
            # In grid mode, show page range
            grid_size = self.pdf_widget.grid_cols * self.pdf_widget.grid_rows
            start_page = self.current_page + 1
            end_page = min(self.current_page + grid_size, self.total_pages)
            self.page_label.setText(f"Pages: {start_page}-{end_page} / {self.total_pages}")
        else:
            # Single page mode
            self.page_label.setText(f"Page: {self.current_page + 1} / {self.total_pages}")
    
    def zoom_changed(self, value):
        zoom_factor = value / 100.0
        self.pdf_widget.zoom_factor = zoom_factor
        self.pdf_widget.update()
    
    def toggle_grid_view(self):
        """Toggle between single page and grid view"""
        if self.grid_view_action.isChecked():
            self.grid_size_combo.setEnabled(True)
            self.pdf_widget.grid_mode = True
            self.change_grid_size(self.grid_size_combo.currentText())
        else:
            self.grid_size_combo.setEnabled(False)
            self.pdf_widget.grid_mode = False
            self.pdf_widget.grid_cols = 1
            self.pdf_widget.grid_rows = 1
            self.render_current_page()
    
    def change_grid_size(self, size_text):
        """Change the grid layout size"""
        if not self.grid_view_action.isChecked():
            return
        
        # Parse grid size (e.g., "2x2" -> 2, 2)
        try:
            cols, rows = map(int, size_text.split('x'))
            self.pdf_widget.grid_cols = cols
            self.pdf_widget.grid_rows = rows
            # Recompute layout and trigger a repaint with placeholders while rendering happens
            try:
                self.pdf_widget.compute_grid_layout()
            except Exception:
                pass
            self.pdf_widget.update()
            # Kick off rendering of grid pages in background
            self.render_current_page()
        except ValueError:
            pass
    
    def toggle_fullscreen(self, checked):
        if checked:
            self.showFullScreen()
        else:
            self.showNormal()
    
    def closeEvent(self, event):
        if self.renderer:
            self.renderer.stop()
            self.renderer.wait()
        if self.thumbnail_worker:
            self.thumbnail_worker.stop()
            self.thumbnail_worker.wait()
        if self.pdf_doc:
            self.pdf_doc.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("GPU PDF Viewer")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("FastPDF")
    
    # Apply OS dark mode theme
    app.setStyle("Fusion")
    palette = app.palette()
    
    # Check for dark mode preference
    try:
        import winreg
        registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
        key = winreg.OpenKey(registry, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
        dark_mode_value = winreg.QueryValueEx(key, "AppsUseLightTheme")[0]
        is_dark_mode = dark_mode_value == 0  # 0 = dark mode, 1 = light mode
        winreg.CloseKey(key)
    except:
        # Fallback: check if system uses dark mode
        is_dark_mode = False
    
    if is_dark_mode:
        # Dark theme colors
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(53,   53, 53))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    
    app.setPalette(palette)
    
    # Create and show main window
    viewer = PDFViewer()
    viewer.show()
    
    # Load PDF from command line if provided
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]) and sys.argv[1].endswith('.pdf'):
        viewer.load_pdf(sys.argv[1])
    
    # Hide the menu bar as per the change request
    viewer.menuBar().hide()
    
    sys.exit(app.exec())
