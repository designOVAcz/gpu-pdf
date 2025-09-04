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
- Keyboard shortcuts: 
  * 'G' to toggle grid view
  * '1-4' to set specific grid size (2x2, 3x1, 3x2, 5x1)
  * 'Tab' to cycle through grid sizes
  * 'L' to cycle layout modes, 'Shift+L' for info
  * 'F11' to toggle fullscreen mode
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

# Essential imports for startup
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
    from PyQt6.QtWidgets import QDialog, QSlider, QHBoxLayout, QFrame, QPushButton
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

# Reduce Qt image allocation for faster startup
os.environ['QT_IMAGEIO_MAXALLOC'] = '256'  # Minimal allocation for fastest startup


class GPUTextureCache:
    """🚀 LAYER 3: Intelligent GPU texture cache with advanced memory management"""
    
    def __init__(self, max_textures=200):  # MASSIVE increase for modern GPUs (was 20)
        self.textures = {}  # (page_num, quality) -> QOpenGLTexture
        self.page_textures = {}  # page_num -> best_quality_texture (for backwards compatibility)
        self.dimensions = {}  # page_num -> (width, height)
        self.qualities = {}  # (page_num, quality) -> quality_value
        self.max_textures = max_textures
        self.access_order = []  # LRU tracking for (page_num, quality) keys
        self.priority_textures = set()  # High-quality textures to keep longer
        
        # 🚀 EXTREME GPU VRAM USAGE for modern graphics cards
        self.texture_memory_usage = {}  # (page_num, quality) -> estimated memory in bytes
        self.total_memory_usage = 0
        self.max_memory_mb = 3072  # 3GB GPU VRAM limit (was 256MB) for maximum performance
        
        # Performance statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._evictions = 0
    
    def get_memory_usage_mb(self):
        """Get current memory usage in MB"""
        return self.total_memory_usage / (1024 * 1024)
    
    def get_cache_stats(self):
        """Get cache performance statistics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        return {
            'hit_rate': hit_rate,
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'evictions': self._evictions,
            'memory_mb': self.get_memory_usage_mb(),
            'textures': len(self.textures)
        }
    
    def cleanup_texture(self, texture):
        """Safely destroy and cleanup a texture"""
        try:
            if texture and hasattr(texture, 'isCreated') and texture.isCreated():
                texture.destroy()
        except Exception as e:
            print(f"Error cleaning up texture: {e}")
    
    def clear_cache(self):
        """Clear all textures from cache"""
        for texture in self.textures.values():
            self.cleanup_texture(texture)
        for texture in self.page_textures.values():
            self.cleanup_texture(texture)
        self.textures.clear()
        self.page_textures.clear()
        self.dimensions.clear()
        self.qualities.clear()
        self.access_order.clear()
        self.priority_textures.clear()
        self.texture_memory_usage.clear()
        self.total_memory_usage = 0
    
    def mark_grid_textures_as_priority(self, page_nums):
        """🚀 LAYER 3: Smart priority marking with memory awareness"""
        for page_num in page_nums:
            for key in self.textures.keys():
                if key[0] == page_num:
                    self.priority_textures.add(key)
                    # Move priority textures to end of LRU (less likely to be evicted)
                    if key in self.access_order:
                        self.access_order.remove(key)
                        self.access_order.append(key)
    
    def unmark_all_grid_priorities(self):
        """Remove grid priority marking from all textures"""
        # Only keep truly high-quality textures as priority
        self.priority_textures = {key for key in self.priority_textures if self.qualities.get(key, 0) >= 4.5}
    
    def get_texture(self, page_num: int, preferred_quality: float = None) -> Optional[QOpenGLTexture]:
        """🚀 LAYER 3: Intelligent texture retrieval with smart quality matching"""
        if preferred_quality is not None:
            # Look for exact quality match first
            key = (page_num, preferred_quality)
            if key in self.textures:
                self._cache_hits += 1
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.textures[key]
            
            # 🚀 LAYER 3: Smart quality fallback with performance awareness
            if preferred_quality > 2.0:
                best_texture = None
                best_quality = 0.0
                best_key = None
                for (p, q), texture in self.textures.items():
                    if p == page_num:
                        # Accept textures within reasonable quality range
                        quality_diff = abs(q - preferred_quality) / preferred_quality
                        if quality_diff <= 0.3:  # Accept within 30% quality difference
                            if q > best_quality:
                                best_quality = q
                                best_texture = texture
                                best_key = (p, q)
                
                if best_texture and best_key:
                    self._cache_hits += 1
                    # Update access order
                    self.access_order.remove(best_key)
                    self.access_order.append(best_key)
                    return best_texture
        
        # Fallback to backwards compatibility - get best available texture for page
        if page_num in self.page_textures:
            self._cache_hits += 1
            return self.page_textures[page_num]
            
        self._cache_misses += 1
        return None
        
    def add_texture(self, page_num, image, page_width, page_height, quality=1.0):
        """🚀 LAYER 3: Advanced texture addition with intelligent memory management"""
        # Estimate memory usage for this texture
        estimated_memory = image.width() * image.height() * 4  # RGBA = 4 bytes per pixel
        
        # 🚀 LAYER 3: Check memory limits before adding
        if self.total_memory_usage + estimated_memory > self.max_memory_mb * 1024 * 1024:
            self._enforce_memory_limit(estimated_memory)
        
        # Create a new OpenGL texture from the QImage
        texture = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
        
        # Set texture properties BEFORE creating/allocating storage
        texture.setSize(image.width(), image.height())
        texture.setFormat(QOpenGLTexture.TextureFormat.RGBA8_UNorm)
        
        # 🚀 LAYER 3: Intelligent filtering based on quality and usage
        if quality >= 4.0:
            # High quality filtering for crisp text
            texture.setMinificationFilter(QOpenGLTexture.Filter.LinearMipMapLinear)
            texture.setMagnificationFilter(QOpenGLTexture.Filter.Linear)
            texture.generateMipMaps()
        elif quality >= 2.0:
            # Medium quality filtering
            texture.setMinificationFilter(QOpenGLTexture.Filter.Linear)
            texture.setMagnificationFilter(QOpenGLTexture.Filter.Linear)
        else:
            # Fast filtering for thumbnails and previews
            texture.setMinificationFilter(QOpenGLTexture.Filter.Nearest)
            texture.setMagnificationFilter(QOpenGLTexture.Filter.Linear)
            
        texture.setWrapMode(QOpenGLTexture.CoordinateDirection.DirectionS, QOpenGLTexture.WrapMode.ClampToEdge)
        texture.setWrapMode(QOpenGLTexture.CoordinateDirection.DirectionT, QOpenGLTexture.WrapMode.ClampToEdge)
        
        # Now create and allocate storage
        texture.create()
        texture.setData(image)
        
        # Store the texture and update tracking
        key = (page_num, quality)
        
        # Remove old texture if it exists
        if key in self.textures:
            self.cleanup_texture(self.textures[key])
            old_memory = self.texture_memory_usage.get(key, 0)
            self.total_memory_usage -= old_memory
        
        self.textures[key] = texture
        self.dimensions[page_num] = (page_width, page_height)
        self.qualities[key] = quality
        self.page_textures[page_num] = texture
        self.texture_memory_usage[key] = estimated_memory
        self.total_memory_usage += estimated_memory
        
        # Update access order for LRU
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        # Enforce cache size limit
        self._enforce_cache_limit()
        
        return texture
    
    def _enforce_memory_limit(self, needed_memory):
        """🚀 LAYER 3: Intelligent memory-based eviction"""
        while (self.total_memory_usage + needed_memory > self.max_memory_mb * 1024 * 1024 
               and self.access_order):
            # Find least recently used, non-priority texture
            evicted = False
            for i, key in enumerate(self.access_order):
                if key not in self.priority_textures:
                    # Evict this texture
                    texture = self.textures[key]
                    self.cleanup_texture(texture)
                    
                    # Update memory tracking
                    memory_freed = self.texture_memory_usage.get(key, 0)
                    self.total_memory_usage -= memory_freed
                    
                    # Remove from all tracking
                    del self.textures[key]
                    del self.texture_memory_usage[key]
                    del self.qualities[key]
                    self.access_order.remove(key)
                    
                    # Update page_textures if this was the best quality for this page
                    page_num = key[0]
                    if page_num in self.page_textures and self.page_textures[page_num] == texture:
                        # Find next best texture for this page
                        best_texture = None
                        best_quality = 0
                        for (p, q), t in self.textures.items():
                            if p == page_num and q > best_quality:
                                best_quality = q
                                best_texture = t
                        self.page_textures[page_num] = best_texture
                        if best_texture is None:
                            del self.page_textures[page_num]
                    
                    self._evictions += 1
                    evicted = True
                    break
            
            if not evicted:
                # All remaining textures are priority - evict oldest priority texture
                if self.access_order:
                    key = self.access_order[0]
                    texture = self.textures[key]
                    self.cleanup_texture(texture)
                    
                    memory_freed = self.texture_memory_usage.get(key, 0)
                    self.total_memory_usage -= memory_freed
                    
                    del self.textures[key]
                    del self.texture_memory_usage[key]
                    del self.qualities[key]
                    self.access_order.remove(key)
                    self.priority_textures.discard(key)
                    
                    page_num = key[0]
                    if page_num in self.page_textures and self.page_textures[page_num] == texture:
                        del self.page_textures[page_num]
                    
                    self._evictions += 1
                else:
                    break
    
    def _enforce_cache_limit(self):
        """🚀 LAYER 3: Standard cache size limit enforcement"""
        while len(self.textures) > self.max_textures and self.access_order:
            # Find least recently used, non-priority texture
            evicted = False
            for key in self.access_order:
                if key not in self.priority_textures:
                    # Evict this texture
                    texture = self.textures[key]
                    self.cleanup_texture(texture)
                    
                    # Update memory tracking
                    memory_freed = self.texture_memory_usage.get(key, 0)
                    self.total_memory_usage -= memory_freed
                    
                    # Remove from all tracking
                    del self.textures[key]
                    del self.texture_memory_usage[key]
                    del self.qualities[key]
                    self.access_order.remove(key)
                    
                    # Update page_textures if this was the best quality for this page
                    page_num = key[0]
                    if page_num in self.page_textures and self.page_textures[page_num] == texture:
                        # Find next best texture for this page
                        best_texture = None
                        best_quality = 0
                        for (p, q), t in self.textures.items():
                            if p == page_num and q > best_quality:
                                best_quality = q
                                best_texture = t
                        self.page_textures[page_num] = best_texture
                        if best_texture is None:
                            del self.page_textures[page_num]
                    
                    self._evictions += 1
                    evicted = True
                    break
            
            if not evicted:
                # All remaining textures are priority - evict oldest priority texture
                if self.access_order:
                    key = self.access_order[0]
                    texture = self.textures[key]
                    self.cleanup_texture(texture)
                    
                    memory_freed = self.texture_memory_usage.get(key, 0)
                    self.total_memory_usage -= memory_freed
                    
                    del self.textures[key]
                    del self.texture_memory_usage[key]
                    del self.qualities[key]
                    self.access_order.remove(key)
                    self.priority_textures.discard(key)
                    
                    page_num = key[0]
                    if page_num in self.page_textures and self.page_textures[page_num] == texture:
                        del self.page_textures[page_num]
                    
                    self._evictions += 1
                else:
                    break

    def add_texture_fast(self, page_num, image, page_width, page_height, quality=1.0):
        """Add texture with optimized settings for maximum speed - GPU optimized"""
        try:
            # Create texture with speed-optimized settings
            texture = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
            
            # SPEED OPTIMIZED: Set minimal properties for fastest creation
            texture.setSize(image.width(), image.height())
            texture.setFormat(QOpenGLTexture.TextureFormat.RGBA8_UNorm)
            
            # Use fastest filtering - Linear for decent quality but maximum speed
            texture.setMinificationFilter(QOpenGLTexture.Filter.Linear)
            texture.setMagnificationFilter(QOpenGLTexture.Filter.Linear)
            texture.setWrapMode(QOpenGLTexture.CoordinateDirection.DirectionS, QOpenGLTexture.WrapMode.ClampToEdge)
            texture.setWrapMode(QOpenGLTexture.CoordinateDirection.DirectionT, QOpenGLTexture.WrapMode.ClampToEdge)
            
            # Fast creation and data transfer
            if not texture.create():
                return None
                
            texture.setData(image)
            
            # Store with minimal bookkeeping for speed
            key = (page_num, quality)
            self.textures[key] = texture
            self.dimensions[page_num] = (page_width, page_height)
            self.qualities[key] = quality
            self.page_textures[page_num] = texture
            
            # Minimal LRU management - just append
            self.access_order.append(key)
            
            # Only enforce cache limit if we're getting close to prevent expensive operations
            if len(self.textures) > self.max_textures + 2:  # Allow small overflow before cleanup
                self._enforce_cache_limit()
            
            return texture
            
        except Exception as e:
            # Fail gracefully without blocking
            return None
        
    def get_dimensions(self, page_num):
        """Get the dimensions of a page"""
        if page_num in self.dimensions:
            return self.dimensions[page_num]
        return None
    
    def get_best_quality_for_page(self, page_num: int) -> float:
        """Get the highest quality available for a page"""
        best_quality = 0.0
        for (p, q) in self.qualities.keys():
            if p == page_num and q > best_quality:
                best_quality = q
        return best_quality
        
    def remove_texture(self, page_num):
        """🚀 LAYER 3: Enhanced texture removal with memory tracking"""
        keys_to_remove = []
        for (p, q) in self.textures.keys():
            if p == page_num:
                keys_to_remove.append((p, q))
        
        for key in keys_to_remove:
            texture = self.textures[key]
            self.cleanup_texture(texture)
            
            # Update memory tracking
            memory_freed = self.texture_memory_usage.get(key, 0)
            self.total_memory_usage -= memory_freed
            
            # Remove from all tracking structures with safety checks
            del self.textures[key]
            if key in self.texture_memory_usage:
                del self.texture_memory_usage[key]
            if key in self.access_order:
                self.access_order.remove(key)
            if key in self.qualities:
                del self.qualities[key]
            self.priority_textures.discard(key)
        
        # Clean up page_textures entry
        if page_num in self.page_textures:
            del self.page_textures[page_num]
        
        if page_num in self.page_textures:
            del self.page_textures[page_num]
        if page_num in self.dimensions:
            del self.dimensions[page_num]
            
    def clear(self):
        """Clear all textures from the cache"""
        for texture in self.textures.values():
            texture.destroy()
        self.textures.clear()
        self.page_textures.clear()
        self.dimensions.clear()
        self.qualities.clear()
        self.access_order.clear()
        self.priority_textures.clear()
        
    def _enforce_cache_limit(self):
        """Remove least recently used textures if cache exceeds max size"""
        while len(self.textures) > self.max_textures:
            # Find the least recently used texture that's not priority
            for key in self.access_order:
                if key not in self.priority_textures:
                    # Store texture reference before deleting
                    texture_to_check = self.textures[key]
                    # Remove this texture
                    self.textures[key].destroy()
                    del self.textures[key]
                    self.access_order.remove(key)
                    del self.qualities[key]
                    # Also remove from page_textures if it's there
                    page_num = key[0]
                    if page_num in self.page_textures and self.page_textures[page_num] == texture_to_check:
                        del self.page_textures[page_num]
                    break


class BackgroundMatchDialog(QDialog):
    """Simple dialog to tune background RGB luminance so thumbnails and page match."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Background match")
        self.setModal(True)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 255)
        # Default to 38
        self.slider.setValue(38)

        self.preview_thumb = QFrame()
        self.preview_thumb.setFixedSize(80, 40)
        self.preview_page = QFrame()
        self.preview_page.setFixedSize(160, 80)

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_and_close)

        layout = QHBoxLayout(self)
        layout.addWidget(self.slider)
        layout.addWidget(self.preview_thumb)
        layout.addWidget(self.preview_page)
        layout.addWidget(self.save_btn)

        self.slider.valueChanged.connect(self.on_change)
        self.on_change(self.slider.value())

    def on_change(self, v: int):
        col = f"rgb({v},{v},{v})"
        self.preview_thumb.setStyleSheet(f"QFrame {{ background: {col}; }}")
        self.preview_page.setStyleSheet(f"QFrame {{ background: {col}; }}")
        viewer = self.parent()
        if viewer and hasattr(viewer, 'thumbnail_list'):
            viewer.thumbnail_list.setStyleSheet(f"QListWidget {{ background-color: {col}; border: none; }}")
        # update GL clear color if pdf_widget present
        if viewer and hasattr(viewer, 'pdf_widget'):
            # convert to float 0-1
            f = v / 255.0
            try:
                viewer.pdf_widget.set_clear_color(f, f, f, 1.0)
            except Exception:
                pass

    def save_and_close(self):
        # Just close; current settings applied live
        self.accept()
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
    
    def add_texture(self, page_num: int, image: QImage, width: float = None, height: float = None, quality: float = 4.0) -> QOpenGLTexture:
        """Add texture with quality tracking"""
        key = (page_num, quality)
        
        # Remove oldest if at capacity - but prefer to keep high-quality textures
        if len(self.textures) >= self.max_textures and key not in self.textures:
            # Try to remove non-priority textures first
            oldest_key = None
            for candidate_key in self.access_order:
                if candidate_key not in self.priority_textures:
                    oldest_key = candidate_key
                    break
            
            # If all textures are priority, remove the oldest one anyway
            if oldest_key is None:
                oldest_key = self.access_order.pop(0)
            else:
                self.access_order.remove(oldest_key)
                
            if oldest_key in self.textures:
                self.cleanup_texture(self.textures[oldest_key])
                del self.textures[oldest_key]
                self.priority_textures.discard(oldest_key)
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
        texture.setFormat(QOpenGLTexture.TextureFormat.RGBA8_UNorm)
        texture.setSize(image.width(), image.height())
        texture.setMinificationFilter(QOpenGLTexture.Filter.Linear)
        texture.setMagnificationFilter(QOpenGLTexture.Filter.Linear)
        texture.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
        texture.create()
        texture.setData(image)
        
        self.textures[key] = texture
        self.qualities[key] = quality
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        # Mark high-quality textures as priority (keep them longer) - but with higher threshold
        if quality >= 4.5:  # Increased threshold to reduce priority textures
            self.priority_textures.add(key)
        
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
            self.priority_textures.discard(key)  # Remove from priority set
        
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
        self.priority_textures.clear()  # Clear priority set


class ThumbnailWorker(QThread):
    """🚀 LAYER 1: Advanced GPU-optimized thumbnail worker with parallel processing and disk caching"""
    thumbnailReady = pyqtSignal(int, QImage)
    finished = pyqtSignal()

    def __init__(self, pdf_doc, page_list=None, limit=50, parent=None, gpu_widget=None):
        super().__init__(parent)
        self.pdf_doc = pdf_doc
        self.limit = limit
        self.page_list = page_list  # Specific pages to load, or None for sequential
        self._running = True
        self.gpu_widget = gpu_widget  # Reference to GPU widget for texture cache access
        
        # LAYER 1 OPTIMIZATION: Initialize disk cache system
        self._init_disk_cache()
        
        # Performance tracking for adaptive quality
        self._render_times = []
        self._adaptive_quality = 0.4  # Start with reasonable quality

    def stop(self):
        """Stop the thumbnail worker"""
        self._running = False

    def _init_disk_cache(self):
        """Initialize persistent disk cache for thumbnails in centralized location"""
        try:
            if hasattr(self.pdf_doc, 'name') and self.pdf_doc.name:
                # Use centralized cache directory in temp folder
                import tempfile
                import hashlib
                
                # Create centralized cache directory
                base_cache_dir = os.path.join(tempfile.gettempdir(), 'pdfview_thumbnails')
                os.makedirs(base_cache_dir, exist_ok=True)
                
                # Generate unique cache prefix from PDF path hash
                pdf_path = os.path.abspath(self.pdf_doc.name)
                path_hash = hashlib.md5(pdf_path.encode('utf-8')).hexdigest()[:8]
                pdf_name = os.path.splitext(os.path.basename(self.pdf_doc.name))[0]
                
                # Use hash-based subdirectory to avoid conflicts
                self.cache_dir = os.path.join(base_cache_dir, path_hash)
                os.makedirs(self.cache_dir, exist_ok=True)
                
                self.cache_prefix = pdf_name
            else:
                self.cache_dir = None
                self.cache_prefix = None
        except Exception as e:
            print(f"Cache init error: {e}")
            self.cache_dir = None
            self.cache_prefix = None

    def _get_cache_path(self, page_num):
        """Get cache file path for a specific page"""
        if not self.cache_dir or not self.cache_prefix:
            return None
        return os.path.join(self.cache_dir, f"{self.cache_prefix}_thumb_{page_num}.webp")

    def _load_from_cache(self, page_num):
        """Try to load thumbnail from disk cache (ultra-fast)"""
        try:
            cache_path = self._get_cache_path(page_num)
            if cache_path and os.path.exists(cache_path):
                cached_image = QImage(cache_path)
                if not cached_image.isNull():
                    return cached_image
        except Exception as e:
            print(f"Cache load error for page {page_num}: {e}")
        return None

    def _save_to_cache(self, page_num, qimage):
        """Save thumbnail to disk cache for future use"""
        try:
            cache_path = self._get_cache_path(page_num)
            if cache_path:
                # Save as WebP for optimal compression and speed
                qimage.save(cache_path, "WEBP", quality=80)
        except Exception as e:
            print(f"Cache save error for page {page_num}: {e}")

    def run(self):
        """🚀 LAYER 1: Advanced parallel thumbnail generation with intelligent caching"""
        try:
            if self.page_list:
                pages_to_process = self.page_list[:self.limit]
            else:
                total = min(self.pdf_doc.page_count if self.pdf_doc else 0, self.limit)
                pages_to_process = list(range(total))
            
            # LAYER 1 OPTIMIZATION: Smart priority loading - prioritize center pages
            if len(pages_to_process) > 10:
                center = len(pages_to_process) // 2
                # Sort by distance from center for smarter loading
                pages_to_process.sort(key=lambda p: abs(p - center))
                print(f"🧠 Smart loading: {len(pages_to_process)} thumbnails, center priority around page {center}")
            
            # LAYER 1 OPTIMIZATION: Parallel processing with ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import threading
            
            # Use optimal number of threads (2-4) to avoid overwhelming the system
            max_workers = min(4, max(2, len(pages_to_process) // 10))
            
            if max_workers > 1 and len(pages_to_process) > 5:
                print(f"🔥 Parallel processing: {max_workers} workers for {len(pages_to_process)} thumbnails")
                # Parallel processing for better performance
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all pages for parallel processing
                    future_to_page = {
                        executor.submit(self._process_single_thumbnail, page_num): page_num
                        for page_num in pages_to_process
                    }
                    
                    # Process completed thumbnails as they finish
                    for future in as_completed(future_to_page):
                        if not self._running:
                            break
                        
                        page_num = future_to_page[future]
                        try:
                            result = future.result(timeout=5.0)  # 5 second timeout per thumbnail
                            if result:  # If thumbnail was generated successfully
                                pass  # Result already emitted in _process_single_thumbnail
                        except Exception as e:
                            print(f"Parallel thumbnail error for page {page_num}: {e}")
            else:
                # Single-threaded processing for small batches
                print(f"📋 Sequential processing: {len(pages_to_process)} thumbnails")
                for page_num in pages_to_process:
                    if not self._running:
                        break
                    self._process_single_thumbnail(page_num)
                        
        except Exception as e:
            print(f"Thumbnail worker error: {e}")
        finally:
            self.finished.emit()

    def _process_single_thumbnail(self, page_num):
        """Process a single thumbnail with all Layer 1 optimizations"""
        try:
            # LAYER 1 OPTIMIZATION 1: Check disk cache first (fastest possible)
            cached_image = self._load_from_cache(page_num)
            if cached_image:
                self.thumbnailReady.emit(page_num, cached_image)
                return True
            
            # LAYER 1 OPTIMIZATION 2: Check GPU texture cache
            if (self.gpu_widget and hasattr(self.gpu_widget, 'texture_cache')):
                texture = self.gpu_widget.texture_cache.get_texture(page_num)
                if texture:
                    # TODO: Convert GPU texture to thumbnail (future optimization)
                    pass
            
            # LAYER 1 OPTIMIZATION 3: Adaptive quality rendering
            if page_num < 0 or page_num >= self.pdf_doc.page_count:
                return False
                
            import time
            start_time = time.time()
            
            page = self.pdf_doc[page_num]
            
            # Adaptive quality based on recent performance
            if len(self._render_times) > 5:
                avg_time = sum(self._render_times[-5:]) / 5
                if avg_time < 0.02:  # Very fast system
                    self._adaptive_quality = min(0.5, self._adaptive_quality + 0.01)
                elif avg_time > 0.1:  # Slower system
                    self._adaptive_quality = max(0.25, self._adaptive_quality - 0.01)
            
            mat = fitz.Matrix(self._adaptive_quality, self._adaptive_quality)
            
            # Fast rendering with optimal settings
            pix = page.get_pixmap(matrix=mat, alpha=False, annots=False)
            
            if pix:
                # LAYER 1 OPTIMIZATION 4: Use WebP for best compression/speed balance
                try:
                    img_data = pix.tobytes("webp", webp_quality=75)
                    qimage = QImage.fromData(img_data, "WEBP")
                except:
                    # Fallback to JPEG if WebP not available
                    img_data = pix.tobytes("jpeg", jpg_quality=75)
                    qimage = QImage.fromData(img_data, "JPEG")
                
                if not qimage.isNull():
                    # LAYER 1 OPTIMIZATION 5: Save to cache for next time
                    self._save_to_cache(page_num, qimage)
                    
                    # Track performance for adaptive quality
                    render_time = time.time() - start_time
                    self._render_times.append(render_time)
                    if len(self._render_times) > 20:
                        self._render_times.pop(0)  # Keep only recent times
                    
                    self.thumbnailReady.emit(page_num, qimage)
                    return True
            
            return False
            
        except Exception as e:
            print(f"Single thumbnail error ({page_num}): {e}")
            return False


class PDFPageRenderer(QThread):
    """Background page renderer that converts PDF pages to QImage and emits pageRendered."""
    pageRendered = pyqtSignal(int, QImage, float)  # Added quality parameter

    def __init__(self, pdf_path: str, quality: float = 4.0, parent=None):
        super().__init__(parent)
        self.pdf_path = pdf_path
        self.quality = quality
        self._running = True
        self._paused = False  # Flag for pausing during panning
        self._queue = []
        self._priority_queue = []  # High priority queue for grid pages
        self._lock = threading.Lock()

    # Delay opening the document until the thread runs to avoid blocking the UI
        # (initialized in __init__ above)

    def add_page_to_queue(self, page_num: int, priority=False, quality=None):
        # Skip queueing entirely during fast operations to prevent lag
        viewer = self.parent() if hasattr(self, 'parent') else None
        if viewer and hasattr(viewer, 'pdf_widget'):
            # Check system performance and throttle accordingly
            performance_level = getattr(viewer.pdf_widget, '_performance_monitor', {}).get('performance_level', 'high')
            
            # When performance is low, be much more aggressive about blocking renders
            if performance_level == 'low':
                # Low performance: Only allow priority renders for current page
                if not priority:
                    return  # Block ALL non-priority renders when performance is poor
                # Reduce quality for faster rendering when performance is low
                if quality and quality > 2.0:
                    quality = min(quality, 2.0)  # Cap quality to reduce rendering time
            
            # Allow priority renders but block regular renders during interactions
            if (getattr(viewer.pdf_widget, '_fast_zoom_mode', False) or 
                getattr(viewer.pdf_widget, 'is_panning', False) or
                getattr(viewer.pdf_widget, '_interaction_mode', False)):
                # During interactions, only allow priority rendering of current page with readable quality
                if not priority:
                    return  # Block all non-priority renders
                # For priority renders during interaction, ensure minimum readable quality
                if quality and quality < 2.8:
                    quality = 2.8  # Ensure readable quality even during interactions
            
            # 🚀 LAYER 3: Apply adaptive quality based on performance
            if quality and hasattr(viewer.pdf_widget, '_get_adaptive_quality'):
                adaptive_factor = viewer.pdf_widget._get_adaptive_quality() / viewer.pdf_widget.base_quality
                quality = quality * adaptive_factor
                quality = max(1.0, min(10.0, quality))  # Keep within bounds
        
        with self._lock:
            # AGGRESSIVE QUEUING for modern systems - but adjust based on performance
            viewer = self.parent() if hasattr(self, 'parent') else None
            performance_level = 'high'
            if viewer and hasattr(viewer, 'pdf_widget'):
                performance_level = getattr(viewer.pdf_widget, '_performance_monitor', {}).get('performance_level', 'high')
            
            if performance_level == 'low':
                # Reduce queue sizes when performance is poor
                max_priority_queue = 5   # Smaller priority queue when struggling
                max_regular_queue = 10   # Much smaller background queue
            elif performance_level == 'medium':
                max_priority_queue = 10  # Medium queues
                max_regular_queue = 25
            else:
                # High performance - large queues for instant quality
                max_priority_queue = 20  # Large priority queue for instant quality
                max_regular_queue = 50   # Massive background queue for preloading
            
            # Check if page is already in either queue
            priority_pages = [p[0] if isinstance(p, tuple) else p for p in self._priority_queue]
            regular_pages = [p[0] if isinstance(p, tuple) else p for p in self._queue]
            
            if page_num in priority_pages or page_num in regular_pages:
                return  # Already queued
                
            # Check queue limits
            if priority and len(self._priority_queue) >= max_priority_queue:
                return  # Priority queue full - skip to avoid lag
            elif not priority and len(self._queue) >= max_regular_queue:
                return  # Regular queue full - skip to avoid lag
                
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
                # CRITICAL: Skip all processing when paused (during panning)
                with self._lock:
                    if self._paused:
                        time.sleep(0.016)  # Sleep while paused
                        continue
                
                # Allow limited processing during interactions but with reduced priority
                viewer = self.parent() if hasattr(self, 'parent') else None
                if viewer and hasattr(viewer, 'pdf_widget'):
                    if (getattr(viewer.pdf_widget, '_fast_zoom_mode', False) or 
                        getattr(viewer.pdf_widget, 'is_panning', False) or
                        getattr(viewer.pdf_widget, '_interaction_mode', False)):
                        time.sleep(0.03)  # Slightly longer sleep during interactions
                        continue
                
                page_info = None
                with self._lock:
                    # Process priority queue first (grid pages)
                    if self._priority_queue:
                        page_info = self._priority_queue.pop(0)
                    elif self._queue:
                        page_info = self._queue.pop(0)

                if page_info is None:
                    # Sleep longer for better performance when no work to do
                    time.sleep(0.016)  # ~60 FPS equivalent for better CPU usage
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
                    
                    # Enhanced high-quality rendering with better antialiasing for text sharpness
                    pix = page.get_pixmap(matrix=mat, alpha=False, annots=True)
                        
                    # Use PNG bytes which QImage understands reliably
                    # For all quality levels, use the reliable PNG method
                    img_data = pix.tobytes("png")
                    qimage = QImage.fromData(img_data, "PNG")
                    
                    if qimage and not qimage.isNull():
                        self.pageRendered.emit(page_num, qimage, render_quality)

                except Exception as e:
                    # Continue processing remaining pages on error
                    print(f"Renderer error rendering page {page_num}: {e}")

        finally:
            try:
                if hasattr(self, 'pdf_doc') and self.pdf_doc is not None:
                    self.pdf_doc.close()
                    self.pdf_doc = None
            except Exception:
                pass

    def is_busy(self):
        """Check if renderer is currently busy processing textures"""
        with self._lock:
            # Consider busy if there are items in either queue or if currently processing
            return (len(self._priority_queue) > 0 or 
                   len(self._queue) > 0 or 
                   self.isRunning())
    
    def pause(self):
        """COMPLETE PAUSE: Stop all background rendering and processing during panning"""
        with self._lock:
            self._paused = True
            # Clear all pending work to stop CPU activity immediately
            self.queue.queue.clear()
            self.low_priority_queue.queue.clear()
    
    def resume(self):
        """Resume background rendering after panning"""
        with self._lock:
            self._paused = False


class GPUPDFWidget(QOpenGLWidget):
    """🚀 LAYER 3 ENHANCED: OpenGL widget with intelligent GPU performance management"""
    
    def __init__(self):
        super().__init__()
        
        # Set widget attributes for transparency
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        
        self.texture_cache = GPUTextureCache()
        self.current_page = 0
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.page_texture = None
        self.page_width = 1.0
        self.page_height = 1.0
        self._pending_images = []  # Queue for pending texture creation
        self.grid_mode = False  # Track if we're currently in grid view mode
        
        # Enable context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    # Remove Qt palette/stylesheet background for OpenGL widget to avoid blending issues

        # Zoom-adaptive quality settings - optimized for sharp text with smooth filtering
        self.base_quality = 4.5  # Higher base quality for very sharp text at 100% zoom
        self.quality_zoom_threshold = 1.2  # Even lower threshold for earlier high quality
        self.max_quality = 7.0  # Higher maximum quality for ultra-sharp text
        self.last_zoom_quality = self.base_quality

        # Grid view attributes
        self.grid_mode = False
        self.grid_cols = 1
        self.grid_rows = 1
        self.grid_layout_mode = 'adaptive'  # 'adaptive', 'uniform', 'proportional'
        self._grid_layout_cache = []
        self._grid_cached_size = (0, 0, 0, 0)  # (width, height, cols, rows)
        self.grid_textures = {}
        
        # Grid layout configuration
        self.grid_gap_ratio = 0.01  # Gap as percentage of widget size (1% instead of 2%)
        self.grid_min_gap = 5.0     # Minimum gap in pixels (reduced from 10)
        self.grid_max_gap = 15.0    # Maximum gap in pixels (reduced from 30)
        self.grid_aspect_weight = 0.7  # How much to weight aspect ratios (0.0-1.0)
        
        # Temporary zoom state for grid view
        self.is_temp_zoomed = False
        self.temp_zoom_factor = 1.0
        self.temp_pan_x = 0.0
        self.temp_pan_y = 0.0
        
        # Performance optimization
        self._last_update_time = 0
        self._min_update_interval = 16  # ~60 FPS limit (16ms between updates)
        
        # Fast zoom optimization variables
        self._fast_zoom_mode = False
        self._fast_zoom_start_time = 0
        self._fast_zoom_timeout = 300  # Shorter timeout for more responsive quality recovery
        self._interaction_mode = False  # Track any user interaction
        self._active_panning = False    # Track active panning for optimizations
        
        # 🚀 LAYER 3: Advanced performance monitoring and resource management
        self._performance_monitor = {
            'frame_times': [],
            'texture_loads': 0,
            'cache_evictions': 0,
            'memory_warnings': 0,
            'last_stats_time': 0,
            'avg_frame_time': 16.67,  # Target 60 FPS
            'performance_level': 'high'  # high, medium, low
        }
        
        # Adaptive quality based on performance
        self._adaptive_quality_enabled = True
        self._performance_quality_multiplier = 1.0
        
        # Rendering and threading
        self.last_mouse_pos = QPointF()
        self.is_panning = False
        
        # Animation timing for loading state
        self.animation_time = 0.0
        
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    def initializeGL(self):
        """Initialize OpenGL context - minimal setup for fast startup"""
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # Standard alpha blending
        glEnable(GL_MULTISAMPLE)  # Enable multisampling if available
        
        # Match container background color exactly: rgb(38, 38, 38)
        glClearColor(38/255.0, 38/255.0, 38/255.0, 1.0)
    
    def update(self, *args, **kwargs):
        """Frame rate limited update method for better performance"""
        import time
        current_time = time.time() * 1000  # Convert to milliseconds
        
        # Dynamic frame rate limiting based on zoom level for close zoom optimization
        effective_zoom = self.zoom_factor
        if hasattr(self, 'is_temp_zoomed') and self.is_temp_zoomed:
            effective_zoom *= getattr(self, 'temp_zoom_factor', 1.0)
        
        # Increase minimum interval for high zoom to reduce lag
        if effective_zoom > 5.0:
            min_interval = 32  # ~30 FPS for very high zoom
        elif effective_zoom > 3.0:
            min_interval = 24  # ~40 FPS for high zoom
        else:
            min_interval = self._min_update_interval  # ~60 FPS for normal zoom
        
        # Only update if enough time has passed since last update
        if current_time - self._last_update_time >= min_interval:
            self._last_update_time = current_time
            super().update(*args, **kwargs)
    
    def force_update(self):
        """Force immediate update bypassing frame rate limiting"""
        super().update()
    
    def paintGL(self):
        try:
            # GPU-ONLY PANNING: Keep existing textures visible, stop CPU processing only
            if getattr(self, '_skip_heavy_processing', False) or getattr(self, '_ultra_fast_mode', False):
                # CLEAR BACKGROUND
                glClearColor(38/255.0, 38/255.0, 38/255.0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT)
                
                if self.grid_mode:
                    # CRITICAL: Render ALL existing textures - don't hide pages during panning
                    self.render_grid_view_existing_only()  # Show all cached textures
                else:
                    self.render_single_page_fast()  # Fast single page rendering
                
                return  # Skip CPU processing but keep all visuals
            
            # Normal rendering path - full functionality when not panning
            import time
            frame_start_time = time.time()
            
            glClearColor(38/255.0, 38/255.0, 38/255.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            
            current_time = time.time()
            self.animation_time = current_time
            
            # Performance monitoring
            self._update_performance_stats(frame_start_time)
            
            # Detect interactions
            renderer_busy = (hasattr(self, 'renderer') and self.renderer and 
                           hasattr(self.renderer, 'is_busy') and self.renderer.is_busy())
            
            is_actively_interacting = (getattr(self, '_fast_zoom_mode', False) or 
                                     getattr(self, '_active_panning', False) or
                                     renderer_busy)
            
            performance_quality = self._get_adaptive_quality()
            
            # STOP background processing during panning but keep visuals
            if getattr(self, '_active_panning', False):
                max_items = 0  # Zero background processing during panning
            elif self.grid_mode and is_actively_interacting:
                max_items = 0  # Zero processing during grid interactions
            elif is_actively_interacting:
                max_items = 0  # Zero processing during interactions
            else:
                # 🚀 LAYER 2 OPTIMIZATION: Adaptive processing based on performance
                effective_zoom = self.zoom_factor
                if hasattr(self, 'is_temp_zoomed') and self.is_temp_zoomed:
                    effective_zoom *= getattr(self, 'temp_zoom_factor', 1.0)
                
                # Performance-based adaptive processing
                if not hasattr(self, '_last_render_time'):
                    self._last_render_time = current_time
                    self._frame_times = []
                
                # Track frame performance
                frame_time = current_time - self._last_render_time
                self._frame_times.append(frame_time)
                if len(self._frame_times) > 10:
                    self._frame_times.pop(0)
                
                avg_frame_time = sum(self._frame_times) / len(self._frame_times) if self._frame_times else 0.016
                
                # Adaptive processing based on performance
                if avg_frame_time > 0.025:  # System struggling (< 40 FPS)
                    texture_processing_factor = 1.0  # Keep normal processing even when struggling
                elif avg_frame_time < 0.012:  # System performing well (> 80 FPS)
                    texture_processing_factor = 3.0  # MASSIVE increase for fast systems (was 1.5)
                else:
                    texture_processing_factor = 2.0  # Higher normal processing (was 1.0)
                
                if self.grid_mode:
                    max_items = max(2, int(4 * texture_processing_factor))  # Much higher for grid (was 1)
                elif effective_zoom > 4.0:
                    max_items = max(3, int(6 * texture_processing_factor))  # Higher for zoomed (was 2)
                elif effective_zoom > 2.0:
                    max_items = max(4, int(8 * texture_processing_factor))  # Higher for moderate zoom (was 2)
                else:
                    max_items = max(5, int(10 * texture_processing_factor))  # Much higher for normal (was 3)
                
                self._last_render_time = current_time
        
            # 🚀 LAYER 2 OPTIMIZATION: Optimized texture processing with error handling
            try:
                if max_items > 0:
                    # Use high-precision timing for texture processing
                    texture_start = time.perf_counter()
                    self.process_pending_images(max_items=max_items)
                    texture_time = time.perf_counter() - texture_start
                    
                    # Track texture processing performance
                    if not hasattr(self, '_texture_processing_times'):
                        self._texture_processing_times = []
                    
                    self._texture_processing_times.append(texture_time)
                    if len(self._texture_processing_times) > 20:
                        self._texture_processing_times.pop(0)
            except Exception as e:
                print(f"Error processing pending images: {e}")
            
            # 🚀 LAYER 2 OPTIMIZATION: Optimized rendering path selection
            try:
                if self.grid_mode:
                    # PRIORITY: Grid rendering gets immediate execution
                    # Skip any blocking operations during grid interactions
                    try:
                        render_start = time.perf_counter()
                        self.render_grid_view(temp=self.is_temp_zoomed)
                        render_time = time.perf_counter() - render_start
                        
                        # Track grid rendering performance
                        if not hasattr(self, '_grid_render_times'):
                            self._grid_render_times = []
                        self._grid_render_times.append(render_time)
                        if len(self._grid_render_times) > 10:
                            self._grid_render_times.pop(0)
                            
                    except TypeError:
                        # Fallback for older signature (defensive)
                        self.render_grid_view()
                    except Exception as e:
                        print(f"Grid rendering error: {e}")
                else:
                    render_start = time.perf_counter()
                    self.render_single_page()
                    render_time = time.perf_counter() - render_start
                    
                    # Track single page rendering performance
                    if not hasattr(self, '_single_render_times'):
                        self._single_render_times = []
                    self._single_render_times.append(render_time)
                    if len(self._single_render_times) > 10:
                        self._single_render_times.pop(0)
                        
            except Exception as e:
                print(f"OpenGL rendering error: {e}")
                # Prevent crash by catching any OpenGL errors
                
        except Exception as e:
            print(f"Main paintGL error: {e}")
            # Prevent crash by catching any OpenGL errors
    
    def render_single_page(self):
        """Render a single page with loading state support"""
        try:
            # Clear grid mode flag for single page rendering to prioritize thumbnails
            self.grid_mode = False
            
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
            
            if widget_width <= 0 or widget_height <= 0 or self.page_width <= 0 or self.page_height <= 0:
                return  # Skip if dimensions are invalid
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
            
            # Bind texture and draw quad
            # Bind texture if not already bound
            if not hasattr(self, '_last_bound_texture') or self._last_bound_texture is not self.page_texture:
                try:
                    self.page_texture.bind()
                    self._last_bound_texture = self.page_texture
                except Exception:
                    pass
            
            glBegin(GL_QUADS)
            # Texture coordinates for PPM image (180° rotation fix)
            glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, -1.0)
            glTexCoord2f(1.0, 1.0); glVertex2f(1.0, -1.0)
            glTexCoord2f(1.0, 0.0); glVertex2f(1.0, 1.0)
            glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, 1.0)
            glEnd()
            
            # Do not release immediately; keep bound to avoid GPU stalls. Release only when changed.
            # Keep reference to bound texture in _last_bound_texture
        
        except Exception as e:
            print(f"OpenGL rendering error in render_single_page: {e}")
            # Clear any OpenGL errors to prevent crashes
            import OpenGL.GL as gl
            while gl.glGetError() != gl.GL_NO_ERROR:
                pass
    
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
    
    def render_single_page_fast(self):
        """Optimized single page rendering for smooth panning and zooming."""
        glLoadIdentity()
        
        # Apply transformations
        glTranslatef(self.pan_x, self.pan_y, 0.0)
        glScalef(self.zoom_factor, self.zoom_factor, 1.0)
        
        # Render the page texture if it exists
        if hasattr(self, 'page_texture') and self.page_texture and self.page_texture.isCreated():
            # Use the reliable aspect ratio calculation
            if hasattr(self, 'page_width') and hasattr(self, 'page_height') and self.page_width > 0 and self.page_height > 0:
                widget_aspect = self.width() / self.height()
                page_aspect = self.page_width / self.page_height
                if widget_aspect > page_aspect:
                    glScalef(page_aspect / widget_aspect, 1.0, 1.0)
                else:
                    glScalef(1.0, widget_aspect / page_aspect, 1.0)

            # Bind and draw
            glColor3f(1.0, 1.0, 1.0)
            self.page_texture.bind()
            glBegin(GL_QUADS)
            glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, -1.0)
            glTexCoord2f(1.0, 1.0); glVertex2f(1.0, -1.0)
            glTexCoord2f(1.0, 0.0); glVertex2f(1.0, 1.0)
            glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, 1.0)
            glEnd()
            self.page_texture.release()
        else:
            # If no texture, render the loading state
            self.render_loading_state()
    
    def render_grid_view_existing_only(self):
        """GPU-ONLY: Render ALL existing textures during panning - no hiding, no loading"""
        
        # Standard OpenGL setup with transformations
        glLoadIdentity()
        if getattr(self, 'is_temp_zoomed', False):
            glTranslatef(self.temp_pan_x, self.temp_pan_y, 0.0)
            glScalef(self.temp_zoom_factor, self.temp_zoom_factor, 1.0)
        else:
            glTranslatef(self.pan_x, self.pan_y, 0.0)
            glScalef(self.zoom_factor, self.zoom_factor, 1.0)
        
        glColor3f(1.0, 1.0, 1.0)
        
        # Ensure we have grid layout
        if not hasattr(self, '_grid_cells') or not self._grid_cells:
            return
            
        viewer = self.window()
        start_page = getattr(viewer, 'current_page', self.current_page)
        total_pages = getattr(viewer, 'total_pages', getattr(self, 'total_pages', 0))
        
        w_px = self.width()
        h_px = self.height()
        
        # CRITICAL: Render ALL grid pages that have existing textures
        # Do NOT limit the number - show everything that's available
        pages_in_grid = min(len(self._grid_cells), self.grid_cols * self.grid_rows)
        
        for idx in range(pages_in_grid):
            page_num = start_page + idx
            if page_num >= total_pages:
                break
            
            # Check if texture exists - if yes, render it, if no, show minimal placeholder
            texture = self.texture_cache.get_texture(page_num)
            if texture and hasattr(texture, 'isCreated') and texture.isCreated():
                # Render existing texture with proper aspect ratio
                self.render_grid_cell_with_existing_texture(idx, page_num, texture, w_px, h_px)
            else:
                # Show a very minimal placeholder to maintain grid structure
                self._draw_minimal_grid_placeholder(idx, w_px, h_px)
    
    def render_grid_cell_with_existing_texture(self, idx, page_num, texture, w_px, h_px):
        """Render grid cell with existing texture - optimized for panning"""
        if idx >= len(self._grid_cells):
            return
            
        cell = self._grid_cells[idx]
        
        # Use cached dimensions if available, otherwise use texture dimensions
        page_dims = self.texture_cache.get_dimensions(page_num)
        if page_dims and page_dims[1] > 0:
            page_aspect = page_dims[0] / page_dims[1]
        else:
            # Use texture dimensions as fallback
            page_aspect = 0.75  # Standard fallback
        
        # Fast aspect fitting
        inner_w_px = cell['inner_w_px']
        inner_h_px = cell['inner_h_px']
        
        if page_aspect > (inner_w_px / inner_h_px):
            fit_w = inner_w_px
            fit_h = fit_w / page_aspect
        else:
            fit_h = inner_h_px
            fit_w = fit_h * page_aspect
        
        # Center within cell
        fitted_x_px = cell['page_x_px'] + (inner_w_px - fit_w) * 0.5
        fitted_y_px = cell['page_y_px'] + (inner_h_px - fit_h) * 0.5
        
        # Convert to OpenGL coordinates
        x_scale = 2.0 / w_px
        y_scale = 2.0 / h_px
        
        quad_x = fitted_x_px * x_scale - 1.0
        quad_y = 1.0 - fitted_y_px * y_scale
        quad_width = fit_w * x_scale
        quad_height = fit_h * y_scale
        
        # Render texture using cached binding to avoid GPU pipeline stalls
        try:
            if not hasattr(self, '_last_bound_texture') or self._last_bound_texture is not texture:
                texture.bind()
                self._last_bound_texture = texture
        except Exception:
            pass
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex2f(quad_x, quad_y - quad_height)
        glTexCoord2f(1.0, 1.0); glVertex2f(quad_x + quad_width, quad_y - quad_height)
        glTexCoord2f(1.0, 0.0); glVertex2f(quad_x + quad_width, quad_y)
        glTexCoord2f(0.0, 0.0); glVertex2f(quad_x, quad_y)
        glEnd()

    def _draw_minimal_grid_placeholder(self, idx, w_px, h_px):
        """Draw an extremely minimal placeholder to maintain grid structure during panning"""
        if not hasattr(self, '_grid_cells') or idx >= len(self._grid_cells):
            return
            
        cell = self._grid_cells[idx]
        inner_w_px = cell['inner_w_px']
        inner_h_px = cell['inner_h_px']
        
        # Use a smaller rectangle than full cell to show grid structure but not distract
        margin = min(inner_w_px, inner_h_px) * 0.05  # 5% margin
        fitted_x_px = cell['page_x_px'] + margin
        fitted_y_px = cell['page_y_px'] + margin
        fit_w = inner_w_px - 2 * margin
        fit_h = inner_h_px - 2 * margin
        
        x_scale = 2.0 / w_px
        y_scale = 2.0 / h_px
        
        quad_x = fitted_x_px * x_scale - 1.0
        quad_y = 1.0 - fitted_y_px * y_scale
        quad_width = fit_w * x_scale
        quad_height = fit_h * y_scale
        
        # Draw a very subtle outline only - no fill
        glDisable(GL_TEXTURE_2D)  # Disable texturing for line drawing
        glColor4f(0.4, 0.4, 0.4, 0.2)  # Very subtle gray, very transparent
        glBegin(GL_LINE_LOOP)  # Draw outline only
        glVertex2f(quad_x, quad_y - quad_height)
        glVertex2f(quad_x + quad_width, quad_y - quad_height)
        glVertex2f(quad_x + quad_width, quad_y)
        glVertex2f(quad_x, quad_y)
        glEnd()
        glEnable(GL_TEXTURE_2D)  # Re-enable texturing
        glColor3f(1.0, 1.0, 1.0)  # Reset color

    def render_grid_ultra_fast(self):
        """EXTREME MINIMAL: Render only 1-2 textures during panning for maximum speed"""
        
        # Emergency mode: If performance is extremely low, render even less
        performance_level = getattr(self, '_performance_monitor', {}).get('performance_level', 'high')
        if performance_level == 'low':
            # Emergency mode: Only render 1 texture and skip all validation
            ultra_minimal_count = 1
            skip_validation = True
        else:
            # Normal/medium performance: render based on grid size
            grid_size = len(self._grid_cells) if hasattr(self, '_grid_cells') else 4
            if grid_size <= 4:  # 2x2 grid
                ultra_minimal_count = 2  # Show 2 textures for 2x2
            else:
                ultra_minimal_count = 1  # Only 1 texture for larger grids
            skip_validation = False
        
        # Instant OpenGL setup
        glLoadIdentity()
        if getattr(self, 'is_temp_zoomed', False):
            glTranslatef(self.temp_pan_x, self.temp_pan_y, 0.0)
            glScalef(self.temp_zoom_factor, self.temp_zoom_factor, 1.0)
        else:
            glTranslatef(self.pan_x, self.pan_y, 0.0)
            glScalef(self.zoom_factor, self.zoom_factor, 1.0)
        
        glColor3f(1.0, 1.0, 1.0)
        
        # EXTREME: Only render the first 1-2 pages during panning - even for 5x2
        if not hasattr(self, '_grid_cells') or len(self._grid_cells) == 0:
            return
        
        viewer = self.window()
        start_page = getattr(viewer, 'current_page', self.current_page)
        total_pages = getattr(viewer, 'total_pages', getattr(self, 'total_pages', 0))
        
        w_px = self.width()
        h_px = self.height()
        
        # Use ultra-minimal count based on performance
        ultra_minimal_count = min(ultra_minimal_count, len(self._grid_cells))
        
        # Pre-calculate for speed
        x_scale = 2.0 / w_px
        y_scale = 2.0 / h_px
        
        # Render only the first 1-2 textures
        for idx in range(ultra_minimal_count):
            page_num = start_page + idx
            if page_num >= total_pages:
                break
            
            # Get texture with robust validation
            texture = self.texture_cache.get_texture(page_num)
            if skip_validation:
                # Emergency mode: Skip texture validation but ensure texture exists
                if not texture:
                    # Instead of continuing, draw a simple colored rectangle as placeholder
                    self._draw_page_placeholder(idx, w_px, h_px, x_scale, y_scale)
                    continue
                # Quick check if texture has any content
                try:
                    if not texture.textureId():
                        self._draw_page_placeholder(idx, w_px, h_px, x_scale, y_scale)
                        continue
                except:
                    self._draw_page_placeholder(idx, w_px, h_px, x_scale, y_scale)
                    continue
            else:
                # Normal validation
                if not texture or not hasattr(texture, 'isCreated') or not texture.isCreated():
                    self._draw_page_placeholder(idx, w_px, h_px, x_scale, y_scale)
                    continue
                # Additional check for valid texture ID
                try:
                    if not texture.textureId():
                        self._draw_page_placeholder(idx, w_px, h_px, x_scale, y_scale)
                        continue
                except:
                    self._draw_page_placeholder(idx, w_px, h_px, x_scale, y_scale)
                    continue
                
            # Ultra-fast rendering with fixed aspect
            cell = self._grid_cells[idx]
            inner_w_px = cell['inner_w_px']
            inner_h_px = cell['inner_h_px']
            
            # Fixed aspect for speed
            page_aspect = 0.75
            
            if page_aspect > (inner_w_px / inner_h_px):
                fit_w = inner_w_px
                fit_h = fit_w / page_aspect
            else:
                fit_h = inner_h_px
                fit_w = fit_h * page_aspect
            
            fitted_x_px = cell['page_x_px'] + (inner_w_px - fit_w) * 0.5
            fitted_y_px = cell['page_y_px'] + (inner_h_px - fit_h) * 0.5
            
            quad_x = fitted_x_px * x_scale - 1.0
            quad_y = 1.0 - fitted_y_px * y_scale
            quad_width = fit_w * x_scale
            quad_height = fit_h * y_scale
            
            # Direct render with cached binding - more robust texture handling
            try:
                # Ensure we have a valid texture before binding
                texture_id = texture.textureId()
                if texture_id == 0:
                    continue
                    
                if not hasattr(self, '_last_bound_texture') or self._last_bound_texture is not texture:
                    texture.bind()
                    self._last_bound_texture = texture
                    
                # Draw with semi-transparent white background during panning for better visibility
                if self._active_panning:
                    glColor4f(1.0, 1.0, 1.0, 0.9)  # Slightly transparent during panning
                else:
                    glColor3f(1.0, 1.0, 1.0)  # Fully opaque when not panning
                    
            except Exception as e:
                # If texture binding fails, skip this texture
                continue
                
            glBegin(GL_QUADS)
            glTexCoord2f(0.0, 1.0); glVertex2f(quad_x, quad_y - quad_height)
            glTexCoord2f(1.0, 1.0); glVertex2f(quad_x + quad_width, quad_y - quad_height)
            glTexCoord2f(1.0, 0.0); glVertex2f(quad_x + quad_width, quad_y)
            glTexCoord2f(0.0, 0.0); glVertex2f(quad_x, quad_y)
            glEnd()

    def _draw_page_placeholder(self, idx, w_px, h_px, x_scale, y_scale):
        """Draw a subtle placeholder when texture is not available during fast panning"""
        if not hasattr(self, '_grid_cells') or idx >= len(self._grid_cells):
            return
            
        cell = self._grid_cells[idx]
        inner_w_px = cell['inner_w_px']
        inner_h_px = cell['inner_h_px']
        
        # Fixed aspect for speed
        page_aspect = 0.75
        
        if page_aspect > (inner_w_px / inner_h_px):
            fit_w = inner_w_px
            fit_h = fit_w / page_aspect
        else:
            fit_h = inner_h_px
            fit_w = fit_h * page_aspect
        
        fitted_x_px = cell['page_x_px'] + (inner_w_px - fit_w) * 0.5
        fitted_y_px = cell['page_y_px'] + (inner_h_px - fit_h) * 0.5
        
        quad_x = fitted_x_px * x_scale - 1.0
        quad_y = 1.0 - fitted_y_px * y_scale
        quad_width = fit_w * x_scale
        quad_height = fit_h * y_scale
        
        # Draw a subtle dark gray placeholder (not white) that's barely visible
        glDisable(GL_TEXTURE_2D)  # Disable texturing for solid color
        glColor4f(0.2, 0.2, 0.2, 0.3)  # Dark gray, semi-transparent
        glBegin(GL_QUADS)
        glVertex2f(quad_x, quad_y - quad_height)
        glVertex2f(quad_x + quad_width, quad_y - quad_height)
        glVertex2f(quad_x + quad_width, quad_y)
        glVertex2f(quad_x, quad_y)
        glEnd()
        glEnable(GL_TEXTURE_2D)  # Re-enable texturing
        glColor3f(1.0, 1.0, 1.0)  # Reset color

    def render_grid_view_fast(self):
        """FAST grid rendering - reduced complexity from ultra-fast mode"""
        glLoadIdentity()
        
        # Apply transformations
        if getattr(self, 'is_temp_zoomed', False):
            glTranslatef(self.temp_pan_x, self.temp_pan_y, 0.0)
            glScalef(self.temp_zoom_factor, self.temp_zoom_factor, 1.0)
        else:
            glTranslatef(self.pan_x, self.pan_y, 0.0)
            glScalef(self.zoom_factor, self.zoom_factor, 1.0)
        
        glColor3f(1.0, 1.0, 1.0)
        
        if not hasattr(self, '_grid_cells') or not self._grid_cells:
            return
            
        viewer = self.window()
        start_page = getattr(viewer, 'current_page', self.current_page)
        total_pages = getattr(viewer, 'total_pages', getattr(self, 'total_pages', 0))
        
        w_px = self.width()
        h_px = self.height()
        
        # Render more pages than ultra-fast but still limited for performance
        pages_to_render = min(len(self._grid_cells), self.grid_cols * self.grid_rows, 8)
        
        for idx in range(pages_to_render):
            page_num = start_page + idx
            if page_num >= total_pages:
                break
            
            texture = self.texture_cache.get_texture(page_num)
            if texture and hasattr(texture, 'isCreated') and texture.isCreated():
                try:
                    cell = self._grid_cells[idx]
                    
                    # Simplified aspect ratio for speed
                    page_aspect = 0.75
                    
                    inner_w_px = cell['inner_w_px']
                    inner_h_px = cell['inner_h_px']
                    
                    if page_aspect > (inner_w_px / inner_h_px):
                        fit_w = inner_w_px
                        fit_h = fit_w / page_aspect
                    else:
                        fit_h = inner_h_px
                        fit_w = fit_h * page_aspect
                    
                    fitted_x_px = cell['page_x_px'] + (inner_w_px - fit_w) * 0.5
                    fitted_y_px = cell['page_y_px'] + (inner_h_px - fit_h) * 0.5
                    
                    x_scale = 2.0 / w_px
                    y_scale = 2.0 / h_px
                    
                    quad_x = fitted_x_px * x_scale - 1.0
                    quad_y = 1.0 - fitted_y_px * y_scale
                    quad_width = fit_w * x_scale
                    quad_height = fit_h * y_scale
                    
                    try:
                        if not hasattr(self, '_last_bound_texture') or self._last_bound_texture is not texture:
                            texture.bind()
                            self._last_bound_texture = texture
                    except Exception:
                        pass
                    glBegin(GL_QUADS)
                    glTexCoord2f(0.0, 1.0); glVertex2f(quad_x, quad_y - quad_height)
                    glTexCoord2f(1.0, 1.0); glVertex2f(quad_x + quad_width, quad_y - quad_height)
                    glTexCoord2f(1.0, 0.0); glVertex2f(quad_x + quad_width, quad_y)
                    glTexCoord2f(0.0, 0.0); glVertex2f(quad_x, quad_y)
                    glEnd()
                except:
                    continue
    
    def render_grid_view(self, temp=False):
        """Render multiple pages in a grid layout - COMPLETELY NON-BLOCKING

        Args:
            temp: if True, apply temporary zoom/pan (used when a thumbnail is focused)
        """
        # Set grid mode flag to optimize resource allocation
        self.grid_mode = True
        
        glLoadIdentity()

        # IMMEDIATE TRANSFORM APPLICATION - Never wait for textures
        if temp and getattr(self, 'is_temp_zoomed', False):
            glTranslatef(self.temp_pan_x, self.temp_pan_y, 0.0)
            glScalef(self.temp_zoom_factor, self.temp_zoom_factor, 1.0)
        else:
            # Apply zoom and pan transformations for grid view - IMMEDIATELY
            glTranslatef(self.pan_x, self.pan_y, 0.0)
            glScalef(self.zoom_factor, self.zoom_factor, 1.0)
        
        # Set common OpenGL state once for all textures (optimization for panning)
        glColor3f(1.0, 1.0, 1.0)  # White color for texture rendering
        
        # Prefer the viewer's current_page/total_pages when available
        viewer = self.window()
        start_page = getattr(viewer, 'current_page', self.current_page)
        total_pages = getattr(viewer, 'total_pages', getattr(self, 'total_pages', 0))

        pages_needed = self.grid_cols * self.grid_rows

        # 🚀 NON-BLOCKING: Recompute grid layout only when widget size or grid dims change
        w_px = max(1, self.width())
        h_px = max(1, self.height())
        if self._grid_cached_size != (w_px, h_px, self.grid_cols, self.grid_rows):
            # Use async layout computation to avoid blocking
            if not hasattr(self, '_layout_computing'):
                self._layout_computing = True
                QTimer.singleShot(0, lambda: (
                    self.compute_grid_layout(),
                    setattr(self, '_layout_computing', False)
                ))

        # COMPLETELY ASYNC RENDERING: Render what we have immediately, queue what we don't
        # Never block for any texture operations
        
        # Iterate over every grid slot using precomputed cell rectangles
        for idx in range(pages_needed):
            page_num = start_page + idx
            # If page index is beyond document end, skip this cell
            if page_num >= total_pages:
                continue
            
            # Check if we have a texture immediately - no blocking operations
            texture = None
            try:
                texture = self.texture_cache.get_texture(page_num)
                is_texture_valid = texture and hasattr(texture, 'isCreated') and texture.isCreated()
            except:
                is_texture_valid = False
            
            if is_texture_valid:
                # RENDER TEXTURE IMMEDIATELY if available
                self.render_grid_cell_with_texture(idx, page_num, texture, w_px, h_px)
            else:
                # RENDER DARK PLACEHOLDER IMMEDIATELY - no white flashing
                self.render_grid_cell_placeholder(idx, page_num, w_px, h_px)
                
                # Queue this page for background loading - but don't wait
                self.queue_page_for_background_loading(page_num, idx < 4)

    def render_grid_cell_with_texture(self, idx, page_num, texture, w_px, h_px):
        """Render a single grid cell with an available texture - GPU optimized"""
        if idx >= len(self._grid_cells):
            return
            
        cell = self._grid_cells[idx]
        
        # Use cached aspect ratio if available for performance
        page_dims = self.texture_cache.get_dimensions(page_num)
        if page_dims and page_dims[1] > 0:
            page_aspect = page_dims[0] / page_dims[1]
        else:
            page_aspect = 0.75  # Default aspect - don't waste time calculating
        
        # Fast aspect fitting calculation
        inner_w_px = cell['inner_w_px']
        inner_h_px = cell['inner_h_px']
        
        if page_aspect > (inner_w_px / inner_h_px):
            # Fit to width
            fit_w = inner_w_px
            fit_h = fit_w / page_aspect
        else:
            # Fit to height  
            fit_h = inner_h_px
            fit_w = fit_h * page_aspect
        
        # Center within cell - simplified calculation
        fitted_x_px = cell['page_x_px'] + (inner_w_px - fit_w) * 0.5
        fitted_y_px = cell['page_y_px'] + (inner_h_px - fit_h) * 0.5
        
        # GPU coordinate conversion - optimized
        x_scale = 2.0 / w_px
        y_scale = 2.0 / h_px
        
        quad_x = fitted_x_px * x_scale - 1.0
        quad_y = 1.0 - fitted_y_px * y_scale
        quad_width = fit_w * x_scale
        quad_height = fit_h * y_scale
        
        # Minimal OpenGL state changes - bind once, render, release
        try:
            texture.bind()
            glBegin(GL_QUADS)
            glTexCoord2f(0.0, 1.0); glVertex2f(quad_x, quad_y - quad_height)
            glTexCoord2f(1.0, 1.0); glVertex2f(quad_x + quad_width, quad_y - quad_height)
            glTexCoord2f(1.0, 0.0); glVertex2f(quad_x + quad_width, quad_y)
            glTexCoord2f(0.0, 0.0); glVertex2f(quad_x, quad_y)
            glEnd()
            texture.release()
        except:
            pass  # Skip rendering on error - don't fall back during fast mode

    def render_grid_cell_placeholder(self, idx, page_num, w_px, h_px):
        """Render a very subtle loading placeholder - nearly invisible"""
        if idx >= len(self._grid_cells):
            return
            
        cell = self._grid_cells[idx]
        
        # Convert pixel coordinates to OpenGL coordinates  
        px_to_gl_x = lambda px: (px / w_px) * 2.0 - 1.0
        px_to_gl_y = lambda py: 1.0 - (py / h_px) * 2.0
        
        # Calculate cell bounds in OpenGL coordinates
        x1 = px_to_gl_x(cell['page_x_px'])
        y1 = px_to_gl_y(cell['page_y_px'])
        x2 = px_to_gl_x(cell['page_x_px'] + cell['inner_w_px'])
        y2 = px_to_gl_y(cell['page_y_px'] + cell['inner_h_px'])
        
        # Disable texturing for placeholder
        glBindTexture(GL_TEXTURE_2D, 0)
        
        # Draw VERY SUBTLE background - almost invisible
        glColor3f(0.15, 0.15, 0.15)  # Very dark gray, barely visible
        glBegin(GL_QUADS)
        glVertex2f(x1, y2)  # Bottom-left
        glVertex2f(x2, y2)  # Bottom-right
        glVertex2f(x2, y1)  # Top-right
        glVertex2f(x1, y1)  # Top-left
        glEnd()
        
        # Draw extremely subtle border - optional
        glColor3f(0.18, 0.18, 0.18)  # Just slightly lighter, barely visible
        glBegin(GL_LINE_LOOP)
        glVertex2f(x1, y2)  # Bottom-left
        glVertex2f(x2, y2)  # Bottom-right
        glVertex2f(x2, y1)  # Top-right  
        glVertex2f(x1, y1)  # Top-left
        glEnd()
        
        # Reset color for next operations
        glColor3f(1.0, 1.0, 1.0)

    def queue_page_for_background_loading(self, page_num, is_priority):
        """Queue a page for background loading without blocking - ULTRA FAST STREAMING"""
        viewer = self.window()
        if not (hasattr(viewer, 'renderer') and viewer.renderer):
            return
            
        # Use ultra-low quality for instant streaming feel
        is_interacting = (getattr(self, '_interaction_mode', False) or 
                        getattr(self, '_fast_zoom_mode', False) or
                        getattr(self, 'is_temp_zoomed', False))
        
        # Ultra-fast streaming quality - like GPU texture streaming
        base_quality = 1.2 if is_interacting else 1.8  # Even lower for instant feel
        
        # IMMEDIATE QUEUE for priority pages - no delays for GPU-like streaming
        if is_priority:
            try:
                viewer.renderer.add_page_to_queue(page_num, priority=True, quality=base_quality)
            except:
                pass
        else:
            # Minimal stagger for background pages - feel like GPU streaming
            delay = 10  # Reduced from 100ms to 10ms for instant streaming feel
            QTimer.singleShot(delay, lambda: (
                viewer.renderer.add_page_to_queue(page_num, priority=False, quality=base_quality)
                if hasattr(viewer, 'renderer') and viewer.renderer else None
            ))

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
        """🚀 NON-BLOCKING: Computes grid layout using cached dimensions only - no UI freeze"""
        w_px = max(1, self.width())
        h_px = max(1, self.height())
        cols = max(1, self.grid_cols)
        rows = max(1, self.grid_rows)
        
        viewer = self.window()
        start_page = getattr(viewer, 'current_page', self.current_page)
        total_pages = getattr(viewer, 'total_pages', getattr(self, 'total_pages', 0))
        
        # 🚀 INSTANT LAYOUT: Use only cached dimensions - no blocking operations
        page_aspects = []
        has_missing_dimensions = False
        
        for idx in range(rows * cols):
            page_num = start_page + idx
            if page_num >= total_pages:
                page_aspects.append(0.75)  # Default portrait aspect
                continue
                
            # Only use cached dimensions - NEVER block for missing ones
            page_dims = self.texture_cache.get_dimensions(page_num)
            if page_dims and page_dims[1] > 0:
                aspect = page_dims[0] / page_dims[1]
                page_aspects.append(aspect)
            else:
                # Missing dimensions - use default and flag for later update
                page_aspects.append(0.75)
                has_missing_dimensions = True
        
        avg_aspect = sum(page_aspects) / len(page_aspects) if page_aspects else 0.75

        # Calculate responsive gap based on widget size and configuration
        gap = min(max(self.grid_gap_ratio * min(w_px, h_px), self.grid_min_gap), self.grid_max_gap)
        
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
        
        # 🚀 DEFERRED UPDATE: If dimensions were missing, schedule layout refinement
        if has_missing_dimensions:
            QTimer.singleShot(200, self._refine_grid_layout_when_ready)

    def _refine_grid_layout_when_ready(self):
        """🚀 BACKGROUND: Refine grid layout once page dimensions are loaded"""
        if not self.grid_mode:
            return
        
        viewer = self.window()
        start_page = getattr(viewer, 'current_page', self.current_page)
        total_pages = getattr(viewer, 'total_pages', getattr(self, 'total_pages', 0))
        
        # Check if we now have dimensions for more pages
        missing_count = 0
        for idx in range(self.grid_cols * self.grid_rows):
            page_num = start_page + idx
            if page_num >= total_pages:
                continue
            if not self.texture_cache.get_dimensions(page_num):
                missing_count += 1
        
        # If still missing many dimensions, try again later
        if missing_count > self.grid_cols * self.grid_rows * 0.5:
            QTimer.singleShot(300, self._refine_grid_layout_when_ready)
            return
        
        # Enough dimensions available - recalculate layout silently
        old_cached_size = self._grid_cached_size
        self._grid_cached_size = (0, 0, 0, 0)  # Force recalculation
        self.compute_grid_layout()
        self.update()  # Trigger redraw with refined layout

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
    
    def _update_performance_stats(self, frame_start_time):
        """🚀 LAYER 3: Update performance monitoring statistics"""
        import time
        current_time = time.time()
        
        if hasattr(self, '_last_frame_time'):
            frame_duration = current_time - self._last_frame_time
            
            # Update frame times with rolling window
            self._performance_monitor['frame_times'].append(frame_duration)
            if len(self._performance_monitor['frame_times']) > 30:  # Keep last 30 frames
                self._performance_monitor['frame_times'].pop(0)
            
            # Calculate average frame time
            if self._performance_monitor['frame_times']:
                self._performance_monitor['avg_frame_time'] = sum(self._performance_monitor['frame_times']) / len(self._performance_monitor['frame_times'])
            
            # Update performance level based on frame times
            avg_frame_time = self._performance_monitor['avg_frame_time'] * 1000  # Convert to ms
            
            previous_level = self._performance_monitor.get('performance_level', 'high')
            
            if avg_frame_time < 20:  # < 20ms = >50 FPS
                self._performance_monitor['performance_level'] = 'high'
                self._performance_quality_multiplier = 1.0
            elif avg_frame_time < 33:  # < 33ms = >30 FPS  
                self._performance_monitor['performance_level'] = 'medium'
                self._performance_quality_multiplier = 0.8
            else:  # >33ms = <30 FPS
                self._performance_monitor['performance_level'] = 'low'
                self._performance_quality_multiplier = 0.6
            
            # Auto-pause renderer when performance drops to very low levels
            current_level = self._performance_monitor['performance_level']
            if hasattr(self, 'renderer') and self.renderer:
                if current_level == 'low' and previous_level != 'low':
                    # Performance just dropped - pause background rendering
                    if hasattr(self.renderer, 'pause') and not getattr(self, '_renderer_manually_paused', False):
                        print("🚀 Auto-pausing renderer due to low performance")
                        self.renderer.pause()
                        self._renderer_auto_paused = True
                elif current_level != 'low' and previous_level == 'low':
                    # Performance recovered - resume background rendering
                    if hasattr(self.renderer, 'resume') and getattr(self, '_renderer_auto_paused', False):
                        print("🚀 Auto-resuming renderer as performance recovered")
                        self.renderer.resume()
                        self._renderer_auto_paused = False
        
        self._last_frame_time = current_time
        
        # Periodic stats logging (every 5 seconds)
        if current_time - self._performance_monitor['last_stats_time'] > 5.0:
            cache_stats = self.texture_cache.get_cache_stats()
            print(f"🚀 PERFORMANCE STATS - Level: {self._performance_monitor['performance_level']}, "
                  f"Avg Frame: {self._performance_monitor['avg_frame_time']*1000:.1f}ms, "
                  f"Cache Hit: {cache_stats['hit_rate']:.1f}%, "
                  f"Memory: {cache_stats['memory_mb']:.1f}MB, "
                  f"Textures: {cache_stats['textures']}")
            self._performance_monitor['last_stats_time'] = current_time
    
    def _get_adaptive_quality(self):
        """🚀 LAYER 3: Get quality factor based on system performance"""
        if not self._adaptive_quality_enabled:
            return 1.0
        
        # Base quality on current performance level
        base_quality = self.base_quality
        
        # Apply performance-based multiplier
        adaptive_quality = base_quality * self._performance_quality_multiplier
        
        # Ensure quality stays within reasonable bounds
        return max(1.5, min(self.max_quality, adaptive_quality))
    
    def get_performance_stats(self):
        """🚀 LAYER 3: Get comprehensive performance statistics"""
        cache_stats = self.texture_cache.get_cache_stats()
        
        return {
            **self._performance_monitor,
            'cache_stats': cache_stats,
            'adaptive_quality': self._get_adaptive_quality(),
            'quality_multiplier': self._performance_quality_multiplier
        }

    def process_pending_images(self, max_items=1):
        """Convert up to max_items QImage entries to GPU textures per call - ULTRA AGGRESSIVE"""
        if max_items == 0:
            return  # Skip entirely during interactions
            
        layout_needs_update = False
        count = 0
        
        # AGGRESSIVE processing - much longer time slices for faster processing
        import time
        start_time = time.time()
        max_time_slice = 0.008  # 8ms maximum processing time for much more work (was 1ms)
        
        while self._pending_images and count < max_items:
            # Check if we've exceeded our time budget
            if time.time() - start_time > max_time_slice:
                break  # Stop processing to prevent lag
                
            try:
                item = self._pending_images.pop(0)
                
                # Check if this is a tuple with the expected structure
                if not isinstance(item, tuple) or len(item) < 5:
                    count += 1
                    continue
                
                page_num, qimage, page_w, page_h, quality = item
                
                # Skip if the image is invalid
                if qimage.isNull() or qimage.width() == 0 or qimage.height() == 0:
                    count += 1
                    continue
                
                # ULTRA FAST texture creation - minimize GPU blocking
                try:
                    # Check if this is new dimension info for grid layout
                    had_dimensions = self.texture_cache.get_dimensions(page_num) is not None
                    
                    # Create texture with optimized settings for speed
                    texture = self.texture_cache.add_texture_fast(page_num, qimage, page_w, page_h, quality)
                    
                    if texture:
                        # Minimal updates - avoid expensive operations
                        if self.grid_mode:
                            # Only update grid if page is visible
                            start_page = getattr(self.window(), 'current_page', self.current_page)
                            pages_needed = self.grid_cols * self.grid_rows
                            if start_page <= page_num < start_page + pages_needed:
                                self.grid_textures[page_num] = texture
                                if not had_dimensions:
                                    layout_needs_update = True
                        else:
                            # Single page mode: minimal update
                            viewer_current_page = getattr(self.window(), 'current_page', self.current_page)
                            if page_num == viewer_current_page:
                                self.set_page_texture(texture, page_w, page_h)
                                
                except Exception as e:
                    # Skip problematic textures to prevent blocking
                    pass
                    
                count += 1
                
            except Exception as e:
                count += 1
                continue
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
        """Set OpenGL background to match container color exactly"""
        # Always use the same dark gray as the container: rgb(38, 38, 38)
        glClearColor(38/255.0, 38/255.0, 38/255.0, 1.0)
        self.update()
    
    def get_zoom_adjusted_quality(self, progressive=False, fast_zoom=False):
        """Calculate rendering quality based on current zoom level and system resources
        
        Args:
            progressive: If True, returns a lower quality for immediate display
            fast_zoom: If True, use optimized quality for responsive zooming
        """
        # Get file size for adaptive quality
        file_size_mb = 0
        try:
            viewer = self.window()
            if viewer and hasattr(viewer, 'pdf_path') and os.path.exists(viewer.pdf_path):
                file_size_mb = os.path.getsize(viewer.pdf_path) / (1024 * 1024)
        except:
            pass
        
        # Memory-aware quality limits based on PDF size
        if file_size_mb > 50:  # Very large PDF
            max_allowed_quality = 3.5
            base_multiplier = 0.8
        elif file_size_mb > 30:  # Large PDF
            max_allowed_quality = 4.5
            base_multiplier = 0.9
        elif file_size_mb > 10:  # Medium PDF
            max_allowed_quality = 6.0
            base_multiplier = 1.0
        else:  # Small PDF
            max_allowed_quality = self.max_quality
            base_multiplier = 1.0
        
        # Calculate effective zoom including temporary zoom
        effective_zoom = self.zoom_factor
        if hasattr(self, 'is_temp_zoomed') and self.is_temp_zoomed:
            effective_zoom *= getattr(self, 'temp_zoom_factor', 1.0)
            
        if effective_zoom <= self.quality_zoom_threshold:
            # High quality even at low zoom, extra boost for 100% zoom
            if 0.85 <= effective_zoom <= 1.15:  # Even wider range for 100% zoom detection
                # Special case for near 100% zoom - use highest quality for razor-sharp text
                base_scaled = min(self.base_quality * 1.5 * base_multiplier, max_allowed_quality)
                return min(base_scaled, max_allowed_quality)
            return min(self.base_quality * base_multiplier, max_allowed_quality)
        
        # For fast zoom at high levels, use readable quality but limited processing
        if fast_zoom and effective_zoom > 2.0:
            # Use decent quality during fast zoom but cap it for performance
            return min(self.base_quality * 1.2 * base_multiplier, min(4.0, max_allowed_quality))
        
        # Optimized quality scaling for performance - more conservative at high zoom
        zoom_excess = effective_zoom - self.quality_zoom_threshold
        
        # Performance-focused quality scaling with better text emphasis
        if effective_zoom <= 3.0:
            # Normal zoom: increased scaling for better text quality
            quality_boost = zoom_excess * 0.45 * base_multiplier  # Scale with memory limits
        elif effective_zoom <= 6.0:
            # High zoom: better scaling for text clarity
            base_boost = (3.0 - self.quality_zoom_threshold) * 0.45 * base_multiplier
            high_boost = (effective_zoom - 3.0) * 0.25 * base_multiplier
            quality_boost = base_boost + high_boost
        else:
            # Ultra-high zoom: maximum text quality (but limited by memory)
            base_boost = (3.0 - self.quality_zoom_threshold) * 0.45 * base_multiplier
            high_boost = 3.0 * 0.25 * base_multiplier
            ultra_boost = min((effective_zoom - 6.0) * 0.15, 0.5) * base_multiplier
            quality_boost = base_boost + high_boost + ultra_boost
            
        adjusted_quality = min(self.base_quality + quality_boost, max_allowed_quality)
        
        # For progressive loading, return a lower quality first
        if progressive:
            return max(self.base_quality * base_multiplier, adjusted_quality * 0.7)
        
        return adjusted_quality
    
    def get_immediate_quality(self):
        """Get quality for immediate display (fast rendering for single page)"""
        # 🚀 OPTIMIZATION: Use moderate quality for immediate display - balance speed and clarity
        try:
            viewer = self.window()
            if viewer and hasattr(viewer, 'pdf_path') and os.path.exists(viewer.pdf_path):
                file_size_mb = os.path.getsize(viewer.pdf_path) / (1024 * 1024)
                if file_size_mb > 30:  # Large PDF
                    return 2.2  # Faster rendering for large files but readable
                elif file_size_mb > 10:  # Medium PDF
                    return 2.5  # Good balance
        except:
            pass
        
        # Use moderate immediate quality for readable immediate display
        return 2.8  # Higher than before for better immediate quality
    
    def check_quality_change(self):
        """Check if quality needs updating due to zoom change and re-render if needed"""
        new_quality = self.get_zoom_adjusted_quality()
        
        # Dynamic quality sensitivity based on zoom level - be less sensitive at high zoom
        effective_zoom = self.zoom_factor
        if hasattr(self, 'is_temp_zoomed') and self.is_temp_zoomed:
            effective_zoom *= getattr(self, 'temp_zoom_factor', 1.0)
        
        # Much higher sensitivity threshold for high zoom to prevent lag - be very conservative
        if effective_zoom > 4.0:
            sensitivity = 2.0  # Very insensitive at high zoom - only major changes
        elif effective_zoom > 2.0:
            sensitivity = 1.5  # Less sensitive at medium zoom
        else:
            sensitivity = 1.0  # Still less sensitive than before for low zoom
        
        if abs(new_quality - self.last_zoom_quality) > sensitivity:
            self.last_zoom_quality = new_quality
            # Re-render current content with progressive quality
            viewer = self.window()
            if viewer and hasattr(viewer, 'render_current_page'):
                if self.grid_mode:
                    # 🚀 GRID ZOOM FIX: Force upgrade of visible grid textures
                    self._upgrade_grid_quality(new_quality)
                    viewer.render_current_page()
                else:
                    # 🚀 SINGLE PAGE ZOOM FIX: Make single page mode as aggressive as grid mode for quality
                    current_page = getattr(viewer, 'current_page', 0)
                    
                    # Calculate effective zoom including temporary zoom
                    effective_zoom = self.zoom_factor
                    if hasattr(self, 'is_temp_zoomed') and self.is_temp_zoomed:
                        effective_zoom *= getattr(self, 'temp_zoom_factor', 1.0)
                    
                    existing_texture = self.texture_cache.get_texture(current_page)
                    current_quality = self.texture_cache.get_best_quality_for_page(current_page) if existing_texture else 0.0
                    
                    # NEW: Single page mode should request high-quality textures just like grid mode
                    if new_quality > current_quality + 0.5:  # Same threshold as grid mode
                        # Request new high-quality texture while keeping current one visible
                        try:
                            viewer.renderer.add_page_to_queue(current_page, priority=True, quality=new_quality)
                            print(f"🔍 Single page quality upgrade: page {current_page}, from {current_quality:.1f} to {new_quality:.1f} (zoom: {effective_zoom:.1f}x)")
                        except Exception as e:
                            print(f"Error queueing single page {current_page} for quality upgrade: {e}")
                    
                    # Only clear texture in extreme cases to avoid visual disruption
                    if existing_texture and effective_zoom > 8.0 and new_quality > current_quality + 3.0:
                        self.texture_cache.remove_texture(current_page)
                        self.set_page_texture(None)
                
                # Trigger progressive re-render (but mainly for background high-quality loading)
                viewer.render_current_page_progressive()
    
    def _schedule_delayed_quality_check(self):
        """Schedule a delayed quality check to avoid lag during rapid zoom changes"""
        if not hasattr(self, '_delayed_quality_timer'):
            self._delayed_quality_timer = QTimer()
            self._delayed_quality_timer.setSingleShot(True)
            self._delayed_quality_timer.timeout.connect(self.check_quality_change)
        
        # Use longer delay for high zoom to reduce processing load
        effective_zoom = self.zoom_factor
        if hasattr(self, 'is_temp_zoomed') and self.is_temp_zoomed:
            effective_zoom *= getattr(self, 'temp_zoom_factor', 1.0)
        
        # Use shorter delays for faster quality recovery while still preventing lag during interaction
        if effective_zoom > 5.0:
            delay = 600  # 600ms for very high zoom
        elif effective_zoom > 3.0:
            delay = 400  # 400ms for high zoom  
        else:
            delay = 300  # 300ms for moderate zoom
            
        self._delayed_quality_timer.start(delay)
    
    def _upgrade_grid_quality(self, new_quality):
        """🚀 GRID ZOOM FIX: Optimized quality upgrade that avoids lag during zoom"""
        viewer = self.window()
        if not (viewer and hasattr(viewer, 'current_page') and hasattr(viewer, 'renderer')):
            return
        
        # Skip quality upgrades if we're in fast zoom mode to prevent lag
        if getattr(self, '_fast_zoom_mode', False):
            return
        
        start_page = viewer.current_page
        pages_needed = min(self.grid_cols * self.grid_rows, 9)  # Limit to max 9 pages for performance
        
        # Less aggressive quality target to reduce processing load
        target_quality = min(new_quality, self.max_quality)
        
        # 🔍 GRID FIX: For small grids (2x2), upgrade ALL pages, not just first row
        if len(self._grid_cells) <= 4:  # 2x2 grid or smaller
            # Always upgrade all visible pages for small grids
            pages_needed = min(pages_needed, len(self._grid_cells))
        else:
            # For larger grids, only upgrade first row during zoom for better responsiveness
            first_row_only = getattr(self, 'is_temp_zoomed', False)
            if first_row_only:
                pages_needed = min(pages_needed, self.grid_cols)
        
        upgrade_count = 0
        for i in range(pages_needed):
            page_num = start_page + i
            if page_num >= getattr(viewer, 'total_pages', 0):
                break
            
            current_quality = self.texture_cache.get_best_quality_for_page(page_num)
            
            # Less aggressive quality checking to reduce queue pressure
            if target_quality > current_quality + 0.5:  # Increased threshold
                # Priority for first page, but process all needed pages
                priority = i == 0
                try:
                    viewer.renderer.add_page_to_queue(page_num, priority=priority, quality=target_quality)
                    upgrade_count += 1
                    print(f"🔍 Grid quality upgrade: page {page_num}, from {current_quality:.1f} to {target_quality:.1f} (#{upgrade_count})")
                    # 🔍 GRID FIX: Increase limit for small grids to ensure all 4 pages in 2x2 get upgraded
                    if len(self._grid_cells) <= 4:
                        max_upgrades = 4  # Allow all 4 pages in 2x2 to be upgraded
                    else:
                        max_upgrades = 3  # Limit larger grids
                    
                    if upgrade_count >= max_upgrades:
                        break
                except Exception as e:
                    print(f"Error queueing page {page_num} for quality upgrade: {e}")
    
    def _enable_fast_zoom_mode(self):
        """Enable fast zoom mode to reduce processing during rapid zoom changes"""
        import time
        self._fast_zoom_mode = True
        self._fast_zoom_start_time = time.time() * 1000
        
        # Auto-disable fast zoom mode after timeout
        if not hasattr(self, '_fast_zoom_timer'):
            self._fast_zoom_timer = QTimer()
            self._fast_zoom_timer.setSingleShot(True)
            self._fast_zoom_timer.timeout.connect(self._disable_fast_zoom_mode)
        
        self._fast_zoom_timer.start(self._fast_zoom_timeout)
    
    def _disable_fast_zoom_mode(self):
        """Disable fast zoom mode and trigger quality update"""
        self._fast_zoom_mode = False
        # Trigger quality check now that zooming has stabilized - but only if not panning
        if not getattr(self, 'is_panning', False):
            self.check_quality_change()
        
        # Update status bar with current zoom level
        self.update_status()
    
    def update_status(self):
        """Update status bar with current info"""
        viewer = self.window()
        if not viewer or not hasattr(viewer, 'status_bar'):
            return
            
        # Format zoom as percentage with 0 decimal places
        zoom_percent = int(self.zoom_factor * 100)
        quality = self.get_zoom_adjusted_quality()
        
        # Include page info if available
        if hasattr(viewer, 'current_page') and hasattr(viewer, 'total_pages'):
            page_info = f"Page {viewer.current_page + 1}/{viewer.total_pages}"
        else:
            page_info = ""
            
        # Create status message
        if page_info:
            status_message = f"{page_info} | Zoom: {zoom_percent}% | Quality: {quality:.1f}"
        else:
            status_message = f"Zoom: {zoom_percent}% | Quality: {quality:.1f}"
            
        viewer.status_bar.showMessage(status_message)
    
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
        # Enable interaction mode during zoom
        self._interaction_mode = True
        
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
        
        # Update status bar with new zoom level
        self.update_status()
        
        # For high zoom levels, enable fast zoom mode and use delayed quality check
        if self.zoom_factor > 3.0:
            self._enable_fast_zoom_mode()
            self._schedule_delayed_quality_check()
        else:
            self.check_quality_change()
        
        # Auto-disable interaction mode after delay
        if not hasattr(self, '_zoom_interaction_timer'):
            self._zoom_interaction_timer = QTimer()
            self._zoom_interaction_timer.setSingleShot(True)
            self._zoom_interaction_timer.timeout.connect(lambda: setattr(self, '_interaction_mode', False))
        
        self._zoom_interaction_timer.start(300)  # 300ms interaction window
        self.update()
    
    def zoom_out(self, cursor_pos=None):
        # Enable interaction mode during zoom
        self._interaction_mode = True
        
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
            
        # Update status bar with new zoom level
        self.update_status()
        
        # For high zoom levels, enable fast zoom mode and use delayed quality check
        if old_zoom > 3.0:  # Use old_zoom to catch zoom out from high levels
            self._enable_fast_zoom_mode()
            self._schedule_delayed_quality_check()
        else:
            self.check_quality_change()
        
        # Auto-disable interaction mode after delay
        if not hasattr(self, '_zoom_interaction_timer'):
            self._zoom_interaction_timer = QTimer()
            self._zoom_interaction_timer.setSingleShot(True)
            self._zoom_interaction_timer.timeout.connect(lambda: setattr(self, '_interaction_mode', False))
        
        self._zoom_interaction_timer.start(300)  # 300ms interaction window
        self.update()
    
    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.check_quality_change()
        self.update_status()
        self.update()
        
    def show_context_menu(self, position):
        """Display context menu with zoom and navigation options"""
        menu = QMenu(self)
        
        # Navigation section
        next_page_action = menu.addAction("Next Page")
        prev_page_action = menu.addAction("Previous Page")
        
        # Add separator
        menu.addSeparator()
        
        # Zoom section
        zoom_in_action = menu.addAction("Zoom In (+)")
        zoom_out_action = menu.addAction("Zoom Out (-)")
        reset_zoom_action = menu.addAction("Reset Zoom (100%)")
        
        # Add separator
        menu.addSeparator()
        
        # View options
        toggle_grid_action = menu.addAction("Toggle Grid View")
        if self.grid_mode:
            toggle_grid_action.setText("Exit Grid View")
        else:
            toggle_grid_action.setText("Grid View")
            
        # Get action from menu
        action = menu.exec(self.mapToGlobal(position))
        
        # Handle actions
        viewer = self.window()
        if action == next_page_action:
            if hasattr(viewer, 'next_page'):
                viewer.next_page()
        elif action == prev_page_action:
            if hasattr(viewer, 'prev_page'):
                viewer.prev_page()
        elif action == zoom_in_action:
            self.zoom_in(position)
        elif action == zoom_out_action:
            self.zoom_out(position)
        elif action == reset_zoom_action:
            self.reset_zoom()
        elif action == toggle_grid_action:
            if hasattr(viewer, 'toggle_grid_view'):
                # Toggle the action state first, then call the toggle method
                current_state = viewer.grid_view_action.isChecked()
                new_state = not current_state
                viewer.grid_view_action.setChecked(new_state)
                viewer.grid_view_menu_action.setChecked(new_state)
                viewer.toggle_grid_view()
            else:
                print("Context menu: viewer does not have toggle_grid_view method")
    
    def zoom_to_grid_page(self, page_num):
        """
        Temporarily zooms to a specific page within the grid view to fit the window.
        """
        if not self.grid_mode or not self._grid_cells:
            return

        # Find the target page in the current grid
        viewer = self.window()
        start_page = getattr(viewer, 'current_page', self.current_page)
        
        # Calculate which cell contains this page
        page_index = page_num - start_page
        if page_index < 0 or page_index >= len(self._grid_cells):
            return
        
        cell = self._grid_cells[page_index]
        
        # Get cell position and dimensions
        page_x_px = cell['page_x_px']
        page_y_px = cell['page_y_px']
        inner_w_px = cell['inner_w_px']
        inner_h_px = cell['inner_h_px']
        
        # Calculate the center of the target page in widget coordinates
        page_center_x = page_x_px + inner_w_px / 2
        page_center_y = page_y_px + inner_h_px / 2
        
        # Calculate the zoom factor needed to make the page fill the window with some margin
        # Use 90% of window size to leave a small margin
        zoom_x = (self.width() * 0.9) / inner_w_px
        zoom_y = (self.height() * 0.9) / inner_h_px
        # Use the smaller zoom factor to ensure the entire page fits
        self.temp_zoom_factor = min(zoom_x, zoom_y)

        # Calculate pan so the page center ends up at the widget center in GL coords.
        w_px = max(1, self.width())
        h_px = max(1, self.height())

        # Page center in GL coords before any transforms
        page_gl_x = (page_center_x / w_px) * 2.0 - 1.0
        page_gl_y = 1.0 - (page_center_y / h_px) * 2.0

        # The widget center in GL coords is (0,0). After scaling by temp_zoom_factor,
        # the page_gl_x * temp_zoom_factor + pan_x should equal 0 -> pan_x = -page_gl_x * temp_zoom
        self.temp_pan_x = -page_gl_x * self.temp_zoom_factor
        self.temp_pan_y = -page_gl_y * self.temp_zoom_factor

        self.is_temp_zoomed = True
        
        # Enable interaction mode during grid zoom for smooth performance
        self._interaction_mode = True
        
        # Enable fast zoom mode for grid temp zoom to improve responsiveness
        self._enable_fast_zoom_mode()
        
        # Trigger quality check with longer delay to avoid lag during grid zoom
        if not hasattr(self, '_zoom_quality_timer'):
            self._zoom_quality_timer = QTimer()
            self._zoom_quality_timer.setSingleShot(True)
            self._zoom_quality_timer.timeout.connect(lambda: self.check_quality_change())
        
        # Longer delay for grid zoom to prevent lag during rapid zooming
        self._zoom_quality_timer.start(600)  # Increased delay for grid zoom
        
        # Auto-disable interaction mode after delay
        if not hasattr(self, '_grid_zoom_timer'):
            self._grid_zoom_timer = QTimer()
            self._grid_zoom_timer.setSingleShot(True)
            self._grid_zoom_timer.timeout.connect(lambda: setattr(self, '_interaction_mode', False))
        
        self._grid_zoom_timer.start(800)  # Extended interaction window for grid zoom
        
        # Request high-quality texture for the focused page if needed - but more performance conscious
        viewer = self.window()
        if viewer and hasattr(viewer, 'renderer') and viewer.renderer:
            effective_zoom = self.zoom_factor * self.temp_zoom_factor
            final_quality = self.get_zoom_adjusted_quality()
            current_best_quality = self.texture_cache.get_best_quality_for_page(page_num)
            
            # Only queue better quality if gap is significant and we're not in heavy interaction
            if (final_quality > current_best_quality + 2.5 and  # Higher threshold for grid zoom
                not getattr(self, '_interaction_mode', False)):
                # Use background priority for smooth zooming
                viewer.renderer.add_page_to_queue(page_num, priority=False, quality=final_quality)
        
        self.update()

    def reset_grid_zoom(self):
        """Resets the temporary zoom on the grid view."""
        if self.is_temp_zoomed:
            self.is_temp_zoomed = False
            self.temp_zoom_factor = 1.0
            self.temp_pan_x = 0.0
            self.temp_pan_y = 0.0
            self.update()

    def wheelEvent(self, event):
        # Enable interaction mode during wheel zoom for better performance
        self._interaction_mode = True
        
        # Auto-disable interaction mode after a delay
        if not hasattr(self, '_interaction_timer'):
            self._interaction_timer = QTimer()
            self._interaction_timer.setSingleShot(True)
            self._interaction_timer.timeout.connect(lambda: setattr(self, '_interaction_mode', False))
        
        self._interaction_timer.start(500)  # 500ms timeout
        
        # Zoom with mouse wheel towards cursor position
        cursor_pos = event.position()
        delta = event.angleDelta().y()
        
        if self.grid_mode and self.is_temp_zoomed:
            # If in temp zoom mode, adjust the temp zoom factor instead of regular zoom
            old_temp_zoom = self.temp_zoom_factor
            if delta > 0:
                self.temp_zoom_factor = min(self.temp_zoom_factor * 1.2, 10.0)
            else:
                self.temp_zoom_factor = max(self.temp_zoom_factor / 1.2, 0.1)
            
            # Adjust temp pan to zoom towards cursor position
            if cursor_pos is not None:
                zoom_ratio = self.temp_zoom_factor / old_temp_zoom
                # Convert cursor position to normalized coordinates (-1 to 1)
                cursor_x = (cursor_pos.x() / self.width()) * 2.0 - 1.0
                cursor_y = 1.0 - (cursor_pos.y() / self.height()) * 2.0
                
                # Adjust temp pan to keep cursor point stable
                self.temp_pan_x = cursor_x + (self.temp_pan_x - cursor_x) * zoom_ratio
                self.temp_pan_y = cursor_y + (self.temp_pan_y - cursor_y) * zoom_ratio
            
            # Enable fast zoom mode for rapid temp zoom changes
            self._enable_fast_zoom_mode()
            
            # Extend interaction mode during wheel zoom in temp zoom
            if not hasattr(self, '_temp_zoom_interaction_timer'):
                self._temp_zoom_interaction_timer = QTimer()
                self._temp_zoom_interaction_timer.setSingleShot(True)
                self._temp_zoom_interaction_timer.timeout.connect(lambda: setattr(self, '_interaction_mode', False))
            
            self._temp_zoom_interaction_timer.start(600)  # Extended timeout for temp zoom
            
            # OPTIMIZED: Much longer delay for quality checks during zoom to prevent lag
            # Only check quality after user has stopped zooming for a while
            if not hasattr(self, '_quality_check_timer'):
                self._quality_check_timer = QTimer()
                self._quality_check_timer.setSingleShot(True)
                self._quality_check_timer.timeout.connect(self.check_quality_change)
            
            # Much longer delay - only upgrade quality when zoom stabilizes
            self._quality_check_timer.start(1000)  # 1 second delay for smoother zooming
            
            # IMMEDIATE UPDATE: Grid zoom changes are instant regardless of texture loading state
            self.update()
        else:
            # Normal zoom behavior
            if delta > 0:
                self.zoom_in(cursor_pos)
            else:
                self.zoom_out(cursor_pos)
            
            if self.grid_mode:
                self.reset_grid_zoom()
                # IMMEDIATE UPDATE: Ensure grid responds instantly to zoom changes
                self.update()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = True
            self._interaction_mode = True
            self._active_panning = True
            self.last_mouse_pos = event.position()
            
            # CRITICAL: Pause background rendering to eliminate CPU dependency
            if hasattr(self, 'renderer') and self.renderer:
                if hasattr(self.renderer, 'pause'):
                    self.renderer.pause()
                    
            # Don't disable Qt updates - we want real-time panning feedback
            if self.grid_mode:
                self._ultra_fast_mode = True
                # Stop any timer-based processing during panning
                if hasattr(self, '_performance_timer'):
                    self._performance_timer.stop()
            # Keep Qt updates enabled for responsive panning visual feedback
                
        elif event.button() == Qt.MouseButton.MiddleButton:
            # Middle mouse click goes to next page
            viewer = self.window()
            if viewer and hasattr(viewer, 'next_page'):
                viewer.next_page()
    
    def mouseMoveEvent(self, event):
        """REAL-TIME PANNING - Smooth visual feedback with optimized performance"""
        
        if self.is_panning:
            delta = event.position() - self.last_mouse_pos
            
            # Skip microscopic movements to reduce unnecessary updates
            if abs(delta.x()) < 2.0 and abs(delta.y()) < 2.0:
                return
            
            # Light throttling for performance while maintaining responsiveness
            current_time = time.time()
            if not hasattr(self, '_last_pan_time'):
                self._last_pan_time = 0
            
            # Throttle to 60 FPS max for smooth panning
            throttle_interval = 1.0 / 60.0  # 60 FPS limit
            if current_time - self._last_pan_time < throttle_interval:
                return
            
            self._last_pan_time = current_time
            
            # Calculate pan delta with appropriate scaling
            if self.grid_mode:
                scale_factor = 4.0 / max(self.width(), self.height())
                if self.is_temp_zoomed:
                    self.temp_pan_x += delta.x() * scale_factor
                    self.temp_pan_y -= delta.y() * scale_factor
                else:
                    self.pan_x += delta.x() * scale_factor
                    self.pan_y -= delta.y() * scale_factor
            else:
                # Single page mode
                scale_factor = 2.0 / max(self.width(), self.height())
                self.pan_x += delta.x() * scale_factor
                self.pan_y -= delta.y() * scale_factor
            
            # Enable ultra-fast mode flags for performance
            self._skip_heavy_processing = True
            self._active_panning = True
            self._ultra_fast_mode = True
            
            # Real-time visual feedback - always update
            self.update()
            
            self.last_mouse_pos = event.position()
            return
        
        # Reset flags when not panning
        self._skip_heavy_processing = False
        self._ultra_fast_mode = False
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False
            self._active_panning = False
            self._interaction_mode = False
            self._skip_heavy_processing = False
            self._ultra_fast_mode = False
            
            # Clear the cached texture binding to force refresh
            if hasattr(self, '_last_bound_texture'):
                self._last_bound_texture = None
                
            # Restart performance timer if it exists
            if hasattr(self, '_performance_timer'):
                self._performance_timer.start(100)
            
            # CRITICAL: Resume background rendering after panning
            if hasattr(self, 'renderer') and self.renderer:
                if hasattr(self.renderer, 'resume') and not getattr(self, '_renderer_auto_paused', False):
                    self.renderer.resume()
            
            # Force a complete repaint to restore proper rendering
            self.update()
            
            # Schedule a second update to ensure content is restored
            QTimer.singleShot(50, self.update)
            
            # Quality check after a brief delay
            if self.grid_mode:
                QTimer.singleShot(100, self.check_quality_change)
            
            # Re-enable rendering after panning stops, but with delay
            if not hasattr(self, '_pan_end_timer'):
                self._pan_end_timer = QTimer()
                self._pan_end_timer.setSingleShot(True)
                self._pan_end_timer.timeout.connect(self._on_pan_end)
            
            # Small delay before quality check to ensure smooth pan end
            self._pan_end_timer.start(250)  # Reduced delay for faster quality recovery
    
    def _on_pan_end(self):
        """Handle actions when panning ends"""
        # 🚀 GRID ZOOM FIX: Trigger quality upgrade for grid mode
        if self.grid_mode:
            current_quality = self.get_zoom_adjusted_quality()
            self._upgrade_grid_quality(current_quality)
        
        # Standard quality check
        self.check_quality_change()


class PDFViewer(QMainWindow):
    """Main PDF viewer application"""
    
    def __init__(self):
        super().__init__()
        # Basic initialization only - defer heavy setup
        self.pdf_path = ""
        self.pdf_doc = None
        self.current_page = 0
        self.total_pages = 0
        self.render_quality = 3.0
        
        self.renderer = None
        self.thumbnail_worker = None
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptDrops, True)
        
        # Defer UI setup to after window is shown
        QTimer.singleShot(0, self.setup_ui_deferred)
        
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        key = event.key()
        
        # Navigation shortcuts
        if key == Qt.Key.Key_Space or key == Qt.Key.Key_Right or key == Qt.Key.Key_Down or key == Qt.Key.Key_N:
            # Next page
            self.next_page()
        elif key == Qt.Key.Key_Left or key == Qt.Key.Key_Up or key == Qt.Key.Key_P or key == Qt.Key.Key_Backspace:
            # Previous page
            self.prev_page()
        elif key == Qt.Key.Key_Home:
            # First page
            self.goto_page(0)
        elif key == Qt.Key.Key_End:
            # Last page
            self.goto_page(self.total_pages - 1)
            
        # Zoom shortcuts
        elif key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            # Zoom in
            self.zoom_in_clicked()
        elif key == Qt.Key.Key_Minus:
            # Zoom out
            self.zoom_out_clicked()
        elif key == Qt.Key.Key_0:
            # Reset zoom (100%)
            self.reset_zoom_clicked()
        elif key == Qt.Key.Key_G:
            # Toggle grid view
            self.toggle_grid_view()
        
        # 🚀 LAYER 3: Performance stats shortcut
        elif key == Qt.Key.Key_F12:
            # Show performance statistics
            self.show_performance_stats()
        else:
            super().keyPressEvent(event)
        
        # Show basic window immediately
        self.setWindowTitle("GPU-Accelerated PDF Viewer")
        self.setGeometry(100, 100, 1200, 800)
    
    def show_performance_stats(self):
        """🚀 LAYER 3: Display comprehensive performance statistics"""
        try:
            if hasattr(self, 'pdf_widget'):
                stats = self.pdf_widget.get_performance_stats()
                cache_stats = stats['cache_stats']
                
                msg = f"""🚀 LAYER 3 PERFORMANCE STATISTICS 🚀
                
Performance Level: {stats['performance_level'].upper()}
Average Frame Time: {stats['avg_frame_time']*1000:.1f}ms
Target FPS: {1.0/stats['avg_frame_time']:.1f}
Quality Multiplier: {stats['quality_multiplier']:.2f}
Adaptive Quality: {stats['adaptive_quality']:.1f}

GPU TEXTURE CACHE:
Cache Hit Rate: {cache_stats['hit_rate']:.1f}%
Cache Hits: {cache_stats['hits']}
Cache Misses: {cache_stats['misses']}
Evictions: {cache_stats['evictions']}
Memory Usage: {cache_stats['memory_mb']:.1f} MB
Active Textures: {cache_stats['textures']}

OPTIMIZATIONS ACTIVE:
✓ Layer 1: Advanced Thumbnail Generation with Parallel Processing
✓ Layer 2: GPU & Rendering Pipeline Enhancements  
✓ Layer 3: Intelligent Resource Management & Performance Monitoring

Press F12 to refresh stats"""
                
                # Show in message box
                from PyQt6.QtWidgets import QMessageBox
                msg_box = QMessageBox()
                msg_box.setWindowTitle("Performance Statistics")
                msg_box.setText(msg)
                msg_box.setIcon(QMessageBox.Icon.Information)
                msg_box.exec()
            else:
                print("🚀 LAYER 3: PDF widget not available for performance stats")
        except Exception as e:
            print(f"🚀 LAYER 3: Error displaying performance stats: {e}")
    
    def setup_ui_deferred(self):
        """Deferred UI setup for faster startup"""
        self.setup_ui()
        self.setup_shortcuts()
        
    def setup_shortcuts(self):
        """Setup keyboard shortcuts and actions"""
        # Create Help menu
        menu_bar = self.menuBar()
        help_menu = menu_bar.addMenu("Help")
        
        # Keyboard shortcuts action
        shortcuts_action = QAction("Keyboard Shortcuts", self)
        shortcuts_action.triggered.connect(self.show_keyboard_shortcuts)
        help_menu.addAction(shortcuts_action)
        
        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        # Make the menu bar visible
        self.menuBar().show()
    
    def show_keyboard_shortcuts(self):
        """Show a dialog with keyboard shortcuts"""
        shortcuts_text = """
        <h3>Keyboard Shortcuts</h3>
        <table>
            <tr><th>Key</th><th>Action</th></tr>
            <tr><td>Space, Right, Down, N</td><td>Next page</td></tr>
            <tr><td>Left, Up, P, Backspace</td><td>Previous page</td></tr>
            <tr><td>Home</td><td>First page</td></tr>
            <tr><td>End</td><td>Last page</td></tr>
            <tr><td>+, =</td><td>Zoom in</td></tr>
            <tr><td>-</td><td>Zoom out</td></tr>
            <tr><td>0</td><td>Reset zoom (100%)</td></tr>
            <tr><td>G</td><td>Toggle grid view</td></tr>
            <tr><td>F12</td><td>🚀 Show performance statistics</td></tr>
        </table>
        <p><b>Tip:</b> Right-click on the document to show the context menu with these options.</p>
        <p><b>🚀 Performance:</b> Press F12 to see Layer 3 optimization statistics!</p>
        """
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Keyboard Shortcuts")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(shortcuts_text)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h3>GPU-Accelerated PDF Viewer</h3>
        <p>A high-performance PDF viewer that uses GPU acceleration for smooth zooming and rendering.</p>
        <p>Features:</p>
        <ul>
            <li>GPU-accelerated rendering</li>
            <li>Smooth zooming and panning</li>
            <li>Grid view for multiple pages</li>
            <li>Thumbnail navigation</li>
            <li>Drag and drop support</li>
        </ul>
        """
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("About")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(about_text)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()
    
    def setup_ui(self):
        self.setWindowTitle("GPU-Accelerated PDF Viewer")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow { 
                background-color: rgb(38, 38, 38); 
                border: none;
                margin: 0px;
                padding: 0px;
            }
        """)
        self.setContentsMargins(0, 0, 0, 0)
        
        # Create status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready - Use File menu or drop file to open a PDF")
        
        # Central widget
        central_widget = QWidget()
        central_widget.setStyleSheet("""
            QWidget { 
                background-color: rgb(38, 38, 38); 
                border: none;
                margin: 0px;
                padding: 0px;
            }
        """)
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Left panel (thumbnails, bookmarks)
        left_panel = self.create_left_panel()
        
        # PDF viewer widget - create with container for consistent background
        self.pdf_widget = GPUPDFWidget()
        # Remove any background styling from the OpenGL widget itself
        self.pdf_widget.setStyleSheet("QOpenGLWidget { border: none; }")
        
        # Create container widget with the background color
        pdf_container = QWidget()
        pdf_container.setStyleSheet("""
            QWidget { 
                background-color: rgb(38, 38, 38); 
                border: none;
                margin: 0px;
                padding: 0px;
            }
        """)
        
        # Layout to hold the PDF widget inside the container
        pdf_layout = QVBoxLayout(pdf_container)
        pdf_layout.setContentsMargins(0, 0, 0, 0)
        pdf_layout.setSpacing(0)
        pdf_layout.addWidget(self.pdf_widget)
        
        # Defer OpenGL context creation
        self.pdf_widget.setUpdatesEnabled(False)  # Disable updates during startup
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("""
            QSplitter { 
                background-color: rgb(38, 38, 38); 
                border: none;
                margin: 0px;
                padding: 0px;
            } 
            QSplitter::handle { 
                background-color: rgb(60, 60, 60); 
                border: none;
                margin: 0px;
                width: 2px;
            }
        """)
        splitter.setHandleWidth(2)
        splitter.addWidget(left_panel)
        splitter.addWidget(pdf_container)
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Set initial splitter sizes (30% for thumbnails, 70% for PDF)
        splitter.setSizes([300, 900])
        
        # Enable OpenGL context after UI is set up
        QTimer.singleShot(100, self.finalize_ui_setup)
    
    def finalize_ui_setup(self):
        """Finalize UI setup after window is visible"""
        # Enable updates on PDF widget
        self.pdf_widget.setUpdatesEnabled(True)
        
        # Create menu and toolbar after initial window is shown
        QTimer.singleShot(50, self.create_menu_deferred)
    
    def create_menu_deferred(self):
        """Create menu after UI is ready"""
        self.create_menu_bar()
        self.create_toolbar()
        
        # Set initial zoom slider value
        if hasattr(self, 'zoom_slider'):
            self.zoom_slider.setValue(100)
    
    def generate_thumbnails_deferred(self, force_new_pdf=False):
        """Generate thumbnails after main view is ready"""
        try:
            # Check if we're in grid mode - if so, minimize thumbnail generation
            # BUT override this check if we're loading a new PDF (force_new_pdf=True)
            if not force_new_pdf and getattr(self.pdf_widget, 'grid_mode', False):
                print("Grid mode active - skipping heavy thumbnail generation to prioritize grid performance")
                return
            
            # If we're forcing generation for new PDF, show status message
            if force_new_pdf:
                print("New PDF loaded - forcing thumbnail generation even in grid mode")
            
            # Force complete thumbnail regeneration for new PDF
            self._selective_thumbnails_enabled = False
            
            # Generate thumbnails with force clear to ensure old thumbnails are removed
            self.generate_thumbnails(force_clear=True)
            
            # Re-enable selective loading after initial generation
            self._selective_thumbnails_enabled = True
            
            # Initialize tracking variables for selective loading
            if not hasattr(self, '_current_thumb_range'):
                self._current_thumb_range = (0, min(30, self.total_pages - 1))  # Initial range
            
            # Show final loaded status
            file_name = os.path.basename(self.pdf_path)
            zoom_percent = int(self.pdf_widget.zoom_factor * 100)
            quality = self.pdf_widget.get_zoom_adjusted_quality()
            self.status_bar.showMessage(f"Loaded: {file_name} | Page 1/{self.total_pages} | Zoom: {zoom_percent}% | Quality: {quality:.1f}")
            
            # Update PDF widget status
            self.pdf_widget.update_status()
            QTimer.singleShot(2000, lambda: self.status_bar.showMessage("Ready"))
            
            # Start aggressive preloading after initial loading is complete
            QTimer.singleShot(3000, self.start_aggressive_preloading)
            
        except Exception as e:
            print(f"Thumbnail generation error: {e}")
            
        except Exception as e:
            tb = traceback.format_exc()
            # Show the exception and include a copyable traceback for debugging
            QMessageBox.critical(self, "Error", f"Failed to load PDF:\n{str(e)}\n\nTraceback:\n{tb}")
    
    def render_current_page(self):
        if self.pdf_doc and 0 <= self.current_page < self.total_pages:
            if self.pdf_widget.grid_mode:
                self.render_grid_pages()
            else:
                # Single page mode - implement immediate display for maximum responsiveness
                texture = self.pdf_widget.texture_cache.get_texture(self.current_page)
                if texture:
                    # We have a texture, set it immediately without blocking operations
                    cached_dims = self.pdf_widget.texture_cache.get_dimensions(self.current_page)
                    if cached_dims:
                        page_width, page_height = cached_dims
                    else:
                        # Fallback to PDF dimensions but don't block the main thread
                        try:
                            page = self.pdf_doc[self.current_page]
                            page_width = page.rect.width
                            page_height = page.rect.height
                            # Cache these dimensions for next time
                            self.pdf_widget.texture_cache.dimensions[self.current_page] = (page_width, page_height)
                        except:
                            # Use default aspect ratio if PDF access fails
                            page_width, page_height = 595, 842  # A4 default
                    
                    # Set texture immediately for instant display
                    self.pdf_widget.set_page_texture(texture, page_width, page_height)
                    self.pdf_widget.set_current_page(self.current_page)
                    
                    # Update window title
                    if self.pdf_path:
                        filename = os.path.basename(self.pdf_path)
                        self.setWindowTitle(f"PDF Viewer - {filename} - Page {self.current_page + 1} of {self.total_pages}")
                    
                    # Schedule background texture generation for better quality if needed
                    QTimer.singleShot(10, lambda: self.ensure_high_quality_texture(self.current_page))
                else:
                    # No texture available - generate one in background while showing loading
                    self.pdf_widget.set_loading_state(True)
                    
                    # For single page mode, always use zoom-adjusted quality for best visual quality
                    target_quality = self.pdf_widget.get_zoom_adjusted_quality()
                    if self.renderer:
                        self.renderer.add_page_to_queue(self.current_page, priority=True, quality=target_quality)
                    
                    # Update window title even without texture
                    if self.pdf_path:
                        filename = os.path.basename(self.pdf_path)
                        self.setWindowTitle(f"PDF Viewer - {filename} - Page {self.current_page + 1} of {self.total_pages} (Loading...)")
    
    def ensure_high_quality_texture(self, page_num):
        """Ensure we have a high-quality texture for the given page"""
        if not self.pdf_doc or page_num < 0 or page_num >= self.total_pages:
            return
        
        # Check if we already have a high-quality texture
        texture = self.pdf_widget.texture_cache.get_texture(page_num)
        if texture:
            # We have a texture, but let's make sure it's high quality
            # This is a background operation so it won't block the UI
            target_quality = self.pdf_widget.get_zoom_adjusted_quality()
            if self.renderer:
                self.renderer.add_page_to_queue(page_num, priority=False, quality=target_quality)
    
    def create_left_panel(self):
        """Create the left panel with thumbnails"""
        panel = QWidget()
        panel.setFixedWidth(200)
        panel.setStyleSheet("""
            QWidget { 
                background-color: rgb(38, 38, 38); 
                border: none;
                margin: 0px;
                padding: 0px;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Thumbnails without title
        self.thumbnail_list = QListWidget()
        self.thumbnail_list.setStyleSheet("""
            QListWidget { 
                background-color: rgb(38, 38, 38); 
                border: none; 
                margin: 0px;
                padding: 2px;
            }
            QListWidget::item {
                background-color: rgb(38, 38, 38);
                border: none;
            }
            QListWidget::item:selected {
                background-color: rgb(60, 60, 60);
            }
        """)
        self.thumbnail_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.thumbnail_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.thumbnail_list.setMovement(QListWidget.Movement.Static)
        self.thumbnail_list.setIconSize(QSize(120, 160))  # Thumbnail size
        self.thumbnail_list.setSpacing(20)
        self.thumbnail_list.itemClicked.connect(self.thumbnail_clicked)
        
        # Add hover-based preloading for instant response
        self.thumbnail_list.itemEntered.connect(self._on_thumbnail_hover)
        
        # Connect scroll events to detect when user reaches edges
        scroll_bar = self.thumbnail_list.verticalScrollBar()
        scroll_bar.valueChanged.connect(self._on_thumbnail_scroll)
        
        layout.addWidget(self.thumbnail_list)
        
        return panel
    
    def create_menu_bar(self):
        """Create the application menu bar"""
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
        
        # Grid view toggle in menu
        grid_view_menu_action = QAction("Grid View", self)
        grid_view_menu_action.setShortcut(QKeySequence("G"))
        grid_view_menu_action.setCheckable(True)
        grid_view_menu_action.triggered.connect(self.toggle_grid_view_from_menu)
        view_menu.addAction(grid_view_menu_action)
        # Keep reference to sync with toolbar action
        self.grid_view_menu_action = grid_view_menu_action
        
        # Grid size submenu
        grid_size_menu = view_menu.addMenu("Grid Size")
        
        size_2x2_action = QAction("2x2", self)
        size_2x2_action.setShortcut(QKeySequence("1"))
        size_2x2_action.triggered.connect(lambda: self.set_grid_size("2x2"))
        grid_size_menu.addAction(size_2x2_action)
        
        size_3x3_action = QAction("3x3", self)
        size_3x3_action.setShortcut(QKeySequence("2"))
        size_3x3_action.triggered.connect(lambda: self.set_grid_size("3x3"))
        grid_size_menu.addAction(size_3x3_action)

        size_5x1_action = QAction("5x1", self)
        size_5x1_action.setShortcut(QKeySequence("4"))
        size_5x1_action.triggered.connect(lambda: self.set_grid_size("5x1"))
        grid_size_menu.addAction(size_5x1_action)
        
        size_5x2_action = QAction("5x2", self)
        size_5x2_action.setShortcut(QKeySequence("5"))
        size_5x2_action.triggered.connect(lambda: self.set_grid_size("5x2"))
        grid_size_menu.addAction(size_5x2_action)
        
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
        self.grid_view_action.setToolTip("Toggle Grid View (G)")
        toolbar.addAction(self.grid_view_action)

        toolbar.addSeparator()

        # Grid size selector for single-row view
        self.grid_size_combo = QComboBox()
        self.grid_size_combo.addItems(["2x2", "3x1", "3x2", "5x1"])
        self.grid_size_combo.currentTextChanged.connect(self.change_grid_size)
        self.grid_size_combo.setEnabled(False)
        self.grid_size_combo.setMaxVisibleItems(10)
        self.grid_size_combo.setToolTip("Grid Size (1-4 keys or Tab to cycle)")
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
        zoom_in_action.triggered.connect(self.zoom_in_clicked)
        toolbar.addAction(zoom_in_action)

        zoom_out_action = QAction("🔍-", self)
        zoom_out_action.triggered.connect(self.zoom_out_clicked)
        toolbar.addAction(zoom_out_action)

        reset_zoom_action = QAction("⌂", self)
        reset_zoom_action.triggered.connect(self.reset_zoom_clicked)
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
        
        # Grid view toggle shortcut
        QShortcut(QKeySequence("G"), self, self.toggle_grid_view_shortcut)
        
        # Grid size shortcuts
        QShortcut(QKeySequence("1"), self, lambda: self.set_grid_size("2x2"))
        QShortcut(QKeySequence("2"), self, lambda: self.set_grid_size("3x1"))
        QShortcut(QKeySequence("3"), self, lambda: self.set_grid_size("3x2"))
        QShortcut(QKeySequence("4"), self, lambda: self.set_grid_size("5x1"))
        QShortcut(QKeySequence("Tab"), self, self.cycle_grid_size)
    
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
- Adaptive: Balances uniform sizing with aspect ratio proportionality
- Uniform: All grid cells have the same dimensions
- Proportional: Row heights strictly follow page aspect ratios

Settings:
- Gap ratio: {self.pdf_widget.grid_gap_ratio:.1%} of widget size
- Aspect weight: {self.pdf_widget.grid_aspect_weight:.0%} (adaptive mode only)
- Gap range: {int(self.pdf_widget.grid_min_gap)}px to {int(self.pdf_widget.grid_max_gap)}px

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
    
    def _rebuild_grid_size_combo(self, current_text=None):
        """
        Safely clears and rebuilds the grid size combo box items.
        This is the most robust way to prevent item duplication after state changes.
        """
        # Disconnect signal to prevent unwanted triggers during rebuild
        try:
            self.grid_size_combo.currentTextChanged.disconnect(self.change_grid_size)
        except TypeError:
            # Signal was not connected, which is fine
            pass

        self.grid_size_combo.clear()
        self.grid_size_combo.addItems(["2x2", "3x1", "3x2", "5x1"])
        
        if current_text and current_text not in ["4x4", "3x3", "5x2"]:  # Skip 4x4, old 3x3, and 5x2 if they were selected
            index = self.grid_size_combo.findText(current_text)
            if index >= 0:
                self.grid_size_combo.setCurrentIndex(index)
        elif current_text == "3x3":
            # Map old 3x3 to new 3x2
            index = self.grid_size_combo.findText("3x2")
            if index >= 0:
                self.grid_size_combo.setCurrentIndex(index)
        elif current_text == "5x2":
            # Map 5x2 to 5x1 as fallback
            index = self.grid_size_combo.findText("5x1")
            if index >= 0:
                self.grid_size_combo.setCurrentIndex(index)
        
        # Reconnect the signal
        self.grid_size_combo.currentTextChanged.connect(self.change_grid_size)

    def restore_combo_box_state(self, grid_size, enabled):
        """Restore combo box state after fullscreen transition"""
        try:
            self._rebuild_grid_size_combo(grid_size)
            self.grid_size_combo.setEnabled(enabled)
            
            # Show helpful message
            if self.isFullScreen():
                self.status_bar.showMessage("Fullscreen: Use keys 1-4 or Tab to change grid size", 3000)
        except Exception as e:
            print(f"Error restoring combo box state: {e}")

    def toggle_fullscreen(self, checked=None):
        """Toggle fullscreen mode"""
        # Save combo box state before fullscreen transition
        current_grid_size = self.grid_size_combo.currentText()
        grid_enabled = self.grid_size_combo.isEnabled()
        
        if checked is None:
            # Called from menu or keyboard shortcut - toggle based on current state
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            # Called from checkable toolbar action - use the checked state
            if checked:
                self.showFullScreen()
            else:
                self.showNormal()
        
        # Restore combo box state after fullscreen transition
        QTimer.singleShot(100, lambda: self.restore_combo_box_state(current_grid_size, grid_enabled))
    
    def dragEnterEvent(self, event):
        """Handle drag enter events"""
        print(f"Drag enter detected with {len(event.mimeData().urls())} URLs")
        if event.mimeData().hasUrls():
            # Check if any of the dragged files is a PDF
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                print(f"Checking file: {file_path}")
                if url.isLocalFile() and file_path.lower().endswith('.pdf'):
                    print("PDF file detected - accepting drag")
                    event.acceptProposedAction()
                    return
        print("No PDF files found - ignoring drag")
        event.ignore()
    
    def dropEvent(self, event):
        """Handle drop events"""
        print("Drop event received")
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    print(f"Processing dropped file: {file_path}")
                    if file_path.lower().endswith('.pdf'):
                        print(f"Loading PDF: {file_path}")
                        self.load_pdf(file_path)
                        event.acceptProposedAction()
                        self.status_bar.showMessage(f"Dropped: {os.path.basename(file_path)}", 3000)
                        return
        print("Drop event ignored")
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
    
    def load_pdf(self, pdf_path: str, quality: float = 5.0):
        try:
            # Clean up previous document
            if hasattr(self, 'pdf_doc') and self.pdf_doc is not None:
                try:
                    self.pdf_doc.close()
                except:
                    pass  # Ignore errors if already closed
                self.pdf_doc = None
                
            if hasattr(self, 'renderer') and self.renderer:
                self.renderer.stop()
                self.renderer.wait()
                self.renderer = None
                
            if hasattr(self, 'thumbnail_worker') and self.thumbnail_worker:
                # Properly disconnect signals before stopping
                try:
                    self.thumbnail_worker.thumbnailReady.disconnect()
                    self.thumbnail_worker.finished.disconnect()
                except:
                    pass
                self.thumbnail_worker.stop()
                self.thumbnail_worker.wait()
                self.thumbnail_worker = None
            
            # Clear thumbnail list immediately when loading new PDF
            if hasattr(self, 'thumbnail_list') and self.thumbnail_list:
                self.thumbnail_list.clear()
                print("Cleared thumbnail list for new PDF")
            
            # Reset thumbnail-related state variables
            if hasattr(self, '_current_thumb_range'):
                delattr(self, '_current_thumb_range')
            if hasattr(self, '_selective_thumbnails_enabled'):
                delattr(self, '_selective_thumbnails_enabled')
            
            # Clear texture cache and grid textures when loading new document
            try:
                if hasattr(self.pdf_widget, 'texture_cache') and self.pdf_widget.texture_cache:
                    if hasattr(self.pdf_widget.texture_cache, 'clear_cache'):
                        self.pdf_widget.texture_cache.clear_cache()
                    elif hasattr(self.pdf_widget.texture_cache, 'clear'):
                        self.pdf_widget.texture_cache.clear()
                    else:
                        # Recreate texture cache if methods are missing
                        print("Warning: Recreating texture cache due to missing methods")
                        self.pdf_widget.texture_cache = GPUTextureCache()
            except Exception as e:
                print(f"Warning: Could not clear texture cache: {e}")
                # Fallback: recreate texture cache
                self.pdf_widget.texture_cache = GPUTextureCache()
            
            # CRITICAL: Reset grid mode state when loading new PDF
            # The grid mode should be determined by the UI toggle state, not previous PDF state
            grid_view_checked = self.grid_view_action.isChecked() if hasattr(self, 'grid_view_action') else False
            self.pdf_widget.grid_mode = grid_view_checked
            
            # Reset all grid-related state variables
            if hasattr(self.pdf_widget, 'is_temp_zoomed'):
                self.pdf_widget.is_temp_zoomed = False
            if hasattr(self.pdf_widget, 'temp_zoom_factor'):
                self.pdf_widget.temp_zoom_factor = 1.0
            if hasattr(self.pdf_widget, 'temp_pan_x'):
                self.pdf_widget.temp_pan_x = 0.0
            if hasattr(self.pdf_widget, 'temp_pan_y'):
                self.pdf_widget.temp_pan_y = 0.0
            
            print(f"PDF loading - Grid mode set to: {self.pdf_widget.grid_mode} (from UI toggle)")
            
            try:
                if hasattr(self.pdf_widget, 'grid_textures'):
                    self.pdf_widget.grid_textures.clear()
            except Exception as e:
                print(f"Warning: Could not clear grid textures: {e}")
                
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
            
            # Start renderer FIRST for immediate processing
            self.renderer.start()
            
            # Render first page immediately for ultra-fast display
            self.render_current_page()
            
            # Update background color for current theme
            self.pdf_widget.update_background_color()
            
            # ALWAYS generate thumbnails for new PDF - override grid mode check
            QTimer.singleShot(100, lambda: self.generate_thumbnails_deferred(force_new_pdf=True))  # Reduced from 200ms to 100ms
            
            # Show immediate status
            self.status_bar.showMessage(f"Loading: {os.path.basename(pdf_path)} ({self.total_pages} pages)")
            
        except Exception as e:
            tb = traceback.format_exc()
            # Show the exception and include a copyable traceback for debugging
            QMessageBox.critical(self, "Error", f"Failed to load PDF:\n{str(e)}\n\nTraceback:\n{tb}")
    
    def generate_thumbnails_deferred(self, force_new_pdf=False):
        """Generate thumbnails after main view is ready"""
        try:
            # Check if we're in grid mode - if so, minimize thumbnail generation
            # BUT override this check if we're loading a new PDF (force_new_pdf=True)
            if not force_new_pdf and getattr(self.pdf_widget, 'grid_mode', False):
                print("Grid mode active - skipping heavy thumbnail generation to prioritize grid performance")
                return
            
            # If we're forcing generation for new PDF, show status message
            if force_new_pdf:
                print("New PDF loaded - forcing thumbnail generation even in grid mode")
            
            # Force complete thumbnail regeneration for new PDF
            self._selective_thumbnails_enabled = False
            
            # Generate thumbnails with force clear to ensure old thumbnails are removed
            self.generate_thumbnails(force_clear=True)
            
            # Re-enable selective loading after initial generation
            self._selective_thumbnails_enabled = True
            
            # Initialize tracking variables for selective loading
            if not hasattr(self, '_current_thumb_range'):
                self._current_thumb_range = (0, min(30, self.total_pages - 1))  # Initial range
            
            # Show final loaded status
            file_name = os.path.basename(self.pdf_path)
            zoom_percent = int(self.pdf_widget.zoom_factor * 100)
            quality = self.pdf_widget.get_zoom_adjusted_quality()
            self.status_bar.showMessage(f"Loaded: {file_name} | Page 1/{self.total_pages} | Zoom: {zoom_percent}% | Quality: {quality:.1f}")
            
            # Update PDF widget status
            self.pdf_widget.update_status()
            QTimer.singleShot(2000, lambda: self.status_bar.showMessage("Ready"))
            
            # Start aggressive preloading after initial loading is complete
            QTimer.singleShot(3000, self.start_aggressive_preloading)
            
        except Exception as e:
            print(f"Thumbnail generation error: {e}")
            
        except Exception as e:
            tb = traceback.format_exc()
            # Show the exception and include a copyable traceback for debugging
            QMessageBox.critical(self, "Error", f"Failed to load PDF:\n{str(e)}\n\nTraceback:\n{tb}")
    
    def render_current_page(self):
        if self.pdf_doc and 0 <= self.current_page < self.total_pages:
            if self.pdf_widget.grid_mode:
                self.render_grid_pages()
            else:
                # Single page mode - implement immediate display for maximum responsiveness
                texture = self.pdf_widget.texture_cache.get_texture(self.current_page)
                if texture:
                    # We have a texture, set it immediately without blocking operations
                    cached_dims = self.pdf_widget.texture_cache.get_dimensions(self.current_page)
                    if cached_dims:
                        page_width, page_height = cached_dims
                    else:
                        # Fallback to PDF dimensions but don't block the main thread
                        try:
                            page = self.pdf_doc[self.current_page]
                            page_width = page.rect.width
                            page_height = page.rect.height
                            # Cache these dimensions for next time
                            self.pdf_widget.texture_cache.dimensions[self.current_page] = (page_width, page_height)
                        except:
                            # Use default aspect ratio if PDF access fails
                            page_width, page_height = 595, 842  # A4 default
                    
                    # Set texture immediately for instant display
                    self.pdf_widget.set_page_texture(texture, page_width, page_height)
                    
                    # Check if we need better quality (do this in background)
                    QTimer.singleShot(50, lambda: self._check_and_queue_better_quality())
                else:
                    # No texture exists - start immediate + background rendering
                    # Clear existing texture to ensure fresh start
                    self.pdf_widget.set_page_texture(None)
                    
                    if self.renderer:
                        # 🚀 ULTRA-FAST: Queue ultra-low quality for instant display (same speed as grid)
                        immediate_quality = 1.5  # Ultra-fast, same as grid mode for consistency
                        self.renderer.add_page_to_queue(self.current_page, priority=True, quality=immediate_quality)
                        
                        # Queue better readable quality very quickly (50ms delay)
                        readable_quality = self.pdf_widget.get_immediate_quality()
                        QTimer.singleShot(50, lambda: self.renderer.add_page_to_queue(self.current_page, priority=True, quality=readable_quality))
                        
                        # Queue final high quality in background (lower priority, delayed)
                        if self.pdf_widget.zoom_factor > self.pdf_widget.quality_zoom_threshold:
                            QTimer.singleShot(200, lambda: self._queue_high_quality_if_needed())
                
                # Always trigger a repaint
                self.pdf_widget.update()
    
    def _check_and_queue_better_quality(self):
        """Check if better quality is needed and queue it (non-blocking)"""
        if not self.renderer or not self.pdf_doc:
            return
            
        if self.pdf_widget.zoom_factor > self.pdf_widget.quality_zoom_threshold:
            final_quality = self.pdf_widget.get_zoom_adjusted_quality()
            current_best_quality = self.pdf_widget.texture_cache.get_best_quality_for_page(self.current_page)
            if final_quality > current_best_quality + 0.5:  # Higher threshold to reduce unnecessary renders
                self.renderer.add_page_to_queue(self.current_page, priority=False, quality=final_quality)
    
    def _queue_high_quality_if_needed(self):
        """Queue high quality version if zoom is high (background operation)"""
        if not self.renderer or not self.pdf_doc:
            return
            
        if self.pdf_widget.zoom_factor > self.pdf_widget.quality_zoom_threshold:
            final_quality = self.pdf_widget.get_zoom_adjusted_quality()
            self.renderer.add_page_to_queue(self.current_page, priority=False, quality=final_quality)
    
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
        """Non-blocking grid pages renderer - triggers async background loading only"""
        if not self.pdf_doc:
            return
        
        # NEVER BLOCK - just ensure grid layout is computed and trigger background loading
        pages_needed = self.pdf_widget.grid_cols * self.pdf_widget.grid_rows
        
        # Quick grid layout update if needed (non-blocking)
        self.pdf_widget.compute_grid_layout()
        
        # Queue pages for background loading without blocking the UI thread
        QTimer.singleShot(0, lambda: self._queue_grid_pages_async())
        
        # IMMEDIATE UPDATE - don't wait for any textures
        self.pdf_widget.update()

    def _queue_grid_pages_async(self):
        """Queue grid pages for background loading - called asynchronously"""
        if not self.pdf_doc or not hasattr(self, 'renderer') or not self.renderer:
            return
            
        pages_needed = self.pdf_widget.grid_cols * self.pdf_widget.grid_rows
        
        # 🚀 GRID ZOOM FIX: Use zoom-adaptive quality for grid mode too!
        zoom_quality = self.pdf_widget.get_zoom_adjusted_quality()
        
        # Check interaction state to adjust quality
        is_interacting = (getattr(self.pdf_widget, '_interaction_mode', False) or 
                         getattr(self.pdf_widget, '_fast_zoom_mode', False) or
                         getattr(self.pdf_widget, 'is_temp_zoomed', False))
        
        if is_interacting:
            # During interaction, use lower quality but still zoom-aware
            base_quality = max(2.0, zoom_quality * 0.6)
        else:
            # When not interacting, use good zoom-adaptive quality
            base_quality = max(3.0, zoom_quality * 0.8)
        
        # Queue all visible pages with appropriate quality
        for i in range(pages_needed):
            page_num = self.current_page + i
            if page_num >= self.total_pages:
                break
            
            # Check if we already have adequate quality for current zoom level
            existing_quality = self.pdf_widget.texture_cache.get_best_quality_for_page(page_num)
            target_quality = base_quality
            
            # Always upgrade if current quality is significantly lower than needed
            if not existing_quality or existing_quality < target_quality * 0.7:
                # Priority for first 4 pages, background for rest
                priority = i < 4
                
                # Stagger background loading to prevent UI blocking
                if priority:
                    self.renderer.add_page_to_queue(page_num, priority=True, quality=target_quality)
                else:
                    delay = i * 30  # Reduced stagger for faster loading
                    QTimer.singleShot(delay, lambda p=page_num, q=target_quality: (
                        self.renderer.add_page_to_queue(p, priority=False, quality=q)
                        if hasattr(self, 'renderer') and self.renderer else None
                    ))

    def generate_thumbnails(self, around_page=None, radius=None, preserve_scroll=True, force_clear=False):
        """Generate thumbnails selectively around current page or specified page
        
        Args:
            around_page: Page number to center loading around (uses current_page if None)
            radius: Number of pages to load before/after the center page (auto-calculated if None)
            preserve_scroll: Whether to preserve current scroll position in thumbnail list
            force_clear: Force clearing thumbnail list regardless of selective loading settings
        """
        if not self.pdf_doc:
            return
        
        # Adaptive radius based on grid size and PDF complexity
        if radius is None:
            max_grid_pages = 9  # Maximum possible grid size (3x3, since 4x4 was removed)
            
            # Check if we're in grid mode - drastically reduce thumbnails to prioritize grid performance
            if getattr(self.pdf_widget, 'grid_mode', False):
                # In grid mode, only load minimal thumbnails to reduce resource competition
                radius = max_grid_pages + 2  # Just a few beyond grid for navigation
                print(f"Grid mode active - using minimal thumbnail radius: {radius}")
            else:
                # Calculate file size for adaptive loading in single-page mode
                file_size_mb = 0
                try:
                    if hasattr(self, 'pdf_path') and os.path.exists(self.pdf_path):
                        file_size_mb = os.path.getsize(self.pdf_path) / (1024 * 1024)
                except:
                    pass
                
                # Adaptive radius: smaller for large PDFs, larger for small PDFs
                if file_size_mb > 30:  # Large PDF
                    radius = max_grid_pages + 3  # Only load a few extra beyond grid
                    print(f"Large PDF ({file_size_mb:.1f}MB) - using minimal thumbnail radius: {radius}")
                elif file_size_mb > 10:  # Medium PDF
                    radius = max_grid_pages + 8  # Moderate lookahead
                else:  # Small PDF
                    radius = 15  # Original radius for small files
        
        # If this is first load or user explicitly requested all pages, load everything
        if not hasattr(self, '_selective_thumbnails_enabled'):
            self._selective_thumbnails_enabled = self.total_pages > 50  # Enable for large docs
        
        # Force clear thumbnails if requested (e.g., when loading new PDF)
        if force_clear:
            self.thumbnail_list.clear()
            preserve_scroll = False  # No point preserving scroll when force clearing
            print("Force cleared thumbnails for new PDF")
        
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
            # IMMEDIATE RESPONSIVENESS: Create placeholder thumbnails instantly
            if self._selective_thumbnails_enabled or force_clear:
                self._create_placeholder_thumbnails(start_page, end_page)
            
            # Start a worker thread to generate real thumbnails and emit signals safely
            if self.thumbnail_worker:
                # Properly disconnect signals before stopping
                try:
                    self.thumbnail_worker.thumbnailReady.disconnect()
                    self.thumbnail_worker.finished.disconnect()
                except:
                    pass
                self.thumbnail_worker.stop()
                self.thumbnail_worker.wait()
            
            # Create worker with limited page range and GPU widget reference
            pages_to_load = list(range(start_page, end_page + 1))
            self.thumbnail_worker = ThumbnailWorker(
                self.pdf_doc, 
                page_list=pages_to_load, 
                limit=len(pages_to_load), 
                parent=self,
                gpu_widget=self.pdf_widget  # Pass GPU widget for texture cache access
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
    
    def _create_placeholder_thumbnails(self, start_page, end_page):
        """Create instant placeholder thumbnails for immediate responsiveness"""
        try:
            # Create a simple, clean placeholder image without numbers
            placeholder_img = QImage(120, 160, QImage.Format.Format_RGB32)
            placeholder_img.fill(QColor(60, 60, 60))  # Dark gray placeholder - clean and simple
            
            for page_num in range(start_page, end_page + 1):
                # Create clean placeholder for each page (no page numbers)
                page_placeholder = placeholder_img.copy()
                
                # Create thumbnail item with clean placeholder
                pixmap = QPixmap.fromImage(page_placeholder)
                qt_icon = QIcon(pixmap)
                item = QListWidgetItem(qt_icon, f"{page_num + 1}")
                item.setData(Qt.ItemDataRole.UserRole, page_num)
                
                # Mark as placeholder for later replacement
                item.setData(Qt.ItemDataRole.UserRole + 1, "placeholder")
                
                self.thumbnail_list.addItem(item)
            
            print(f"Created {end_page - start_page + 1} clean placeholder thumbnails instantly")
            
        except Exception as e:
            print(f"Error creating placeholder thumbnails: {e}")
    
    def _load_additional_thumbnails(self, page_list):
        """Load additional thumbnails without clearing existing ones"""
        try:
            if self.thumbnail_worker:
                # Properly disconnect signals before stopping
                try:
                    self.thumbnail_worker.thumbnailReady.disconnect()
                    self.thumbnail_worker.finished.disconnect()
                except:
                    pass
                self.thumbnail_worker.stop()
                self.thumbnail_worker.wait()
            
            self.thumbnail_worker = ThumbnailWorker(
                self.pdf_doc, 
                page_list=page_list, 
                limit=len(page_list), 
                parent=self,
                gpu_widget=self.pdf_widget  # Pass GPU widget for texture cache access
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
            # OPTIMIZATION: Look for existing placeholder to replace
            existing_item = None
            for i in range(self.thumbnail_list.count()):
                item_at_i = self.thumbnail_list.item(i)
                if item_at_i and item_at_i.data(Qt.ItemDataRole.UserRole) == page_num:
                    existing_item = item_at_i
                    break
            
            if isinstance(img_or_icon, QImage):
                qimage = img_or_icon

                thumb = qimage.scaled(120, 160, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                pixmap = QPixmap.fromImage(thumb)
                qt_icon = QIcon(pixmap)
                
                if existing_item:
                    # Replace existing placeholder or outdated thumbnail
                    existing_item.setIcon(qt_icon)
                    existing_item.setData(Qt.ItemDataRole.UserRole + 1, "real")  # Mark as real thumbnail
                    return
                else:
                    # Create new item
                    item = QListWidgetItem(qt_icon, f"{page_num + 1}")
                    item.setData(Qt.ItemDataRole.UserRole, page_num)
                    item.setData(Qt.ItemDataRole.UserRole + 1, "real")  # Mark as real thumbnail
                
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
            load_above_item.setBackground(QColor(38, 38, 38))
            load_above_item.setForeground(QColor(200, 200, 200))
            self.thumbnail_list.insertItem(0, load_above_item)
        
        # Add "Load More Below" indicator if there are pages after current range
        if end_page < self.total_pages - 1:
            load_below_item = QListWidgetItem("⬇️ Load Later Pages")
            load_below_item.setData(Qt.ItemDataRole.UserRole, -2)  # Special marker
            load_below_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            load_below_item.setBackground(QColor(38, 38, 38))
            load_below_item.setForeground(QColor(200, 200, 200))
            self.thumbnail_list.addItem(load_below_item)
    
    def thumbnail_clicked(self, item):
        """Handle thumbnail click to navigate to page - INSTANT UI RESPONSE"""
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
            
            # Handle regular page navigation - IMMEDIATE UI RESPONSE
            if page_num is not None and 0 <= page_num < self.total_pages:
                if self.pdf_widget.grid_mode:
                    # Grid mode: INSTANT navigation with immediate UI update
                    pages_needed = self.pdf_widget.grid_cols * self.pdf_widget.grid_rows
                    start_page = getattr(self, 'current_page', 0)

                    if not (start_page <= page_num < start_page + pages_needed):
                        # INSTANT UI UPDATE: Switch grid immediately, queue rendering after
                        new_start = min(max(0, page_num), max(0, self.total_pages - pages_needed))
                        self.current_page = new_start
                        self.update_page_label()

                        # 🚀 INSTANT LAYOUT: Force cache invalidation and trigger async layout
                        self.pdf_widget._grid_cached_size = (0, 0, 0, 0)
                        # Compute layout asynchronously to avoid UI freeze
                        QTimer.singleShot(0, lambda: (
                            self.pdf_widget.compute_grid_layout(),
                            self.pdf_widget.update()
                        ))

                        # DEFERRED RENDERING: Queue texture loading after UI update
                        self._queue_grid_textures_deferred(new_start, pages_needed, page_num)
                    else:
                        # Page already visible - INSTANT zoom without rendering delay
                        self.pdf_widget.zoom_to_grid_page(page_num)
                        
                        # 🔍 QUALITY CHECK: Ensure texture quality matches zoom level
                        effective_zoom = self.pdf_widget.zoom_factor
                        if hasattr(self.pdf_widget, 'is_temp_zoomed') and self.pdf_widget.is_temp_zoomed:
                            effective_zoom *= getattr(self.pdf_widget, 'temp_zoom_factor', 1.0)
                        
                        if effective_zoom > self.pdf_widget.quality_zoom_threshold:
                            current_quality = self.pdf_widget.texture_cache.get_best_quality_for_page(page_num)
                            target_quality = self.pdf_widget.get_zoom_adjusted_quality()
                            
                            if target_quality > current_quality + 0.5:  # Same threshold as quality check
                                print(f"🔍 Grid click quality upgrade: page {page_num}, from {current_quality:.1f} to {target_quality:.1f} (zoom: {effective_zoom:.1f}x)")
                                if self.renderer:
                                    self.renderer.add_page_to_queue(page_num, priority=True, quality=target_quality)
                        
                        # DEFERRED TEXTURE CHECK: Ensure texture exists
                        if page_num not in self.pdf_widget.grid_textures:
                            self._queue_single_texture_deferred(page_num)
                else:
                    # Single-page mode: Use the simplified, more stable go_to_page
                    self.go_to_page(page_num)

    def _queue_grid_textures_deferred(self, start_page, pages_needed, focus_page):
        """Queue grid textures for background loading after UI update - DEFERRED 10ms"""
        def queue_textures():
            if not self.renderer:
                return
            try:
                # 🔍 GRID QUALITY: Use zoom-aware quality for all grid pages
                effective_zoom = self.pdf_widget.zoom_factor
                if hasattr(self.pdf_widget, 'is_temp_zoomed') and self.pdf_widget.is_temp_zoomed:
                    effective_zoom *= getattr(self.pdf_widget, 'temp_zoom_factor', 1.0)
                
                if effective_zoom > self.pdf_widget.quality_zoom_threshold:
                    # If zoomed, use high quality for all visible pages
                    base_quality = self.pdf_widget.get_zoom_adjusted_quality()
                    focus_quality = base_quality
                else:
                    # Normal zoom levels
                    base_quality = 2.5  # Better base quality than before
                    focus_quality = 3.0  # Focus page gets extra quality
                
                # Priority loading for visible pages
                for i in range(pages_needed):
                    page_num = start_page + i
                    if page_num >= self.total_pages:
                        break
                    
                    # Focus page gets best quality, others get base quality
                    quality = focus_quality if page_num == focus_page else base_quality
                    priority = i < 4  # First 4 pages get priority
                    
                    self.renderer.add_page_to_queue(page_num, priority=priority, quality=quality)
                    
                # Focus page gets special treatment for zoom
                if focus_page >= start_page and focus_page < start_page + pages_needed:
                    # Defer zoom after initial textures
                    QTimer.singleShot(50, lambda: self.pdf_widget.zoom_to_grid_page(focus_page))
            except:
                pass
        
        # CRITICAL: 10ms delay allows UI to update first
        QTimer.singleShot(10, queue_textures)

    def _queue_single_texture_deferred(self, page_num):
        """Queue single texture for background loading - DEFERRED 10ms with zoom-aware quality"""
        def queue_texture():
            if self.renderer:
                try:
                    # 🔍 GRID ZOOM-AWARE: Request appropriate quality based on current zoom/temp zoom
                    effective_zoom = self.pdf_widget.zoom_factor
                    if hasattr(self.pdf_widget, 'is_temp_zoomed') and self.pdf_widget.is_temp_zoomed:
                        effective_zoom *= getattr(self.pdf_widget, 'temp_zoom_factor', 1.0)
                    
                    if effective_zoom > self.pdf_widget.quality_zoom_threshold:
                        # If zoomed, request high quality immediately
                        quality = self.pdf_widget.get_zoom_adjusted_quality()
                        print(f"🔍 Grid thumbnail quality upgrade: page {page_num}, quality {quality:.1f} (zoom: {effective_zoom:.1f}x)")
                    else:
                        # Normal zoom, use standard quality
                        quality = self.pdf_widget.get_immediate_quality()
                    
                    self.renderer.add_page_to_queue(page_num, priority=True, quality=quality)
                except:
                    pass
        
        # CRITICAL: 10ms delay allows UI to update first
        QTimer.singleShot(10, queue_texture)
    
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
        
        # Check if scrolled near top or bottom (within 15% of edges for earlier detection)
        near_top = value <= max_value * 0.15 and value < 100
        near_bottom = value >= max_value * 0.85 and value > max_value - 100  # Earlier detection
        
        if near_top or near_bottom:
            self._edge_scroll_direction = 'up' if near_top else 'down'
            self._scroll_load_timer.start(150)  # Reduced delay from 300ms to 150ms for more responsive loading
    
    def _handle_edge_scroll(self):
        """Handle loading more thumbnails when scrolled to edge - OPTIMIZED for smooth scrolling"""
        if not hasattr(self, '_current_thumb_range') or not hasattr(self, '_edge_scroll_direction'):
            return
        
        start_page, end_page = self._current_thumb_range
        
        if self._edge_scroll_direction == 'up' and start_page > 0:
            # Load earlier pages - SMALL BATCHES for smooth performance
            new_start = max(0, start_page - 8)  # Reduced from 15 to 8 pages
            center_page = (new_start + end_page) // 2
            self.generate_thumbnails(around_page=center_page, radius=12, preserve_scroll=True)  # Reduced from 20 to 12
            
        elif self._edge_scroll_direction == 'down' and end_page < self.total_pages - 1:
            # Load later pages - SMALL BATCHES for smooth performance
            new_end = min(self.total_pages - 1, end_page + 8)  # Reduced from 15 to 8 pages
            center_page = (start_page + new_end) // 2
            self.generate_thumbnails(around_page=center_page, radius=12, preserve_scroll=True)  # Reduced from 20 to 12
    
    def go_to_page(self, page_num):
        """Navigate to a specific page with the safest possible rendering path."""
        if not self.pdf_doc or not (0 <= page_num < self.total_pages):
            return
            
        # Update page number and label immediately
        self.current_page = page_num
        self.update_page_label()
        
        # Always clear any cached texture binding
        if hasattr(self, '_last_bound_texture'):
            delattr(self, '_last_bound_texture')

        # Reset GPU flags
        self._skip_heavy_processing = False
        self._ultra_fast_mode = False
        self._active_panning = False

        # Try to get an existing texture
        texture = self.pdf_widget.texture_cache.get_texture(self.current_page)
        dims = self.pdf_widget.texture_cache.get_dimensions(self.current_page)
        if texture and texture.isCreated() and dims:
            self.pdf_widget.set_page_texture(texture, dims[0], dims[1])
            
            # 🔍 QUALITY CHECK: If zoomed and texture quality is too low, request upgrade
            if self.pdf_widget.zoom_factor > self.pdf_widget.quality_zoom_threshold:
                current_quality = self.pdf_widget.texture_cache.get_best_quality_for_page(self.current_page)
                target_quality = self.pdf_widget.get_zoom_adjusted_quality()
                
                if target_quality > current_quality + 0.5:  # Same threshold as quality check
                    print(f"🔍 Navigation quality upgrade: page {page_num}, from {current_quality:.1f} to {target_quality:.1f} (zoom: {self.pdf_widget.zoom_factor:.1f}x)")
                    if hasattr(self, 'renderer') and self.renderer:
                        self.renderer.add_page_to_queue(page_num, priority=True, quality=target_quality)
        else:
            # Show a placeholder (gray) if no texture is available
            self.pdf_widget.set_page_texture(None)

        # Always update the widget to force a redraw
        self.pdf_widget.update()

        # If no texture, queue a render for this page
        if (not texture or not texture.isCreated()) and hasattr(self, 'renderer') and self.renderer:
            # 🔍 ZOOM-AWARE NAVIGATION: Request appropriate quality based on current zoom
            if self.pdf_widget.zoom_factor > self.pdf_widget.quality_zoom_threshold:
                # If zoomed in, request high quality immediately to avoid zoom out/in cycle
                target_quality = self.pdf_widget.get_zoom_adjusted_quality()
                print(f"🔍 Zoomed navigation: page {page_num}, requesting quality {target_quality:.1f} (zoom: {self.pdf_widget.zoom_factor:.1f}x)")
                self.renderer.add_page_to_queue(page_num, priority=True, quality=target_quality)
            else:
                # Normal zoom, use standard quality
                self.renderer.add_page_to_queue(page_num, priority=True, quality=2.5)
        
    def _background_page_operations(self):
        """Perform background operations after page navigation (non-blocking)"""
        try:
            # Clean up distant textures to save memory
            self.pdf_widget.cleanup_distant_textures()
            
            # AGGRESSIVE PRELOADING for much larger GPU cache
            if self.renderer:
                # Preload many more adjacent pages with the increased VRAM
                current = self.current_page
                
                # Preload 30 pages ahead and 30 pages behind (was only 1 each)
                for offset in range(-30, 31):
                    page_to_preload = current + offset
                    if 0 <= page_to_preload < self.total_pages and page_to_preload != current:
                        # Use progressive quality - closer pages get higher quality
                        distance = abs(offset)
                        if distance <= 5:
                            quality = 3.0  # High quality for very close pages
                        elif distance <= 15:
                            quality = 2.0  # Medium quality for close pages
                        else:
                            quality = 1.5  # Fast quality for distant pages
                        
                        self.renderer.add_page_to_queue(page_to_preload, priority=False, quality=quality)
                    
            # Extended thumbnail range for larger cache
            if (hasattr(self, '_selective_thumbnails_enabled') and self._selective_thumbnails_enabled and
                hasattr(self, '_current_thumb_range')):
                start_page, end_page = self._current_thumb_range
                # If navigating near the edge of loaded range, preload much more
                if self.current_page <= start_page + 10 or self.current_page >= end_page - 10:
                    # Preload 50 pages around current position (was 15)
                    self.generate_thumbnails(around_page=self.current_page, radius=50, preserve_scroll=True)
                    
        except Exception as e:
            print(f"Background operations error: {e}")
    
    def start_aggressive_preloading(self):
        """Start aggressive preloading to fill GPU VRAM for maximum performance"""
        if not self.pdf_doc or not self.renderer:
            return
        
        # Check current performance - skip aggressive preloading if performance is poor
        performance_level = getattr(self.pdf_widget, '_performance_monitor', {}).get('performance_level', 'high')
        if performance_level == 'low':
            print("🚀 Skipping aggressive preloading due to low performance")
            return
            
        # Preload entire document in background with low priority
        print(f"🚀 Starting aggressive preloading of {self.total_pages} pages...")
        
        for page_num in range(self.total_pages):
            if page_num != self.current_page:  # Skip current page (already loaded)
                # Use distance-based quality
                distance = abs(page_num - self.current_page)
                if distance <= 10:
                    quality = 2.5  # Good quality for close pages
                elif distance <= 50:
                    quality = 2.0  # Medium quality 
                else:
                    quality = 1.5  # Fast quality for distant pages
                
                self.renderer.add_page_to_queue(page_num, priority=False, quality=quality)
                
        print(f"🚀 Queued {self.total_pages} pages for aggressive preloading")
    
    def _on_thumbnail_hover(self, item):
        """Preload page texture when hovering over thumbnail for instant response"""
        if item and self.renderer:
            page_num = item.data(Qt.ItemDataRole.UserRole)
            if page_num is not None and 0 <= page_num < self.total_pages:
                # Check if we already have a texture for this page
                if not self.pdf_widget.texture_cache.get_texture(page_num):
                    # No texture - preload a fast one in background
                    self.renderer.add_page_to_queue(page_num, priority=False, quality=1.5)  # Very low quality for preloading
    
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
        except AttributeError:
            # Fix missing _pending_images attribute
            print("Adding missing _pending_images attribute to GPUPDFWidget")
            self.pdf_widget._pending_images = [(page_num, image, page_width, page_height, quality)]
            self.pdf_widget.update()
        except Exception as e:
            print(f"Error in pending images queue: {e}")
            # Fallback: synchronous path (if something unexpected)
            try:
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
            except AttributeError as e:
                print(f"Warning: Texture cache method missing: {e}")
                # Recreate texture cache if it's missing the add_texture method
                print("Recreating texture cache with missing methods")
                self.pdf_widget.texture_cache = GPUTextureCache()
                # Try again with the new cache
                try:
                    texture = self.pdf_widget.texture_cache.add_texture(page_num, image, page_width, page_height, quality)
                    if page_num == self.current_page:
                        self.pdf_widget.set_page_texture(texture, page_width, page_height)
                except Exception as e2:
                    print(f"Error creating texture after cache recreation: {e2}")
            except Exception as e:
                print(f"Error creating texture: {e}")
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
        
        # Preload adjacent pages for smoother navigation (ultra-fast quality)
        if self.renderer:
            if self.current_page > 0:
                self.renderer.add_page_to_queue(self.current_page - 1, priority=False, quality=1.5)  # Ultra-fast preload
            if self.current_page < self.total_pages - 1:
                self.renderer.add_page_to_queue(self.current_page + 1, priority=False, quality=1.5)  # Ultra-fast preload
    
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
        """Handle zoom slider changes"""
        zoom_factor = value / 100.0
        if self.pdf_widget.grid_mode and self.pdf_widget.is_temp_zoomed:
            # Update temp zoom factor
            self.pdf_widget.temp_zoom_factor = zoom_factor
        else:
            # Update normal zoom factor
            self.pdf_widget.zoom_factor = zoom_factor
        self.pdf_widget.update()
    
    def zoom_in_clicked(self):
        """Handle zoom in button click"""
        if self.pdf_widget.grid_mode and self.pdf_widget.is_temp_zoomed:
            # If in temp zoom mode, use temp zoom
            self.pdf_widget.temp_zoom_factor = min(self.pdf_widget.temp_zoom_factor * 1.2, 10.0)
            self.pdf_widget.update()
        else:
            # Normal zoom
            self.pdf_widget.zoom_in()
    
    def zoom_out_clicked(self):
        """Handle zoom out button click"""
        if self.pdf_widget.grid_mode and self.pdf_widget.is_temp_zoomed:
            # If in temp zoom mode, use temp zoom
            self.pdf_widget.temp_zoom_factor = max(self.pdf_widget.temp_zoom_factor / 1.2, 0.1)
            self.pdf_widget.update()
        else:
            # Normal zoom
            self.pdf_widget.zoom_out()
    
    def reset_zoom_clicked(self):
        """Handle reset zoom/center button click"""
        if self.pdf_widget.grid_mode and self.pdf_widget.is_temp_zoomed:
            # Reset temp zoom
            self.pdf_widget.reset_grid_zoom()
        else:
            # Normal reset
            self.pdf_widget.reset_zoom()
    
    def set_grid_size(self, size_text):
        """Set grid size via keyboard shortcut"""
        if not self.grid_view_action.isChecked():
            # Auto-enable grid view if not already enabled
            self.grid_view_action.setChecked(True)
            self.grid_view_menu_action.setChecked(True)
            self.toggle_grid_view()
        
        # Set the combo box selection
        index = self.grid_size_combo.findText(size_text)
        if index >= 0:
            self.grid_size_combo.setCurrentIndex(index)
            self.change_grid_size(size_text)
            # Show feedback to user
            self.status_bar.showMessage(f"Grid Size: {size_text}", 2000)

    def cycle_grid_size(self):
        """Cycle through available grid sizes"""
        if not self.grid_view_action.isChecked():
            # Auto-enable grid view if not already enabled
            self.grid_view_action.setChecked(True)
            self.grid_view_menu_action.setChecked(True)
            self.toggle_grid_view()
        
        # Ensure combo box has items
        self._rebuild_grid_size_combo(self.grid_size_combo.currentText())
        
        # Get current index and move to next
        current_index = self.grid_size_combo.currentIndex()
        if current_index < 0:  # Handle invalid index
            current_index = 0
        next_index = (current_index + 1) % self.grid_size_combo.count()
        self.grid_size_combo.setCurrentIndex(next_index)
        size_text = self.grid_size_combo.currentText()
        self.change_grid_size(size_text)
        # Show feedback to user
        self.status_bar.showMessage(f"Grid Size: {size_text}", 2000)

    def toggle_grid_view_from_menu(self):
        """Toggle grid view from menu action"""
        # Sync toolbar action with menu action
        checked_state = self.grid_view_menu_action.isChecked()
        self.grid_view_action.setChecked(checked_state)
        # Call the main toggle method
        self.toggle_grid_view()

    def toggle_grid_view_shortcut(self):
        """Toggle grid view via keyboard shortcut"""
        # Toggle the action's checked state for both toolbar and menu
        new_state = not self.grid_view_action.isChecked()
        self.grid_view_action.setChecked(new_state)
        self.grid_view_menu_action.setChecked(new_state)
        # Call the toggle method to update the UI
        self.toggle_grid_view()

    def toggle_grid_view(self):
        """Toggle between single page and grid view"""
        # Ensure both toolbar and menu actions are in sync
        checked_state = self.grid_view_action.isChecked()
        self.grid_view_menu_action.setChecked(checked_state)
        
        if checked_state:
            # SWITCHING TO GRID MODE
            self._rebuild_grid_size_combo(self.grid_size_combo.currentText())
            
            self.grid_size_combo.setEnabled(True)
            self.pdf_widget.grid_mode = True
            
            # Get current text, with fallback if empty
            current_text = self.grid_size_combo.currentText()
            if not current_text or current_text.strip() == "":
                current_text = "2x2"  # Default
                self.grid_size_combo.setCurrentText(current_text)
                
            self.change_grid_size(current_text)
        else:
            # SWITCHING TO SINGLE PAGE MODE - ROBUST TRANSITION
            self.grid_size_combo.setEnabled(False)
            
            # STEP 1: Reset all flags and states immediately
            self.pdf_widget.grid_mode = False
            self.pdf_widget.grid_cols = 1
            self.pdf_widget.grid_rows = 1
            self.pdf_widget._skip_heavy_processing = False
            self.pdf_widget._interaction_mode = False
            self.pdf_widget._active_panning = False
            
            # STEP 2: Reset grid zoom state completely
            self.pdf_widget.is_temp_zoomed = False
            self.pdf_widget.temp_zoom_factor = 1.0
            self.pdf_widget.temp_pan_x = 0.0
            self.pdf_widget.temp_pan_y = 0.0
            
            # STEP 3: Clear grid-related data
            if hasattr(self.pdf_widget, 'grid_textures'):
                self.pdf_widget.grid_textures = {}
            if hasattr(self.pdf_widget, '_grid_cells'):
                self.pdf_widget._grid_cells = []
            
            # STEP 4: Force clear existing texture and trigger immediate re-render
            # This ensures we get fresh single-page quality texture
            if hasattr(self.pdf_widget.texture_cache, 'remove_texture'):
                self.pdf_widget.texture_cache.remove_texture(self.current_page)
            self.pdf_widget.set_page_texture(None)
            
            # STEP 5: Force immediate repaint to clear grid visuals
            self.pdf_widget.update()
            
            # STEP 6: Start fresh single-page rendering with proper priorities
            if self.renderer and self.pdf_doc:
                # Queue immediate low-quality render for instant feedback
                immediate_quality = self.pdf_widget.get_immediate_quality()
                self.renderer.add_page_to_queue(self.current_page, priority=True, quality=immediate_quality)
                
                # Queue high-quality render for final result - ensure it matches grid quality
                final_quality = self.pdf_widget.get_zoom_adjusted_quality()
                # Boost quality slightly for single page mode to match grid experience
                if final_quality < 3.0:
                    final_quality = max(final_quality, 3.0)  # Minimum high quality for single page
                QTimer.singleShot(100, lambda: self.renderer.add_page_to_queue(self.current_page, priority=True, quality=final_quality))
            
            # STEP 7: Update widget after brief delay to ensure all state is reset
            QTimer.singleShot(50, lambda: self.pdf_widget.update())
    
    def change_grid_size(self, size_text):
        """Change the grid layout size"""
        if not self.grid_view_action.isChecked():
            return
        
        # Block 4x4 grid completely and handle old 3x3
        if size_text == "4x4":
            self.status_bar.showMessage("4x4 grid disabled - switching to 3x2", 2000)
            # Switch to 3x2 instead
            size_text = "3x2"
            self.grid_size_combo.setCurrentText(size_text)
        elif size_text == "3x3":
            self.status_bar.showMessage("3x3 grid changed to 3x2", 2000)
            # Switch to 3x2 instead of old 3x3
            size_text = "3x2"
            self.grid_size_combo.setCurrentText(size_text)
        elif size_text == "5x2":
            self.status_bar.showMessage("5x2 grid removed - switching to 5x1", 2000)
            # Switch to 5x1 instead of 5x2
            size_text = "5x1"
            self.grid_size_combo.setCurrentText(size_text)
        
        # Handle empty or invalid size_text
        if not size_text or size_text.strip() == "":
            # Fallback to default size if combo box is empty
            size_text = "2x2"
            # Try to restore combo box if it's empty
            self._rebuild_grid_size_combo()
            self.grid_size_combo.setCurrentText(size_text)
        
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
            
            # 🚀 GRID QUALITY FIX: Trigger grid quality upgrade after switching grid size
            # This ensures all visible pages get proper quality for current zoom level
            zoom_quality = self.pdf_widget.get_zoom_adjusted_quality()
            QTimer.singleShot(100, lambda: self.pdf_widget._upgrade_grid_quality(zoom_quality))
            
            # Kick off rendering of grid pages in background
            self.render_current_page()
        except ValueError:
            pass
    
    def closeEvent(self, event):
        if hasattr(self, 'renderer') and self.renderer:
            self.renderer.stop()
            self.renderer.wait()
        if hasattr(self, 'thumbnail_worker') and self.thumbnail_worker:
            self.thumbnail_worker.stop()
            self.thumbnail_worker.wait()
        if hasattr(self, 'pdf_doc') and self.pdf_doc is not None:
            try:
                self.pdf_doc.close()
            except:
                pass  # Ignore errors if already closed
        event.accept()


if __name__ == "__main__":
    # Fast startup: Create QApplication immediately to show window
    app = QApplication(sys.argv)
    
    # Set basic application properties quickly
    app.setApplicationName("GPU PDF Viewer")
    app.setStyle("Fusion")  # Skip version and organization for faster startup
    
    # Skip dark mode detection for faster startup - use default theme
    # (Dark mode can be applied later if needed)
    
    # Show main window immediately with minimal initialization
    viewer = PDFViewer()
    viewer.show()
    
    # Process events to make window visible immediately
    app.processEvents()
    
    # Load PDF from command line if provided (after window is shown)
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]) and sys.argv[1].endswith('.pdf'):
        # Use QTimer to load PDF after window is shown
        QTimer.singleShot(100, lambda: viewer.load_pdf(sys.argv[1]))
    
    # Hide the menu bar as per the change request
    viewer.menuBar().hide()
    
    sys.exit(app.exec())
