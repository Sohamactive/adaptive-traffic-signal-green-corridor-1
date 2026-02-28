"""Flask GUI package for the adaptive traffic signal system.

Provides ``create_app()`` — an application factory that wires up routes
and shared resources (detector instance) without polluting module-level state.
"""

from gui.app import create_app

__all__ = ["create_app"]
