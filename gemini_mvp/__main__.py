# gemini_mvp/__main__.py - ACTIVELY USED
# Entry point for running the package as a module

"""
Main entry point for the Gemini MVP package.

This module provides the entry point when the package is run
directly with `python -m gemini_mvp`.
"""

from .cli import cli

if __name__ == '__main__':
    cli()
