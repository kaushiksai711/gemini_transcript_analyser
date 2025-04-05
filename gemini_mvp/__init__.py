# gemini_mvp/__init__.py - PACKAGE DEFINITION
# Package initialization and exports

"""
Gemini Video Transcript Analysis MVP
====================================

A package for processing video transcripts and supporting multi-turn
question answering using Google's Gemini API.
"""

from .transcript_processor import TranscriptProcessor
from .version import __version__

__all__ = ["TranscriptProcessor", "__version__"]
