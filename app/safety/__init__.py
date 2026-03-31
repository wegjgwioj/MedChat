# -*- coding: utf-8 -*-

"""Safety helpers."""

from .safety_fuse import (
    apply_confirmed_safety_fuse_to_text,
    apply_confirmed_safety_fuse_to_triage_answer,
)

__all__ = [
    "apply_confirmed_safety_fuse_to_text",
    "apply_confirmed_safety_fuse_to_triage_answer",
]
