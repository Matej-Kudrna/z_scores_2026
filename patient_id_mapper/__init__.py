"""
Patient ID Mapper Library
Maps between RC (rodné číslo), iMed_ID, and paciid.
"""

from .mapper import (
    fill_from_RC,
    fill_from_iMed,
    normalize_paciid,
    _load_mapping
)

__version__ = "1.0.0"
__all__ = [
    "fill_from_RC",
    "fill_from_iMed", 
    "normalize_paciid",
    "_load_mapping"
]
