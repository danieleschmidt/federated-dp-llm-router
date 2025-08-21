"""
Global module for multi-region and internationalization support
"""

from .multi_region_manager import MultiRegionManager, global_region_manager
from .i18n_manager import I18nManager, global_i18n_manager

__all__ = [
    'MultiRegionManager',
    'global_region_manager', 
    'I18nManager',
    'global_i18n_manager'
]