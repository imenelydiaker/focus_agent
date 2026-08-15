"""
Attack implementations for testing agent robustness.

This module provides various attack types and implementations for testing
browser-based agents against adversarial inputs.
"""

from .banner_attacks import svg_attack_configs
from .adaptive_attacks import TaskAdaptiveBannerAttack
from .div_injection import GoalRevealAttack, AdditionalFormFieldAttack
from .user_generated_content import (
    UserGeneratedContentAttack,
    InformationTheftCommentAttack,
    GoalRevealCommentAttack,
)
from .wasp_attacks import (
    WaspGoalHijackingUrlInjectionAttack,
    WaspGoalHijackingPlainTextAttack,
    WaspGenericUrlInjectionAttack,
    WaspGenericPlainTextAttack,
    get_wasp_attack,
    WASP_TEMPLATES,
)
