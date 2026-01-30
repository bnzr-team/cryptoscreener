"""Policy inputs providers.

Provides implementations for PolicyInputsProvider protocol.
"""

from cryptoscreener.trading.policy.providers.constant import ConstantPolicyInputsProvider
from cryptoscreener.trading.policy.providers.fixture import FixturePolicyInputsProvider

__all__ = [
    "ConstantPolicyInputsProvider",
    "FixturePolicyInputsProvider",
]
