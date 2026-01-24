"""
LLM Explain module for CryptoScreener-X.

Provides text explanations for trading signals without modifying
any numeric values or prediction statuses.

Per CLAUDE.md §9:
- LLM is text-only; no scoring/status changes allowed
- All outputs validated for no-new-numbers constraint
- Fallback on any failure (exception/timeout/invalid output)
"""

from cryptoscreener.explain_llm.explainer import (
    AnthropicExplainer,
    AnthropicExplainerConfig,
    ExplainLLM,
    MockExplainer,
)

__all__ = [
    "AnthropicExplainer",
    "AnthropicExplainerConfig",
    "ExplainLLM",
    "MockExplainer",
]
