"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

# Embedding model configuration
MODEL_ALIASES = {
    'mini': 'sentence-transformers/all-MiniLM-L6-v2',      # 384 dims, ~5x faster
    'base': 'sentence-transformers/all-mpnet-base-v2',     # 768 dims, balanced
    'large': 'sentence-transformers/all-mpnet-base-v2',    # Same as base for now
}

# Default model for new indexes
DEFAULT_MODEL = MODEL_ALIASES['mini']

def resolve_model_alias(model_name: str) -> str:
    """
    Resolve model alias to full model name
    
    Args:
        model_name: Model name or alias (mini, base, large)
        
    Returns:
        Full model name
    """
    return MODEL_ALIASES.get(model_name, model_name)