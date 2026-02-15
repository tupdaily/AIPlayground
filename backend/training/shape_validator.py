"""
Shape validation for model inference inputs.

Validates that user-provided input tensors match the model's expected input shape.
"""

import json
from typing import Tuple, Dict, Any, Optional


def validate_input_shape(
    actual_shape: Tuple[int, ...],
    expected_shape: Tuple[int, ...],
    model_id: str = "unknown",
) -> Dict[str, Any]:
    """
    Validate if an actual input shape matches the expected shape for a model.

    Rules:
    - First dimension (batch size) is flexible - can be any positive integer
    - All remaining dimensions must match exactly
    - Example: expected (784,) matches actual (1, 784), (5, 784), etc.
    - Example: expected (784,) does NOT match (500,) or (28, 28)

    Args:
        actual_shape: Shape of the provided input tensor, e.g., (1, 784) or (5, 1536)
        expected_shape: Expected shape from model's input node, e.g., (784,) or (1536,)
        model_id: Model identifier for error messages

    Returns:
        Dictionary with:
        - valid (bool): Whether shapes match
        - actual_shape (list): Provided shape
        - expected_shape (list): Expected shape
        - error (str|None): Error message if invalid, None if valid
        - message (str): User-friendly message
        - suggestion (str|None): Helpful suggestion if invalid
    """

    # Convert to lists for JSON serialization
    actual_list = list(actual_shape)
    expected_list = list(expected_shape)

    # Handle empty shapes
    if len(expected_shape) == 0:
        return {
            "valid": False,
            "actual_shape": actual_list,
            "expected_shape": expected_list,
            "error": "Model has no input shape defined",
            "message": "Cannot validate: model input shape is not configured",
            "suggestion": None,
        }

    if len(actual_shape) == 0:
        return {
            "valid": False,
            "actual_shape": actual_list,
            "expected_shape": expected_list,
            "error": "Provided input has no data",
            "message": "Your input is empty",
            "suggestion": "Please provide at least one sample",
        }

    # Extract batch size (first dimension) - it's flexible
    # Expected shape tells us the feature dimensions
    expected_features = expected_shape  # Full expected shape (may include batch or not)
    actual_batch = actual_shape[0] if len(actual_shape) > 0 else 1
    actual_features = actual_shape[1:] if len(actual_shape) > 1 else (actual_shape[0],)

    # Check if remaining dimensions match
    # Case 1: Expected shape is [features] (no batch), actual is [batch, features]
    if len(expected_shape) == 1 and len(actual_shape) == 2:
        if actual_features[0] == expected_shape[0]:
            return {
                "valid": True,
                "actual_shape": actual_list,
                "expected_shape": expected_list,
                "error": None,
                "message": f"✓ Input shape matches (batch={actual_batch}, features={expected_shape[0]})",
                "suggestion": None,
            }
        else:
            suggestion = (
                f"Try resizing your image to {int(expected_shape[0] ** 0.5)} × {int(expected_shape[0] ** 0.5)} pixels"
                if expected_shape[0] in [784, 3072]
                else "Check that your input has the correct dimensions"
            )
            return {
                "valid": False,
                "actual_shape": actual_list,
                "expected_shape": expected_list,
                "error": f"Feature dimension mismatch: expected {expected_shape[0]}, got {actual_features[0]}",
                "message": f"❌ Input has {actual_features[0]} features, but model expects {expected_shape[0]}",
                "suggestion": suggestion,
            }

    # Case 2: Both have batch dimension
    if len(actual_shape) > 1 and len(expected_shape) > 1:
        if actual_features == expected_shape[1:]:
            return {
                "valid": True,
                "actual_shape": actual_list,
                "expected_shape": expected_list,
                "error": None,
                "message": f"✓ Input shape matches {actual_list}",
                "suggestion": None,
            }
        else:
            return {
                "valid": False,
                "actual_shape": actual_list,
                "expected_shape": expected_list,
                "error": f"Shape mismatch: expected {list(expected_shape)}, got {actual_list}",
                "message": f"❌ Input shape {actual_list} doesn't match expected {list(expected_shape)}",
                "suggestion": "Check the dimensions of your input file",
            }

    # Case 3: Just check if they match after accounting for batch
    if actual_shape == expected_shape:
        return {
            "valid": True,
            "actual_shape": actual_list,
            "expected_shape": expected_list,
            "error": None,
            "message": f"✓ Input shape matches {actual_list}",
            "suggestion": None,
        }

    # Default mismatch
    return {
        "valid": False,
        "actual_shape": actual_list,
        "expected_shape": expected_list,
        "error": f"Shape mismatch: expected {list(expected_shape)}, got {actual_list}",
        "message": f"❌ Input shape {actual_list} doesn't match expected {list(expected_shape)}",
        "suggestion": "Check the dimensions of your input",
    }


def _nodes_from_graph(graph_json: Dict[str, Any]) -> list:
    """Get nodes list from graph; handle 'nodes' or 'Nodes' and ensure list."""
    nodes = graph_json.get("nodes") or graph_json.get("Nodes")
    if isinstance(nodes, list):
        return nodes
    return []


def _params_from_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """Get params dict from node; handle params, Params, or data.params."""
    params = node.get("params") or node.get("Params")
    if isinstance(params, dict):
        return params
    data = node.get("data")
    if isinstance(data, dict):
        p = data.get("params") or data.get("Params")
        if isinstance(p, dict):
            return p
    return {}


def _shape_from_params(params: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
    """Extract a single (total_features,) shape from params from any known param key."""
    # shape: list/tuple e.g. [1, 28, 28] (or JSON string)
    for key in ("shape", "Shape"):
        if key not in params:
            continue
        val = params[key]
        if isinstance(val, str) and val.strip().startswith("["):
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(val, (list, tuple)) and len(val) > 0:
            total = 1
            for dim in val:
                try:
                    total *= int(dim)
                except (TypeError, ValueError):
                    pass
            if total > 0:
                return (total,)
        break

    # input_shape / inputShape: string "C,H,W" or list
    for key in ("input_shape", "inputShape"):
        if key not in params:
            continue
        raw = params[key]
        if isinstance(raw, str) and raw.strip():
            dims = []
            for x in raw.split(","):
                try:
                    dims.append(int(x.strip()))
                except (ValueError, TypeError):
                    pass
            if dims:
                total = 1
                for d in dims:
                    total *= d
                return (total,)
        if isinstance(raw, (list, tuple)) and len(raw) > 0:
            total = 1
            for d in raw:
                try:
                    total *= int(d)
                except (TypeError, ValueError):
                    pass
            if total > 0:
                return (total,)
        break

    return None


def infer_expected_shape_from_input_node(graph_json: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
    """
    Extract expected input shape from a model's graph JSON.

    - Looks for Input/input or TextInput/text_input node and shape params.
    - Handles alternate keys: nodes/Nodes, type/Type, params/Params, data.params.
    - Fallback: infer from first linear layer's in_features (flattened input).
    """

    nodes = _nodes_from_graph(graph_json)

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = (node.get("type") or node.get("Type") or "").strip()
        type_lower = node_type.lower()
        if type_lower not in ("input", "text_input"):
            continue

        params = _params_from_node(node)

        # Vision: shape or input_shape
        shape = _shape_from_params(params)
        if shape is not None:
            return shape

        # Text input: seq_len
        if type_lower == "text_input":
            for key in ("seq_len", "seqLen"):
                if key in params:
                    try:
                        return (int(params[key]),)
                    except (TypeError, ValueError):
                        pass

    # Fallback: first linear layer in_features (model expects flattened vector)
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = (node.get("type") or node.get("Type") or "").strip()
        if node_type.lower() != "linear":
            continue
        params = _params_from_node(node)
        for key in ("in_features", "inFeatures"):
            if key in params:
                try:
                    return (int(params[key]),)
                except (TypeError, ValueError):
                    pass
        break

    return None


def _full_shape_from_params(params: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
    """Extract full shape tuple (e.g. (1, 28, 28)) from params for image resizing."""
    for key in ("shape", "Shape"):
        if key not in params:
            continue
        val = params[key]
        if isinstance(val, str) and val.strip().startswith("["):
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(val, (list, tuple)) and len(val) > 0:
            try:
                return tuple(int(x) for x in val if x is not None)
            except (TypeError, ValueError):
                pass
        break
    for key in ("input_shape", "inputShape"):
        if key not in params:
            continue
        raw = params[key]
        if isinstance(raw, str) and raw.strip():
            try:
                dims = [int(x.strip()) for x in raw.split(",") if x.strip()]
                if dims:
                    return tuple(dims)
            except (ValueError, TypeError):
                pass
        if isinstance(raw, (list, tuple)) and len(raw) > 0:
            try:
                return tuple(int(x) for x in raw if x is not None)
            except (TypeError, ValueError):
                pass
        break
    return None


def infer_expected_input_shape_full(graph_json: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
    """
    Extract full expected input shape from the model graph (e.g. (1, 28, 28) for image).
    Used to resize the input image to match the model automatically.
    """
    nodes = _nodes_from_graph(graph_json)
    for node in nodes:
        if not isinstance(node, dict):
            continue
        type_lower = (node.get("type") or node.get("Type") or "").strip().lower()
        if type_lower not in ("input", "text_input"):
            continue
        params = _params_from_node(node)
        full = _full_shape_from_params(params)
        if full is not None:
            return full
    return None
