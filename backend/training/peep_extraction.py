"""Extract weights, activations, and attention maps from saved models for visualization."""

import torch
import torch.nn as nn
import base64
import io
import time
import logging
from typing import Any

from compiler.model_builder import build_model, DynamicModel
from models.schemas import GraphSchema
from training.shape_validator import infer_expected_shape_from_input_node

logger = logging.getLogger(__name__)

# Cap per-response tensor size to keep payloads reasonable (~400KB JSON).
MAX_TENSOR_ELEMENTS = 100_000

# ---------------------------------------------------------------------------
# Model cache — avoids re-downloading / rebuilding when the user clicks
# multiple blocks on the same saved model.
# ---------------------------------------------------------------------------

_model_cache: dict[str, tuple[DynamicModel, float]] = {}
_CACHE_TTL_SEC = 300  # 5 minutes


def _get_cached_model(
    model_id: str,
    model_state_dict_b64: str,
    graph_json: dict,
) -> DynamicModel:
    now = time.time()
    if model_id in _model_cache:
        model, loaded_at = _model_cache[model_id]
        if now - loaded_at < _CACHE_TTL_SEC:
            return model

    model = _load_model(model_state_dict_b64, graph_json)
    _model_cache[model_id] = (model, now)

    # Evict stale entries
    for k in list(_model_cache.keys()):
        if now - _model_cache[k][1] > _CACHE_TTL_SEC:
            del _model_cache[k]

    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(state_dict_b64: str, graph_json: dict) -> DynamicModel:
    model_bytes = base64.b64decode(state_dict_b64)
    state_dict = torch.load(io.BytesIO(model_bytes), map_location="cpu")

    if isinstance(graph_json, dict):
        graph = GraphSchema(**graph_json)
    else:
        graph = graph_json

    model = build_model(graph)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _truncate_tensor(
    tensor: torch.Tensor, max_elements: int = MAX_TENSOR_ELEMENTS
) -> torch.Tensor:
    if tensor.numel() <= max_elements:
        return tensor
    if tensor.dim() >= 2:
        elements_per_slice = tensor[0].numel()
        max_slices = max(1, max_elements // elements_per_slice)
        return tensor[:max_slices]
    return tensor[:max_elements]


def _to_tensor_slice(tensor: torch.Tensor) -> dict[str, Any]:
    t = _truncate_tensor(tensor)
    return {
        "data": t.detach().cpu().float().flatten().tolist(),
        "shape": list(t.shape),
    }


# ---------------------------------------------------------------------------
# Weight extraction (no forward pass needed)
# ---------------------------------------------------------------------------

def extract_block_weights(
    model: DynamicModel,
    block_id: str,
) -> dict[str, Any]:
    """Extract weight tensors for a specific block.

    Returns dict with keys: weights, filters, gradients (always None for saved).
    """
    result: dict[str, Any] = {"weights": None, "filters": None, "gradients": None}

    if block_id not in model.layers:
        return result

    layer = model.layers[block_id]

    if not hasattr(layer, "weight"):
        return result

    weight = layer.weight.data

    if isinstance(layer, nn.Conv2d):
        # weight shape: [out_channels, in_channels, kH, kW]
        if weight.dim() == 4:
            # Filters: average across input channels → [out_channels, kH, kW]
            filters_2d = weight.mean(dim=1)
            result["filters"] = _to_tensor_slice(filters_2d)
            # Weight heatmap: flatten spatial dims → [out_channels, in_channels*kH*kW]
            result["weights"] = _to_tensor_slice(
                weight.reshape(weight.shape[0], -1)
            )
    elif isinstance(layer, nn.Linear):
        result["weights"] = _to_tensor_slice(weight)
    elif isinstance(layer, nn.Embedding):
        result["weights"] = _to_tensor_slice(weight)
    elif isinstance(layer, nn.MultiheadAttention):
        if hasattr(layer, "in_proj_weight") and layer.in_proj_weight is not None:
            result["weights"] = _to_tensor_slice(layer.in_proj_weight.data)
        elif hasattr(layer, "q_proj_weight") and layer.q_proj_weight is not None:
            result["weights"] = _to_tensor_slice(layer.q_proj_weight.data)
    else:
        result["weights"] = _to_tensor_slice(weight)

    return result


# ---------------------------------------------------------------------------
# Activation capture (requires forward pass)
# ---------------------------------------------------------------------------

def capture_block_activations(
    model: DynamicModel,
    block_id: str,
    input_tensor: list[list[float]] | None,
    graph_json: dict,
) -> dict[str, Any]:
    """Run a forward pass with capture to get activations / attention maps.

    If input_tensor is None, a random input matching the graph's input shape
    is generated automatically.
    """
    result: dict[str, Any] = {"activations": None, "attentionMap": None}

    # Build the input tensor
    if input_tensor is not None:
        inp = torch.tensor(input_tensor, dtype=torch.float32)
    else:
        expected_shape = infer_expected_shape_from_input_node(graph_json)
        if expected_shape:
            inp = torch.randn(1, *expected_shape)
        else:
            # Fallback: try a small default
            inp = torch.randn(1, 1, 28, 28)

    with torch.no_grad():
        try:
            _, activation, attention = model.forward_with_capture(inp, block_id)
        except Exception as e:
            logger.warning(f"Forward pass failed during peep capture: {e}")
            return result

    if activation is not None:
        if isinstance(activation, tuple):
            activation = activation[0]

        if activation.dim() == 4:
            # Conv output [B, C, H, W] → [C, H, W] (first batch item)
            result["activations"] = _to_tensor_slice(activation[0])
        elif activation.dim() == 3:
            # [B, seq, features] → [seq, features]
            result["activations"] = _to_tensor_slice(activation[0])
        elif activation.dim() >= 1:
            result["activations"] = _to_tensor_slice(activation)

    if attention is not None:
        # attention shape: [B, num_heads, seq_q, seq_k]
        if attention.dim() == 4:
            result["attentionMap"] = _to_tensor_slice(attention[0])  # [heads, sq, sk]
        elif attention.dim() == 3:
            # [B, sq, sk] — averaged heads; wrap as [1, sq, sk]
            result["attentionMap"] = _to_tensor_slice(attention[0].unsqueeze(0))

    return result
