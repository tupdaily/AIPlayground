"""Models API endpoints for saving and running inference."""

import json
import logging
import math
import asyncio
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from models.schemas import SaveModelRequest, InferenceRequest, InferenceResponse, ShapeValidationError
from supabase_client import (
    save_model_to_db,
    get_model_from_db,
    get_model_state_dict,
    list_user_models,
    list_playground_models,
)
from training.inference import run_inference_local
from training.input_processor import process_input
from training.shape_validator import (
    validate_input_shape,
    infer_expected_shape_from_input_node,
    infer_expected_input_shape_full,
)
from config import settings

# Import RunPod inference only if enabled
if settings.runpod_enabled:
    from training.runpod_inference import run_inference_runpod_flash

logger = logging.getLogger(__name__)
router = APIRouter(tags=["models"])


def _normalize_graph_json(raw):
    """Ensure graph_json from DB is a dict. Parse JSON string if needed."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse graph_json string: %s", raw[:200] if len(raw) > 200 else raw)
            return None
    return None


def _graph_debug_snippet(graph_json) -> str:
    """Return a short string describing graph structure for error messages."""
    try:
        if not isinstance(graph_json, dict):
            return f"graph is {type(graph_json).__name__}, not dict"
        top = list(graph_json.keys())
        nodes = graph_json.get("nodes") or graph_json.get("Nodes") or []
        if not isinstance(nodes, list):
            return f"top_keys={top}, nodes type={type(nodes).__name__}"
        snippets = []
        for i, node in enumerate(nodes[:3]):
            if not isinstance(node, dict):
                snippets.append(f"node[{i}]={type(node).__name__}")
                continue
            t = node.get("type") or node.get("Type") or "?"
            params = node.get("params") or node.get("Params") or node.get("data") or {}
            pk = list(params.keys())[:8] if isinstance(params, dict) else []
            snippets.append(f"node[{i}] type={t!r} params_keys={pk}")
        return f"top_keys={top}; " + "; ".join(snippets)
    except Exception as e:
        return f"error building snippet: {e}"


def _log_graph_structure_for_debug(graph_json, model_id: str) -> None:
    """Log top-level keys and each node's type/params keys when shape inference fails."""
    try:
        top_keys = list(graph_json.keys()) if isinstance(graph_json, dict) else []
        logger.warning(
            "[inference] model_id=%s graph_json top-level keys: %s",
            model_id,
            top_keys,
        )
        nodes = graph_json.get("nodes") or graph_json.get("Nodes") or []
        if not isinstance(nodes, list):
            logger.warning("[inference] model_id=%s 'nodes' is not a list: %s", model_id, type(nodes))
            return
        for i, node in enumerate(nodes[:5]):
            if not isinstance(node, dict):
                logger.warning("[inference] model_id=%s node[%s] is not a dict: %s", model_id, i, type(node))
                continue
            node_type = node.get("type") or node.get("Type") or "(missing)"
            params = node.get("params") or node.get("Params") or node.get("data") or {}
            param_keys = list(params.keys()) if isinstance(params, dict) else []
            logger.warning(
                "[inference] model_id=%s node[%s] type=%r params_keys=%s",
                model_id,
                i,
                node_type,
                param_keys,
            )
    except Exception as e:
        logger.warning("[inference] Failed to log graph structure: %s", e)

# Pending inference requests: request_id -> asyncio.Event
inference_responses: dict[str, dict] = {}
inference_events: dict[str, asyncio.Event] = {}


@router.post("/api/models/save")
async def save_trained_model(request: SaveModelRequest):
    """Save a trained model to Supabase with all metadata.

    Args:
        request: SaveModelRequest containing model data, graph, and metrics

    Returns:
        {model_id: str}
    """
    try:
        model_id = save_model_to_db(
            user_id=request.user_id,
            playground_id=request.playground_id,
            model_name=request.model_name,
            model_state_dict_b64=request.model_state_dict_b64,
            graph_json=request.graph_json.dict(),
            training_config=request.training_config.dict(),
            final_metrics={
                "loss": request.final_metrics.loss,
                "accuracy": request.final_metrics.accuracy,
                "history": request.final_metrics.history,
            },
            description=request.description,
        )

        logger.info(f"Model saved successfully: {model_id}")
        return {"model_id": model_id}

    except Exception as e:
        logger.exception(f"Failed to save model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/models/{model_id}")
async def get_model(model_id: str):
    """Get model metadata.

    Args:
        model_id: Trained model ID

    Returns:
        Model metadata dictionary
    """
    try:
        model_data = get_model_from_db(model_id)
        return model_data

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to get model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/models/{model_id}/debug-graph")
async def debug_model_graph(model_id: str):
    """Debug: show what graph_json is stored for this model and what input shape we infer.

    Call this after saving a model to verify the graph structure. Useful when inference
    fails with "could not determine model's expected input shape".
    """
    try:
        model_data = get_model_from_db(model_id)
        graph_json = _normalize_graph_json(model_data.get("graph_json"))
        if graph_json is None:
            return {
                "model_id": model_id,
                "error": "No graph_json in model record",
                "graph_top_keys": None,
                "nodes_preview": None,
                "inferred_expected_shape": None,
            }

        top_keys = list(graph_json.keys()) if isinstance(graph_json, dict) else []
        nodes = graph_json.get("nodes") or graph_json.get("Nodes") or []
        if not isinstance(nodes, list):
            nodes = []

        nodes_preview = []
        for i, node in enumerate(nodes[:10]):
            if not isinstance(node, dict):
                nodes_preview.append({"index": i, "type": None, "params_keys": None, "shape_from_params": None, "note": "not a dict"})
                continue
            t = node.get("type") or node.get("Type") or "?"
            params = node.get("params") or node.get("Params") or node.get("data") or {}
            if not isinstance(params, dict):
                params = {}
            params_keys = list(params.keys())
            shape_val = None
            if "shape" in params:
                shape_val = params["shape"]
            elif "input_shape" in params:
                shape_val = params["input_shape"]
            elif "inputShape" in params:
                shape_val = params["inputShape"]
            nodes_preview.append({
                "index": i,
                "type": t,
                "params_keys": params_keys,
                "shape_from_params": shape_val,
            })

        inferred = infer_expected_shape_from_input_node(graph_json)
        full_shape = infer_expected_input_shape_full(graph_json)
        return {
            "model_id": model_id,
            "graph_top_keys": top_keys,
            "nodes_preview": nodes_preview,
            "inferred_expected_shape": list(inferred) if inferred else None,
            "inferred_expected_shape_full": list(full_shape) if full_shape else None,
            "inference_ready": inferred is not None,
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to debug graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/users/{user_id}/models")
async def list_user_trained_models(user_id: str):
    """List all trained models for a user.

    Args:
        user_id: User ID

    Returns:
        List of model metadata
    """
    try:
        models = list_user_models(user_id)
        return {"models": models}

    except Exception as e:
        logger.exception(f"Failed to list user models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/playgrounds/{playground_id}/models")
async def list_playground_trained_models(playground_id: str):
    """List all trained models for a specific playground.

    Args:
        playground_id: Playground ID

    Returns:
        List of model metadata
    """
    try:
        models = list_playground_models(playground_id)
        return {"models": models}

    except Exception as e:
        logger.exception(f"Failed to list playground models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/models/{model_id}/infer")
async def run_model_inference(model_id: str, request: InferenceRequest):
    """Run inference using a trained model.

    Args:
        model_id: Trained model ID
        request: InferenceRequest with input tensor

    Returns:
        InferenceResponse with output tensor and shape
    """
    try:
        # Fetch model metadata and state dict
        model_data = get_model_from_db(model_id)
        graph_json = _normalize_graph_json(model_data.get("graph_json"))
        if not graph_json:
            raise ValueError("Model has no valid graph_json")
        model_state_dict_b64 = get_model_state_dict(model_id)

        # Estimate payload size — RunPod has request size limits
        # Each float in JSON is ~8 chars; limit to ~10MB
        total_floats = sum(len(row) for row in request.input_tensor)
        payload_too_large = total_floats > 500_000

        # Route to local or RunPod inference
        if settings.runpod_enabled and not payload_too_large:
            logger.info(f"Running inference on RunPod for model {model_id}")
            result = await run_inference_runpod_flash(
                model_state_dict_b64=model_state_dict_b64,
                graph_json=graph_json,
                input_tensor=request.input_tensor,
                model_id=model_id,
                backend_url=settings.backend_url,
            )
        else:
            if payload_too_large:
                logger.info(f"Input too large for RunPod ({total_floats} floats), using local inference for model {model_id}")
            else:
                logger.info(f"Running local inference for model {model_id}")
            result = await run_inference_local(
                model_state_dict_b64=model_state_dict_b64,
                graph_json=graph_json,
                input_tensor=request.input_tensor,
            )

        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to run inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/models/{model_id}/infer/file")
async def run_model_inference_with_file(
    model_id: str,
    input_type: str = Form(...),  # "image" | "text" | "tensor"
    file: UploadFile | None = File(None),
    text_content: str | None = Form(None),
    image_width: int | None = Form(None),
    image_height: int | None = Form(None),
    image_channels: int | None = Form(None),
):
    """Run inference with multi-format input (image, text, or tensor file).

    Args:
        model_id: Trained model ID
        input_type: "image" | "text" | "tensor"
        file: Uploaded file (for image or tensor)
        text_content: Text string (for text input)
        image_width, image_height, image_channels: Optional overrides for image

    Returns:
        InferenceResponse or ShapeValidationError (400) if shape mismatch
    """
    try:
        # Fetch model metadata
        model_data = get_model_from_db(model_id)
        graph_json = _normalize_graph_json(model_data.get("graph_json"))
        if not graph_json:
            raise ValueError("Model has no valid graph_json (missing or invalid)")

        # Get expected input shape from model (flattened for validation)
        expected_shape = infer_expected_shape_from_input_node(graph_json)
        if not expected_shape:
            # Log and include structure in error so we can fix shape inference
            debug_snippet = _graph_debug_snippet(graph_json)
            _log_graph_structure_for_debug(graph_json, model_id)
            raise ValueError(
                "Could not determine model's expected input shape. "
                f"Graph structure: {debug_snippet}"
            )

        # For image input: get full expected shape from model and resize image to match
        image_width_to_use = image_width
        image_height_to_use = image_height
        image_channels_to_use = image_channels
        if input_type == "image":
            full_shape = infer_expected_input_shape_full(graph_json)
            if full_shape and len(full_shape) >= 2:
                # full_shape is typically (C, H, W) e.g. (1, 28, 28) or (3, 32, 32)
                if len(full_shape) == 3:
                    ch, h, w = full_shape[0], full_shape[1], full_shape[2]
                    if image_height_to_use is None:
                        image_height_to_use = h
                    if image_width_to_use is None:
                        image_width_to_use = w
                    if image_channels_to_use is None:
                        image_channels_to_use = ch
                else:
                    # (H, W) -> treat as H, W, 1
                    h, w = full_shape[0], full_shape[1]
                    if image_height_to_use is None:
                        image_height_to_use = h
                    if image_width_to_use is None:
                        image_width_to_use = w
                    if image_channels_to_use is None:
                        image_channels_to_use = 1

        # Initialize OpenAI client if needed
        openai_client = None
        if input_type == "text":
            try:
                from openai import OpenAI
                openai_client = OpenAI(api_key=settings.openai_api_key)
            except Exception as e:
                logger.error(f"OpenAI client initialization failed: {e}")
                raise ValueError("OpenAI API not configured for text embeddings")

        # Read file bytes if provided
        file_bytes = None
        filename = None
        if file:
            file_bytes = await file.read()
            filename = file.filename

        # Process input based on type (image resized to model's expected dimensions when set)
        result = await process_input(
            input_type=input_type,
            file_bytes=file_bytes,
            text_content=text_content,
            filename=filename,
            openai_client=openai_client,
            image_width=image_width_to_use,
            image_height=image_height_to_use,
            image_channels=image_channels_to_use,
        )

        # Check for processing errors
        if result.get("error"):
            raise ValueError(result["error"])

        tensor_data = result["tensor_data"]
        actual_shape = result["actual_shape"]

        if not tensor_data or not actual_shape:
            raise ValueError("Failed to process input")

        # Normalize actual_shape for validation when image returns (H, W, C) and expected is (features,)
        actual_for_validation = actual_shape
        if len(expected_shape) == 1 and len(actual_shape) >= 2:
            prod = math.prod(actual_shape)
            if prod == expected_shape[0]:
                actual_for_validation = (1, prod)

        # Validate shape
        validation = validate_input_shape(actual_for_validation, expected_shape, model_id)

        if not validation["valid"]:
            # Return shape mismatch error (400 Bad Request)
            raise HTTPException(
                status_code=400,
                detail={
                    "error": validation["error"],
                    "message": validation["message"],
                    "expected_shape": validation["expected_shape"],
                    "actual_shape": validation["actual_shape"],
                    "suggestion": validation["suggestion"],
                },
            )

        logger.info(
            f"Shape validation passed for model {model_id}: "
            f"expected {expected_shape}, got {actual_shape}"
        )

        # Get model state dict
        model_state_dict_b64 = get_model_state_dict(model_id)

        # Estimate payload size — RunPod has request size limits
        total_floats = sum(len(row) for row in tensor_data)
        payload_too_large = total_floats > 500_000

        # Route to local or RunPod inference
        if settings.runpod_enabled and not payload_too_large:
            logger.info(f"Running inference on RunPod for model {model_id}")
            inference_result = await run_inference_runpod_flash(
                model_state_dict_b64=model_state_dict_b64,
                graph_json=graph_json,
                input_tensor=tensor_data,
                model_id=model_id,
                backend_url=settings.backend_url,
            )
        else:
            if payload_too_large:
                logger.info(f"Input too large for RunPod ({total_floats} floats), using local inference for model {model_id}")
            else:
                logger.info(f"Running local inference for model {model_id}")
            inference_result = await run_inference_local(
                model_state_dict_b64=model_state_dict_b64,
                graph_json=graph_json,
                input_tensor=tensor_data,
            )

        return inference_result

    except HTTPException:
        raise
    except ValueError as e:
        logger.exception(f"Input validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to run file-based inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/models/callback")
async def inference_callback(callback_data: dict):
    """Receive inference results from RunPod.

    Args:
        callback_data: Dict with inference results

    Returns:
        {status: "ok"}
    """
    try:
        request_id = callback_data.get("request_id")

        if not request_id:
            logger.warning("Callback received without request_id")
            raise HTTPException(status_code=400, detail="request_id required")

        # Store result and signal event
        inference_responses[request_id] = callback_data
        if request_id in inference_events:
            inference_events[request_id].set()

        logger.info(f"Inference callback received for request {request_id}")
        return {"status": "ok"}

    except Exception as e:
        logger.exception(f"Failed to handle inference callback: {e}")
        raise HTTPException(status_code=500, detail=str(e))
