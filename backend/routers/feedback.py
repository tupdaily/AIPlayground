"""Design feedback on playground graphs via OpenAI."""

import json
import os

from fastapi import APIRouter

from config import settings
from models.schemas import FeedbackRequest

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


def _build_graph_context(graph: dict) -> str:
    """Build graph description for the LLM context."""
    nodes_summary = []
    for n in graph.get("nodes", []):
        t = n.get("type", "unknown")
        p = n.get("params", {})
        nodes_summary.append(f"- {t}: {json.dumps(p)}")
    graph_desc = "\n".join(nodes_summary)
    edges = graph.get("edges", [])
    edge_desc = ", ".join(f"{e['source']}→{e['target']}" for e in edges[:20])
    if len(edges) > 20:
        edge_desc += f" ... ({len(edges)} total)"
    return f"""Graph:
Nodes:
{graph_desc}

Connections: {edge_desc}"""


async def _generate_feedback(graph: dict, messages: list[dict]) -> str:
    """Use OpenAI to generate design feedback from a chat conversation."""
    api_key = settings.openai_api_key
    if not api_key:
        return (
            "Feedback unavailable: OPENAI_API_KEY not set. "
            "Add it to your backend .env to enable AI feedback."
        )

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        graph_context = _build_graph_context(graph)

        system_content = f"""You are an expert in deep learning and neural network architecture.
The user is building a neural network in a visual playground. You see the current design below.
Answer their questions and give concise, constructive feedback. Be practical and actionable.

Current design:
{graph_context}"""

        api_messages = [{"role": "system", "content": system_content}]
        api_messages.extend([{"role": m["role"], "content": m["content"]} for m in messages])

        response = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=api_messages,
            max_tokens=500,
        )
        text = response.choices[0].message.content
        return text.strip() if text else "No feedback generated."
    except Exception as e:
        return f"Error generating feedback: {str(e)}"


@router.post("")
async def get_feedback(req: FeedbackRequest) -> dict:
    """
    Chat about the playground graph design. Accepts graph + message history.
    Set OPENAI_API_KEY in your environment.
    """
    graph_dict = req.graph.model_dump()
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    feedback = await _generate_feedback(graph_dict, messages)
    return {"feedback": feedback}
