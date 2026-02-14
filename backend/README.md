# AIPlayground Backend

**Python 3.12 or 3.13 required.** Python 3.14 is not yet supported by pydantic-core.

Create and activate a virtualenv (use the name `.venv` so it stays out of the way):

```bash
# Use Python 3.13 (e.g. python3.13 from Homebrew: brew install python@3.13)
python3.13 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the API:

```bash
python -m uvicorn main:app --reload --port 8000
```

Use `main:app` (the app lives in `main.py`), not `app.main:app`.

## Feedback

The "Get feedback" button uses OpenAI to analyze your design. Set `OPENAI_API_KEY` in your `.env` to enable it. Optional: `OPENAI_MODEL` (default: `gpt-4o-mini`).
