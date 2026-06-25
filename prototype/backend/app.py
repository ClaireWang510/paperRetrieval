from __future__ import annotations

import os
import sys
from io import BytesIO
from functools import lru_cache
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv


BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from venue_profile.ccf import resolve_venue_tier


# Ensure the release package is importable when running from prototype/backend.
RELEASE_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RELEASE_SRC) not in sys.path:
    sys.path.insert(0, str(RELEASE_SRC))

# Load backend-local environment variables from .env (if present).
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

# Keep the prototype stable on shared machines where GPU memory may already be exhausted.
os.environ.setdefault("BGE_RERANKER_DEVICE", "cpu")
os.environ.setdefault("BGE_EMBEDDING_DEVICE", "cpu")

from scientific_resource_release.retrieval.schemas import SearchRequest
from scientific_resource_release.retrieval.service import RetrievalKnowledgeService


class SearchPayload(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=20, ge=1, le=50)
    sparse_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    intent_decomposer: str = Field(default="llm")


@lru_cache(maxsize=1)
def get_service() -> RetrievalKnowledgeService:
    database_url = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/scientific_resource")
    return RetrievalKnowledgeService(database_url=database_url)


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    raw_roots = []
    configured_raw_root = os.environ.get("RAW_DATA_ROOT", "").strip()
    if configured_raw_root:
        raw_roots.append(Path(configured_raw_root).expanduser().resolve())
    for candidate in (
        Path(__file__).resolve().parents[3] / "data" / "raw",
        Path(__file__).resolve().parents[2] / "data" / "raw",
        Path(__file__).resolve().parents[1] / "data" / "raw",
    ):
        if candidate not in raw_roots:
            raw_roots.append(candidate)

    pdf_roots = [root / "pdfs" for root in raw_roots]
    latex_roots = [root / "latex" for root in raw_roots]

    def normalize_arxiv_id(value: str) -> str:
        arxiv_id = value.strip()
        if arxiv_id.lower().startswith("arxiv:"):
            arxiv_id = arxiv_id[6:]
        return arxiv_id.split("v", 1)[0]

    def resolve_pdf_path(arxiv_id: str) -> Path | None:
        for pdf_root in pdf_roots:
            exact = pdf_root / f"{arxiv_id}.pdf"
            if exact.is_file():
                return exact
            versioned = sorted(pdf_root.glob(f"{arxiv_id}v*.pdf"))
            if versioned:
                return versioned[0]
        return None

    def resolve_latex_dir(arxiv_id: str) -> Path | None:
        for latex_root in latex_roots:
            exact = latex_root / arxiv_id
            if exact.is_dir():
                return exact
            versioned = sorted(latex_root.glob(f"{arxiv_id}v*"))
            if versioned and versioned[0].is_dir():
                return versioned[0]
        return None

    @app.get("/api/health")
    def health() -> tuple[dict, int]:
        return {
            "status": "ok",
            "database_url": os.environ.get("DATABASE_URL", "postgresql://localhost:5432/scientific_resource"),
        }, 200

    @app.post("/api/search")
    def search() -> tuple[dict, int]:
        try:
            payload = SearchPayload.model_validate(request.get_json(force=True, silent=False) or {})
        except ValidationError as exc:
            return {"error": "Invalid request payload", "details": exc.errors()}, 400
        except Exception:
            return {"error": "Request body must be valid JSON"}, 400

        try:
            service = get_service()
            req = SearchRequest(
                query=payload.query.strip(),
                top_k=payload.top_k,
                sparse_weight=payload.sparse_weight,
                intent_decomposer=payload.intent_decomposer,
            )
            result = service.search_with_knowledge(
                request=req,
                top_k=payload.top_k,
                reason_top_k=min(10, payload.top_k),
                rerank_pool_size=max(payload.top_k * 4, 40),
                rerank_intent_decomposer=payload.intent_decomposer,
            )
            results = result.get("results") if isinstance(result, dict) else None
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict):
                        item["ccf_tier"] = resolve_venue_tier(item.get("venue"))
            return jsonify(result), 200
        except Exception as exc:
            return {"error": "Search failed", "details": str(exc)}, 500

    @app.get("/api/download/pdf/<path:arxiv_id>")
    def download_pdf(arxiv_id: str):
        normalized_id = normalize_arxiv_id(arxiv_id)
        pdf_path = resolve_pdf_path(normalized_id)
        if pdf_path is None:
            return {"error": f"PDF not found for arXiv id {normalized_id}"}, 404
        return send_file(pdf_path, as_attachment=True, download_name=pdf_path.name)

    @app.get("/api/download/latex/<path:arxiv_id>")
    def download_latex(arxiv_id: str):
        normalized_id = normalize_arxiv_id(arxiv_id)
        latex_dir = resolve_latex_dir(normalized_id)
        if latex_dir is None:
            return {"error": f"LaTeX source not found for arXiv id {normalized_id}"}, 404

        buffer = BytesIO()
        with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
            for file_path in latex_dir.rglob("*"):
                if file_path.is_file():
                    archive.write(file_path, arcname=file_path.relative_to(latex_dir))
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"{normalized_id}.zip",
            mimetype="application/zip",
        )

    return app


app = create_app()


if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "8001"))
    app.run(host=host, port=port, debug=False)
