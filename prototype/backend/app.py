from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import uuid
from datetime import datetime, timezone
from io import BytesIO
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZipFile

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from werkzeug.utils import secure_filename


BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from venue_profile.ccf import resolve_venue_tier


if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

logger = logging.getLogger(__name__)


# Ensure the release package is importable when running from prototype/backend.
RELEASE_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RELEASE_SRC) not in sys.path:
    sys.path.insert(0, str(RELEASE_SRC))

# Load backend-local environment variables from .env (if present).
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

# Keep the prototype stable on shared machines where GPU memory may already be exhausted.
os.environ.setdefault("BGE_RERANKER_DEVICE", "cpu")
os.environ.setdefault("BGE_EMBEDDING_DEVICE", "cpu")

# from semantic_workspace import run_document_semantic_pipeline
from scientific_resource_release.retrieval.schemas import SearchRequest
from scientific_resource_release.retrieval.service import RetrievalKnowledgeService


class SearchPayload(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=20, ge=1, le=50)
    sparse_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    intent_decomposer: str = Field(default="none")
    generate_reasons: bool = Field(default=False)
    reason_top_k: int = Field(default=0, ge=0, le=50)
    generate_answer: bool = Field(default=False)
    rerank_pool_size: int | None = Field(default=20, ge=1, le=200)


class ReasonPayload(BaseModel):
    query: str = Field(min_length=1)
    arxiv_id: str = Field(min_length=1)
    title: str | None = None
    abstract: str | None = None
    top_n: int = Field(default=3, ge=1, le=5)


class AnswerPayload(BaseModel):
    query: str = Field(min_length=1)
    top_n_papers: int = Field(default=10, ge=1, le=10)
    top_n_units: int = Field(default=3, ge=1, le=5)


class AgentWorkspaceCreatePayload(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)


class AgentArxivIngestPayload(BaseModel):
    arxiv_id: str = Field(min_length=3, max_length=64)


class AgentChatPayload(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


class AgentDocumentProcessPayload(BaseModel):
    document_id: str = Field(min_length=1)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(raw_name: str) -> str:
    cleaned = secure_filename((raw_name or "").strip())
    if cleaned:
        return cleaned
    return "uploaded.pdf"


def _normalize_arxiv_id(raw_value: str) -> str:
    arxiv_id = (raw_value or "").strip()
    if arxiv_id.lower().startswith("arxiv:"):
        arxiv_id = arxiv_id[6:]
    arxiv_id = arxiv_id.split("v", 1)[0]
    arxiv_id = re.sub(r"\s+", "", arxiv_id)
    return arxiv_id


def _download_url(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "ScientificResourceAgent/1.0"})
    with urlopen(req, timeout=60) as response, target_path.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _agent_llm_call(messages: list[dict[str, str]]) -> str:
    api_key = (
        os.environ.get("AGENT_LLM_API_KEY", "").strip()
        or os.environ.get("LLM_API_KEY", "").strip()
    )
    base_url = (
        os.environ.get("AGENT_LLM_BASE_URL", "").strip()
        or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").strip()
    )
    base_url = base_url.lstrip("=")
    model = os.environ.get("AGENT_LLM_MODEL", "").strip() or os.environ.get("LLM_MODEL", "gpt-4o-mini").strip()
    if not api_key:
        raise RuntimeError("Missing AGENT_LLM_API_KEY (or fallback LLM_API_KEY)")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()


class AgentWorkspaceStore:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.data_dir = root_dir / "data"
        self.workspace_assets_dir = root_dir / "workspace_assets"
        self.workspaces_file = self.data_dir / "workspaces.json"
        self.tasks_file = self.data_dir / "tasks.json"
        self._lock = threading.Lock()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_assets_dir.mkdir(parents=True, exist_ok=True)
        if not self.workspaces_file.exists():
            self._write_json_file(self.workspaces_file, [])
        if not self.tasks_file.exists():
            self._write_json_file(self.tasks_file, {})

    def _read_json_file(self, path: Path, default):
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def _write_json_file(self, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp:
            json.dump(payload, tmp, ensure_ascii=False, indent=2)
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)

    def list_workspaces(self) -> list[dict]:
        with self._lock:
            items = self._read_json_file(self.workspaces_file, [])
            return sorted(items, key=lambda x: x.get("updated_at", ""), reverse=True)

    def get_workspace(self, workspace_id: str) -> dict | None:
        with self._lock:
            items = self._read_json_file(self.workspaces_file, [])
            for item in items:
                if item.get("id") == workspace_id:
                    return item
        return None

    def create_workspace(self, name: str, description: str) -> dict:
        now = _utc_now_iso()
        workspace = {
            "id": str(uuid.uuid4()),
            "name": name.strip(),
            "description": description.strip(),
            "created_at": now,
            "updated_at": now,
            "documents": [],
            "conversation": [],
        }
        with self._lock:
            items = self._read_json_file(self.workspaces_file, [])
            items.append(workspace)
            self._write_json_file(self.workspaces_file, items)
        (self.workspace_assets_dir / workspace["id"]).mkdir(parents=True, exist_ok=True)
        return workspace

    def append_document(self, workspace_id: str, document: dict) -> dict | None:
        with self._lock:
            items = self._read_json_file(self.workspaces_file, [])
            for idx, item in enumerate(items):
                if item.get("id") != workspace_id:
                    continue
                docs = item.setdefault("documents", [])
                docs.append(document)
                item["updated_at"] = _utc_now_iso()
                items[idx] = item
                self._write_json_file(self.workspaces_file, items)
                return item
        return None

    def get_document(self, workspace_id: str, document_id: str) -> dict | None:
        workspace = self.get_workspace(workspace_id)
        if workspace is None:
            return None
        docs = workspace.get("documents") or []
        for doc in docs:
            if doc.get("id") == document_id:
                return doc
        return None

    def update_document(self, workspace_id: str, document_id: str, updates: dict) -> dict | None:
        with self._lock:
            items = self._read_json_file(self.workspaces_file, [])
            for idx, item in enumerate(items):
                if item.get("id") != workspace_id:
                    continue
                docs = item.setdefault("documents", [])
                for doc_idx, doc in enumerate(docs):
                    if doc.get("id") != document_id:
                        continue
                    doc.update(updates)
                    docs[doc_idx] = doc
                    item["documents"] = docs
                    item["updated_at"] = _utc_now_iso()
                    items[idx] = item
                    self._write_json_file(self.workspaces_file, items)
                    return item
        return None

    def append_conversation(self, workspace_id: str, role: str, content: str) -> dict | None:
        entry = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "created_at": _utc_now_iso(),
        }
        with self._lock:
            items = self._read_json_file(self.workspaces_file, [])
            for idx, item in enumerate(items):
                if item.get("id") != workspace_id:
                    continue
                conv = item.setdefault("conversation", [])
                conv.append(entry)
                if len(conv) > 40:
                    item["conversation"] = conv[-40:]
                item["updated_at"] = _utc_now_iso()
                items[idx] = item
                self._write_json_file(self.workspaces_file, items)
                return item
        return None

    def create_task(self, workspace_id: str, task_type: str, payload: dict) -> dict:
        task = {
            "id": str(uuid.uuid4()),
            "workspace_id": workspace_id,
            "type": task_type,
            "status": "queued",
            "payload": payload,
            "result": None,
            "error": None,
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }
        with self._lock:
            tasks = self._read_json_file(self.tasks_file, {})
            tasks[task["id"]] = task
            self._write_json_file(self.tasks_file, tasks)
        return task

    def update_task(self, task_id: str, **updates) -> dict | None:
        with self._lock:
            tasks = self._read_json_file(self.tasks_file, {})
            task = tasks.get(task_id)
            if not task:
                return None
            task.update(updates)
            task["updated_at"] = _utc_now_iso()
            tasks[task_id] = task
            self._write_json_file(self.tasks_file, tasks)
            return task

    def get_task(self, task_id: str) -> dict | None:
        with self._lock:
            tasks = self._read_json_file(self.tasks_file, {})
            return tasks.get(task_id)


AGENT_STORE = AgentWorkspaceStore(BACKEND_ROOT / "agent_workspace_store")


def _start_document_semantic_task(workspace_id: str, document_id: str) -> dict:
    task = AGENT_STORE.create_task(
        workspace_id=workspace_id,
        task_type="semantic_pipeline",
        payload={"document_id": document_id},
    )
    logger.info("[semantic-task] queued workspace=%s document=%s task=%s", workspace_id, document_id, task["id"])
    AGENT_STORE.update_document(
        workspace_id,
        document_id,
        {
            "semantic_status": "queued",
            "semantic_task_id": task["id"],
        },
    )
    worker = threading.Thread(
        target=_run_document_semantic_task,
        kwargs={
            "task_id": task["id"],
            "workspace_id": workspace_id,
            "document_id": document_id,
        },
        daemon=True,
    )
    worker.start()
    return task


def _run_document_semantic_task(task_id: str, workspace_id: str, document_id: str) -> None:
    logger.info("[semantic-task] start task=%s workspace=%s document=%s", task_id, workspace_id, document_id)
    AGENT_STORE.update_task(task_id, status="running", stage="starting")
    AGENT_STORE.update_document(workspace_id, document_id, {"semantic_status": "running"})
    try:
        workspace_root = AGENT_STORE.workspace_assets_dir / workspace_id
        document = AGENT_STORE.get_document(workspace_id, document_id)
        if document is None:
            raise ValueError("Document not found")

        logger.info(
            "[semantic-task] pipeline begin task=%s source_kind=%s title=%s",
            task_id,
            document.get("kind"),
            document.get("title") or document.get("arxiv_id") or document_id,
        )
        AGENT_STORE.update_task(task_id, stage="pipeline_running")
        structured = run_document_semantic_pipeline(workspace_root=workspace_root, document=document)
        logger.info(
            "[semantic-task] pipeline finished task=%s facts=%s units=%s",
            task_id,
            structured.get("atomic_fact_count", 0),
            structured.get("semantic_unit_count", 0),
        )
        AGENT_STORE.update_document(
            workspace_id,
            document_id,
            {
                "semantic_status": "completed",
                "semantic_result": structured,
                "semantic_completed_at": _utc_now_iso(),
            },
        )
        AGENT_STORE.update_task(
            task_id,
            status="completed",
            stage="completed",
            result={
                "document_id": document_id,
                "paper_id": structured.get("paper_id"),
                "semantic_unit_count": structured.get("semantic_unit_count", 0),
                "atomic_fact_count": structured.get("atomic_fact_count", 0),
            },
            error=None,
        )
    except Exception as exc:
        logger.exception("[semantic-task] failed task=%s workspace=%s document=%s", task_id, workspace_id, document_id)
        AGENT_STORE.update_document(
            workspace_id,
            document_id,
            {
                "semantic_status": "failed",
                "semantic_error": str(exc),
            },
        )
        AGENT_STORE.update_task(task_id, status="failed", stage="failed", error=str(exc))


def _run_arxiv_download_task(task_id: str, workspace_id: str, arxiv_id: str) -> None:
    AGENT_STORE.update_task(task_id, status="running")
    normalized = _normalize_arxiv_id(arxiv_id)
    try:
        workspace_root = AGENT_STORE.workspace_assets_dir / workspace_id
        arxiv_dir = workspace_root / "arxiv" / normalized
        arxiv_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = arxiv_dir / f"{normalized}.pdf"
        latex_path = arxiv_dir / f"{normalized}.tar"

        _download_url(f"https://arxiv.org/pdf/{normalized}.pdf", pdf_path)
        _download_url(f"https://arxiv.org/e-print/{normalized}", latex_path)

        doc = {
            "id": str(uuid.uuid4()),
            "kind": "arxiv",
            "arxiv_id": normalized,
            "title": f"arXiv:{normalized}",
            "status": "downloaded",
            "pdf_path": str(pdf_path),
            "latex_path": str(latex_path),
            "created_at": _utc_now_iso(),
            "task_id": task_id,
            "semantic_status": "pending",
        }
        AGENT_STORE.append_document(workspace_id, doc)
        semantic_task = _start_document_semantic_task(workspace_id, doc["id"])
        AGENT_STORE.update_task(
            task_id,
            status="completed",
            result={
                "arxiv_id": normalized,
                "pdf_path": str(pdf_path),
                "latex_path": str(latex_path),
                "semantic_status": "queued",
                "semantic_task_id": semantic_task["id"],
            },
            error=None,
        )
    except (HTTPError, URLError) as exc:
        AGENT_STORE.update_task(task_id, status="failed", error=f"arXiv download failed: {exc}")
    except Exception as exc:
        AGENT_STORE.update_task(task_id, status="failed", error=str(exc))


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
        return _normalize_arxiv_id(value)

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

    @app.get("/api/agent/workspaces")
    def list_agent_workspaces() -> tuple[dict, int]:
        return {"workspaces": AGENT_STORE.list_workspaces()}, 200

    @app.post("/api/agent/workspaces")
    def create_agent_workspace() -> tuple[dict, int]:
        try:
            payload = AgentWorkspaceCreatePayload.model_validate(request.get_json(force=True, silent=False) or {})
        except ValidationError as exc:
            return {"error": "Invalid request payload", "details": exc.errors()}, 400
        except Exception:
            return {"error": "Request body must be valid JSON"}, 400

        workspace = AGENT_STORE.create_workspace(payload.name, payload.description)
        return {"workspace": workspace}, 201

    @app.get("/api/agent/workspaces/<string:workspace_id>")
    def get_agent_workspace(workspace_id: str) -> tuple[dict, int]:
        workspace = AGENT_STORE.get_workspace(workspace_id)
        if workspace is None:
            return {"error": "Workspace not found"}, 404
        return {"workspace": workspace}, 200

    @app.post("/api/agent/workspaces/<string:workspace_id>/upload_pdf")
    def upload_workspace_pdf(workspace_id: str) -> tuple[dict, int]:
        workspace = AGENT_STORE.get_workspace(workspace_id)
        if workspace is None:
            return {"error": "Workspace not found"}, 404

        file = request.files.get("file")
        if file is None:
            return {"error": "Missing file field"}, 400

        filename = _safe_filename(file.filename or "uploaded.pdf")
        workspace_root = AGENT_STORE.workspace_assets_dir / workspace_id
        upload_dir = workspace_root / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        target_name = f"{uuid.uuid4().hex}_{filename}"
        target_path = upload_dir / target_name
        file.save(target_path)

        document = {
            "id": str(uuid.uuid4()),
            "kind": "pdf_upload",
            "title": request.form.get("title", "").strip() or filename,
            "filename": filename,
            "storage_path": str(target_path),
            "status": "uploaded",
            "created_at": _utc_now_iso(),
            "semantic_status": "pending",
        }
        AGENT_STORE.append_document(workspace_id, document)
        semantic_task = _start_document_semantic_task(workspace_id, document["id"])
        return {
            "workspace": AGENT_STORE.get_workspace(workspace_id),
            "document": document,
            "semantic_task": semantic_task,
        }, 201

    @app.post("/api/agent/workspaces/<string:workspace_id>/ingest_arxiv")
    def ingest_workspace_arxiv(workspace_id: str) -> tuple[dict, int]:
        workspace = AGENT_STORE.get_workspace(workspace_id)
        if workspace is None:
            return {"error": "Workspace not found"}, 404

        try:
            payload = AgentArxivIngestPayload.model_validate(request.get_json(force=True, silent=False) or {})
        except ValidationError as exc:
            return {"error": "Invalid request payload", "details": exc.errors()}, 400
        except Exception:
            return {"error": "Request body must be valid JSON"}, 400

        normalized = _normalize_arxiv_id(payload.arxiv_id)
        if not normalized:
            return {"error": "Invalid arXiv id"}, 400

        task = AGENT_STORE.create_task(
            workspace_id=workspace_id,
            task_type="arxiv_ingest",
            payload={"arxiv_id": normalized},
        )
        worker = threading.Thread(
            target=_run_arxiv_download_task,
            kwargs={
                "task_id": task["id"],
                "workspace_id": workspace_id,
                "arxiv_id": normalized,
            },
            daemon=True,
        )
        worker.start()
        return {"task": task}, 202

    @app.post("/api/agent/workspaces/<string:workspace_id>/process_document_semantics")
    def process_document_semantics(workspace_id: str) -> tuple[dict, int]:
        workspace = AGENT_STORE.get_workspace(workspace_id)
        if workspace is None:
            return {"error": "Workspace not found"}, 404

        try:
            payload = AgentDocumentProcessPayload.model_validate(request.get_json(force=True, silent=False) or {})
        except ValidationError as exc:
            return {"error": "Invalid request payload", "details": exc.errors()}, 400
        except Exception:
            return {"error": "Request body must be valid JSON"}, 400

        doc = AGENT_STORE.get_document(workspace_id, payload.document_id)
        if doc is None:
            return {"error": "Document not found"}, 404

        task = _start_document_semantic_task(workspace_id, payload.document_id)
        return {"task": task}, 202

    @app.get("/api/agent/tasks/<string:task_id>")
    def get_agent_task(task_id: str) -> tuple[dict, int]:
        task = AGENT_STORE.get_task(task_id)
        if task is None:
            return {"error": "Task not found"}, 404
        return {"task": task}, 200

    @app.post("/api/agent/workspaces/<string:workspace_id>/chat")
    def chat_in_workspace(workspace_id: str) -> tuple[dict, int]:
        workspace = AGENT_STORE.get_workspace(workspace_id)
        if workspace is None:
            return {"error": "Workspace not found"}, 404

        try:
            payload = AgentChatPayload.model_validate(request.get_json(force=True, silent=False) or {})
        except ValidationError as exc:
            return {"error": "Invalid request payload", "details": exc.errors()}, 400
        except Exception:
            return {"error": "Request body must be valid JSON"}, 400

        user_message = payload.message.strip()
        updated_workspace = AGENT_STORE.append_conversation(workspace_id, "user", user_message)
        if updated_workspace is None:
            return {"error": "Workspace not found"}, 404

        docs = updated_workspace.get("documents") or []
        doc_summary_lines: list[str] = []
        for doc in docs[:20]:
            kind = str(doc.get("kind") or "unknown")
            title = str(doc.get("title") or doc.get("filename") or doc.get("arxiv_id") or "untitled")
            status = str(doc.get("status") or "unknown")
            semantic_status = str(doc.get("semantic_status") or "pending")
            doc_summary_lines.append(
                f"- [{kind}] {title} (status={status}, semantic_status={semantic_status})"
            )
        doc_summary = "\n".join(doc_summary_lines) if doc_summary_lines else "- No documents yet"

        conversation = updated_workspace.get("conversation") or []
        recent_turns = conversation[-12:]
        chat_messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are the Knowledge Exploration Agent for scientific paper assistants. "
                    "Base your response on workspace documents and user intent. "
                    "If information is missing, clearly say what to ingest next."
                ),
            },
            {
                "role": "system",
                "content": (
                    f"Workspace: {updated_workspace.get('name') or 'Untitled'}\n"
                    f"Description: {updated_workspace.get('description') or '-'}\n"
                    f"Documents:\n{doc_summary}"
                ),
            },
        ]
        for turn in recent_turns:
            role = str(turn.get("role") or "user")
            if role not in {"user", "assistant"}:
                continue
            content = str(turn.get("content") or "").strip()
            if not content:
                continue
            chat_messages.append({"role": role, "content": content})

        try:
            assistant_reply = _agent_llm_call(chat_messages)
        except Exception as exc:
            return {"error": "Agent chat failed", "details": str(exc)}, 500

        after_reply = AGENT_STORE.append_conversation(workspace_id, "assistant", assistant_reply)
        return {
            "reply": assistant_reply,
            "workspace": after_reply,
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
            reason_top_k = payload.reason_top_k if payload.generate_reasons else 0
            result = service.search_with_knowledge(
                request=req,
                top_k=payload.top_k,
                reason_top_k=min(reason_top_k, payload.top_k),
                rerank_pool_size=payload.rerank_pool_size,
                rerank_intent_decomposer=payload.intent_decomposer,
                generate_answer=payload.generate_answer,
            )
            results = result.get("results") if isinstance(result, dict) else None
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict):
                        item["ccf_tier"] = resolve_venue_tier(item.get("venue"))
            service.cache_search_context(query=payload.query.strip(), search_payload=result)
            return jsonify(result), 200
        except Exception as exc:
            return {"error": "Search failed", "details": str(exc)}, 500

    @app.post("/api/reason")
    def generate_reason() -> tuple[dict, int]:
        try:
            payload = ReasonPayload.model_validate(request.get_json(force=True, silent=False) or {})
        except ValidationError as exc:
            return {"error": "Invalid request payload", "details": exc.errors()}, 400
        except Exception:
            return {"error": "Request body must be valid JSON"}, 400

        try:
            service = get_service()
            result = service.generate_recommendation_reason(
                query=payload.query.strip(),
                arxiv_id=payload.arxiv_id.strip(),
                title=payload.title,
                abstract=payload.abstract,
                top_n=payload.top_n,
            )
            return jsonify(result), 200
        except ValueError as exc:
            return {"error": str(exc)}, 400
        except Exception as exc:
            return {"error": "Reason generation failed", "details": str(exc)}, 500

    @app.post("/api/answer")
    def generate_answer() -> tuple[dict, int]:
        try:
            payload = AnswerPayload.model_validate(request.get_json(force=True, silent=False) or {})
        except ValidationError as exc:
            return {"error": "Invalid request payload", "details": exc.errors()}, 400
        except Exception:
            return {"error": "Request body must be valid JSON"}, 400

        try:
            service = get_service()
            result = service.generate_answer_for_query(
                query=payload.query.strip(),
                top_n_papers=payload.top_n_papers,
                top_n_units=payload.top_n_units,
            )
            return jsonify(result), 200
        except ValueError as exc:
            return {"error": str(exc)}, 400
        except Exception as exc:
            return {"error": "Answer generation failed", "details": str(exc)}, 500

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
