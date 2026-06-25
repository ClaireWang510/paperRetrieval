from __future__ import annotations

import json
import logging
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scientific_resource_release.models import PreprocessConfig
from scientific_resource_release.utils.io import read_json, write_json
from scientific_resource_release.utils.paths import resolve_project_paths
from scientific_resource_release.utils.semantic_scholar import SemanticScholarClient


logger = logging.getLogger(__name__)


class PreprocessPipeline:
    """
    Standalone preprocessing pipeline.

    Stages:
    1) Parse local LaTeX source into metadata.json.
    2) Build section figure/table annotation.json.
    3) Optional figure/table semantic understanding via LLM API.
    3) Optional Semantic Scholar metadata refresh.
    """

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.paths = resolve_project_paths(config.project_root)

        self.semantic_scholar = SemanticScholarClient()

        self.latex_dir = config.latex_dir or self.paths.latex_root
        self.pdf_dir = config.pdf_dir or self.paths.pdf_root
        self.extracted_dir = config.extracted_dir or self.paths.extracted_root
        self.jsonl_path = config.jsonl_path or self.paths.default_jsonl

    def process_one(self, arxiv_id: str) -> Path:
        clean_id = arxiv_id.split("v")[0]
        out_dir = self.extracted_dir / clean_id
        metadata_path = out_dir / "metadata.json"

        if self.config.skip_done and metadata_path.exists():
            logger.info("Skip %s: metadata already exists", clean_id)
        else:
            self._run_latex_stage(clean_id)

        if self.config.run_annotation:
            self._run_annotation_stage(metadata_path)

        if self.config.refresh_semantic_scholar:
            self._refresh_semantic_scholar(metadata_path, clean_id)

        return metadata_path

    def process_batch(self, arxiv_ids: List[str]) -> Dict[str, Path]:
        results: Dict[str, Path] = {}
        for arxiv_id in arxiv_ids:
            clean_id = arxiv_id.split("v")[0]
            try:
                results[clean_id] = self.process_one(clean_id)
            except Exception as exc:
                logger.exception("Preprocess failed for %s: %s", clean_id, exc)
        return results

    def process_batch_from_jsonl(self) -> Dict[str, Path]:
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL not found: {self.jsonl_path}")
        arxiv_ids = self._load_arxiv_ids_from_jsonl(self.jsonl_path)
        logger.info("Loaded %d arXiv IDs from %s", len(arxiv_ids), self.jsonl_path)
        return self.process_batch(arxiv_ids)

    def _run_latex_stage(self, arxiv_id: str) -> None:
        logger.info("Run standalone LaTeX extraction for %s", arxiv_id)
        paper_dir = self.latex_dir / arxiv_id
        if not paper_dir.exists():
            raise FileNotFoundError(f"LaTeX source directory not found: {paper_dir}")

        main_tex = self._find_main_tex_file(paper_dir)
        if main_tex is None:
            raise FileNotFoundError(f"No main .tex file found in: {paper_dir}")

        raw = main_tex.read_text(encoding="utf-8", errors="ignore")
        title = self._extract_title(raw)
        abstract = self._extract_abstract(raw)
        authors = self._extract_authors(raw)
        sections = self._extract_sections(raw)
        figures = self._extract_figures(raw, paper_dir, arxiv_id)
        tables = self._extract_tables(raw, arxiv_id)
        links = self._extract_resource_links(raw)

        out_dir = self.extracted_dir / arxiv_id
        metadata_path = out_dir / "metadata.json"
        out_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "sections": sections,
            "figures": figures,
            "tables": tables,
            "resource_links": links,
            "latex_source_dir": str(paper_dir),
            "citationCount": None,
            "influentialCitationCount": None,
            "fieldsOfStudy": [],
            "published_date": None,
            "venue": None,
            "extra_metadata": {},
        }

        # Merge metadata from input jsonl if available.
        if self.jsonl_path.exists():
            extra = self._lookup_jsonl_record(arxiv_id, self.jsonl_path)
            if extra:
                for key in [
                    "citationCount",
                    "influentialCitationCount",
                    "fieldsOfStudy",
                    "published_date",
                    "venue",
                ]:
                    if extra.get(key) is not None:
                        metadata[key] = extra[key]

        write_json(metadata_path, metadata, indent=2)

    def _run_annotation_stage(self, metadata_path: Path) -> None:
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found: {metadata_path}")
        logger.info("Run annotation stage for %s", metadata_path.parent.name)
        meta = read_json(metadata_path)
        ann_sections = self._build_section_annotations(
            meta.get("sections") or [],
            figure_ids={x.get("figure_id", "") for x in meta.get("figures") or []},
            table_ids={x.get("table_id", "") for x in meta.get("tables") or []},
        )
        annotation = {
            "arxiv_id": meta.get("arxiv_id"),
            "sections": ann_sections,
        }
        write_json(metadata_path.parent / "annotation.json", annotation, indent=2)

        if self.config.run_figure_table_semantics:
            self._run_figure_table_semantics(meta, ann_sections, metadata_path.parent)

    def _refresh_semantic_scholar(self, metadata_path: Path, arxiv_id: str) -> None:
        if not metadata_path.exists():
            return
        extra = self.semantic_scholar.fetch_arxiv_metadata(arxiv_id)
        if not extra:
            return
        metadata = read_json(metadata_path)

        for key in [
            "citationCount",
            "influentialCitationCount",
            "fieldsOfStudy",
            "published_date",
            "venue",
        ]:
            value = extra.get(key)
            if value is not None:
                metadata[key] = value

        write_json(metadata_path, metadata, indent=2)
        logger.info("Semantic Scholar metadata refreshed for %s", arxiv_id)

    @staticmethod
    def _load_arxiv_ids_from_jsonl(jsonl_path: Path) -> List[str]:
        ids: List[str] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                aid = (obj.get("arxiv_id") or "").strip()
                if aid:
                    ids.append(aid.split("v")[0])
        return ids

    @staticmethod
    def _lookup_jsonl_record(arxiv_id: str, jsonl_path: Path) -> Optional[Dict[str, Any]]:
        target = arxiv_id.split("v")[0]
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                aid = (obj.get("arxiv_id") or "").split("v")[0]
                if aid == target:
                    return obj
        return None

    @staticmethod
    def _find_main_tex_file(paper_dir: Path) -> Optional[Path]:
        tex_files = sorted(paper_dir.rglob("*.tex"))
        for tex in tex_files:
            text = tex.read_text(encoding="utf-8", errors="ignore")
            if "\\begin{document}" in text:
                return tex
        return tex_files[0] if tex_files else None

    @staticmethod
    def _clean_latex_text(text: str) -> str:
        text = re.sub(r"%.*", "", text)
        text = re.sub(r"\\\w+\*?(\[[^\]]*\])?\{([^{}]*)\}", r"\2", text)
        text = re.sub(r"\\\w+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_title(self, raw: str) -> str:
        m = re.search(r"\\title\{(.+?)\}", raw, flags=re.S)
        if not m:
            return ""
        return self._clean_latex_text(m.group(1))

    def _extract_abstract(self, raw: str) -> str:
        m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", raw, flags=re.S)
        if not m:
            return ""
        return self._clean_latex_text(m.group(1))

    def _extract_authors(self, raw: str) -> List[Dict[str, Any]]:
        m = re.search(r"\\author\{(.+?)\}", raw, flags=re.S)
        if not m:
            return []
        author_blob = self._clean_latex_text(m.group(1))
        names = [x.strip() for x in re.split(r"\band\b|,", author_blob) if x.strip()]
        uniq = []
        seen = set()
        for n in names:
            if n.lower() in seen:
                continue
            seen.add(n.lower())
            uniq.append({"name": n, "affiliations": []})
        return uniq

    def _extract_sections(self, raw: str) -> List[Dict[str, Any]]:
        pattern = re.compile(r"\\(section|subsection|subsubsection)\*?\{([^}]*)\}")
        matches = list(pattern.finditer(raw))
        if not matches:
            return []

        root: List[Dict[str, Any]] = []
        stack: List[Tuple[int, Dict[str, Any]]] = []

        for i, m in enumerate(matches):
            kind = m.group(1)
            title = self._clean_latex_text(m.group(2))
            level = {"section": 1, "subsection": 2, "subsubsection": 3}[kind]
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            content = self._clean_latex_text(raw[start:end])

            node = {
                "title": title,
                "content": content,
                "level": level,
                "children": [],
            }

            while stack and stack[-1][0] >= level:
                stack.pop()
            if not stack:
                root.append(node)
            else:
                stack[-1][1]["children"].append(node)
            stack.append((level, node))

        return root

    def _extract_figures(self, raw: str, paper_dir: Path, arxiv_id: str) -> List[Dict[str, Any]]:
        figures: List[Dict[str, Any]] = []
        fig_blocks = re.finditer(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", raw, flags=re.S)
        out_dir = self.extracted_dir / arxiv_id / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)

        for block in fig_blocks:
            chunk = block.group(1)
            label_m = re.search(r"\\label\{([^}]+)\}", chunk)
            cap_m = re.search(r"\\caption\{?([^}]+)\}?", chunk, flags=re.S)
            inc_m = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", chunk)
            if not label_m:
                continue
            figure_id = label_m.group(1).strip()
            caption = self._clean_latex_text(cap_m.group(1)) if cap_m else ""
            source_path = None
            saved_path = None
            file_type = None
            if inc_m:
                rel = inc_m.group(1).strip()
                source_path, file_type = self._resolve_graphics_path(paper_dir, rel)
            if source_path and source_path.exists():
                target = out_dir / f"{figure_id.replace(':', '_')}{source_path.suffix}"
                try:
                    shutil.copy2(source_path, target)
                    saved_path = target
                except Exception:
                    saved_path = source_path

            figures.append(
                {
                    "figure_id": figure_id,
                    "caption": caption,
                    "file_path": str(saved_path) if saved_path else None,
                    "file_type": file_type,
                }
            )
        return figures

    @staticmethod
    def _resolve_graphics_path(paper_dir: Path, relative: str) -> Tuple[Optional[Path], Optional[str]]:
        candidates = []
        rel_path = Path(relative)
        if rel_path.suffix:
            candidates.append(paper_dir / rel_path)
        else:
            for ext in [".pdf", ".png", ".jpg", ".jpeg"]:
                candidates.append(paper_dir / f"{relative}{ext}")
        for p in candidates:
            if p.exists():
                return p, p.suffix.lstrip(".")
        return None, None

    def _extract_tables(self, raw: str, arxiv_id: str) -> List[Dict[str, Any]]:
        tables: List[Dict[str, Any]] = []
        table_blocks = re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", raw, flags=re.S)
        for block in table_blocks:
            chunk = block.group(0)
            label_m = re.search(r"\\label\{([^}]+)\}", chunk)
            cap_m = re.search(r"\\caption\{([^}]*)\}", chunk, flags=re.S)
            if not label_m:
                continue
            table_id = label_m.group(1).strip()
            caption = self._clean_latex_text(cap_m.group(1)) if cap_m else ""
            tables.append(
                {
                    "table_id": table_id,
                    "caption": caption,
                    "content": {},
                    "latex_content": chunk,
                    "file_path": None,
                    "table_path": None,
                }
            )
        return tables

    @staticmethod
    def _extract_resource_links(raw: str) -> List[Dict[str, str]]:
        urls = set(re.findall(r"https?://[^\s\]\)\}]+", raw))
        links = []
        for url in sorted(urls):
            link_type = "github" if "github.com" in url else "url"
            links.append({
                "url": url,
                "link_type": link_type,
                "description": url,
            })
        return links

    def _build_section_annotations(
        self,
        sections: List[Dict[str, Any]],
        figure_ids: set,
        table_ids: set,
    ) -> List[Dict[str, Any]]:
        out = []
        for sec in sections:
            text = sec.get("content") or ""
            found_figs = sorted([fid for fid in figure_ids if fid and fid in text])
            found_tabs = sorted([tid for tid in table_ids if tid and tid in text])
            out.append(
                {
                    "title": sec.get("title", ""),
                    "level": sec.get("level", 1),
                    "figure_ids": found_figs,
                    "table_ids": found_tabs,
                    "children": self._build_section_annotations(sec.get("children") or [], figure_ids, table_ids),
                }
            )
        return out

    def _run_figure_table_semantics(
        self,
        metadata: Dict[str, Any],
        ann_sections: List[Dict[str, Any]],
        out_dir: Path,
    ) -> None:
        api_key = (self.config.llm_api_key or "").strip()
        items = []

        # Build section lookup for each figure/table.
        refs_by_id: Dict[str, List[str]] = {}
        self._collect_refs_from_annotations(ann_sections, refs_by_id)

        for fig in metadata.get("figures") or []:
            item_id = fig.get("figure_id")
            items.append(
                self._build_semantics_item(
                    item_id=item_id,
                    item_type="figure",
                    caption=fig.get("caption", ""),
                    sections=refs_by_id.get(item_id or "", []),
                    api_key=api_key,
                )
            )

        for tab in metadata.get("tables") or []:
            item_id = tab.get("table_id")
            items.append(
                self._build_semantics_item(
                    item_id=item_id,
                    item_type="table",
                    caption=tab.get("caption", ""),
                    sections=refs_by_id.get(item_id or "", []),
                    api_key=api_key,
                )
            )

        write_json(out_dir / "figure_table_semantics.json", {"items": items}, indent=2)

    def _build_semantics_item(
        self,
        item_id: Optional[str],
        item_type: str,
        caption: str,
        sections: List[str],
        api_key: str,
    ) -> Dict[str, Any]:
        summary = self._heuristic_semantics_summary(item_type, caption)
        keywords = self._extract_keywords(caption)

        if api_key:
            llm_result = self._call_llm_figure_table_semantics(item_type, caption, api_key)
            if llm_result:
                summary = llm_result.get("summary") or summary
                kws = llm_result.get("keywords")
                if isinstance(kws, list) and kws:
                    keywords = kws[:10]

        return {
            "item_id": item_id,
            "item_type": item_type,
            "caption": caption,
            "mentioned_in_sections": sections,
            "semantic_summary": summary,
            "keywords": keywords,
        }

    @staticmethod
    def _collect_refs_from_annotations(nodes: List[Dict[str, Any]], refs: Dict[str, List[str]]) -> None:
        for node in nodes:
            title = node.get("title", "")
            for fid in node.get("figure_ids") or []:
                refs.setdefault(fid, []).append(title)
            for tid in node.get("table_ids") or []:
                refs.setdefault(tid, []).append(title)
            PreprocessPipeline._collect_refs_from_annotations(node.get("children") or [], refs)

    @staticmethod
    def _heuristic_semantics_summary(item_type: str, caption: str) -> str:
        if not caption:
            return f"This {item_type} supports the paper's discussion."
        return f"This {item_type} shows: {caption}"

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
        stop = {
            "the", "and", "for", "with", "from", "this", "that", "are", "was", "were", "into",
            "using", "study", "result", "results", "method", "model", "models",
        }
        uniq = []
        seen = set()
        for t in tokens:
            if t in stop or t in seen:
                continue
            seen.add(t)
            uniq.append(t)
            if len(uniq) >= 10:
                break
        return uniq

    def _call_llm_figure_table_semantics(self, item_type: str, caption: str, api_key: str) -> Optional[Dict[str, Any]]:
        base_url = (self.config.llm_base_url or "https://api.openai.com/v1").rstrip("/")
        model = self.config.llm_model or "gpt-4o-mini"
        url = f"{base_url}/chat/completions"
        prompt = (
            "You are helping with scientific figure/table understanding. "
            "Return STRICT JSON with keys: summary (string), keywords (array of <=10 strings).\n"
            f"Item type: {item_type}\nCaption: {caption}"
        )
        body = {
            "model": model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.llm_timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            return None

        try:
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception:
            return None
