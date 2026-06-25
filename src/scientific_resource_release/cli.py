from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from scientific_resource_release.models import PreprocessConfig, SemanticUnitConfig
from scientific_resource_release.preprocess.pipeline import PreprocessPipeline
from scientific_resource_release.semantic_units.pipeline import SemanticUnitReleasePipeline
from scientific_resource_release.utils.logging_utils import setup_logging


logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ScientificResource release pipelines")
    sub = parser.add_subparsers(dest="command", required=True)

    preprocess = sub.add_parser("preprocess", help="Run data preprocessing")
    preprocess_sub = preprocess.add_subparsers(dest="mode", required=True)

    p_one = preprocess_sub.add_parser("one", help="Preprocess one paper")
    _add_preprocess_common_args(p_one)
    p_one.add_argument("--arxiv-id", required=True, type=str)

    p_batch = preprocess_sub.add_parser("batch", help="Preprocess batch papers")
    _add_preprocess_common_args(p_batch)
    p_batch.add_argument("--jsonl", type=Path, default=None)

    sem = sub.add_parser("semantic-units", help="Generate semantic units")
    sem_sub = sem.add_subparsers(dest="mode", required=True)

    s_one = sem_sub.add_parser("one", help="Generate semantic units for one paper")
    _add_semantic_common_args(s_one)
    s_one.add_argument("--arxiv-id", required=True, type=str)

    s_batch = sem_sub.add_parser("batch", help="Generate semantic units in batch")
    _add_semantic_common_args(s_batch)

    retrieval = sub.add_parser("retrieval", help="Run retrieval ingestion/search")
    retrieval_sub = retrieval.add_subparsers(dest="mode", required=True)

    r_ingest = retrieval_sub.add_parser("ingest", help="Ingest extracted json into pgvector tables")
    r_ingest.add_argument("--data-root", type=Path, required=True)
    r_ingest.add_argument("--arxiv-ids", nargs="*", default=None)
    r_ingest.add_argument("--skip-existing-semantic-units", action="store_true")
    r_ingest.add_argument("--skip-existing-title-abstract-embedding", action="store_true")
    r_ingest.add_argument("--skip-existing-fulltext-chunks", action="store_true")
    r_ingest.add_argument("--fulltext-chunk-size", type=int, default=256)

    r_search = retrieval_sub.add_parser("search", help="Search with hybrid-doc-semantic and optional knowledge outputs")
    r_search.add_argument("--query", type=str, required=True)
    r_search.add_argument("--top-k", type=int, default=10)
    r_search.add_argument("--sparse-weight", type=float, default=0.5)
    r_search.add_argument("--doc-level-weight", type=float, default=0.7)
    r_search.add_argument("--semantic-top-n", type=int, default=2)
    r_search.add_argument("--unit-source", choices=["semantic_units", "chunks"], default="semantic_units")
    r_search.add_argument("--intent-decomposer", choices=["llm", "relational", "lexical", "none"], default="llm")
    r_search.add_argument("--with-knowledge", action="store_true")
    r_search.add_argument("--reason-top-k", type=int, default=5)

    return parser


def _add_preprocess_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--latex-dir", type=Path, default=None)
    parser.add_argument("--pdf-dir", type=Path, default=None)
    parser.add_argument("--extracted-dir", type=Path, default=None)
    parser.add_argument("--no-annotation", action="store_true")
    parser.add_argument("--no-figure-table-semantics", action="store_true")
    parser.add_argument("--no-semantic-scholar", action="store_true")
    parser.add_argument("--no-skip-done", action="store_true")
    parser.add_argument("--llm-api-key", type=str, default=None)
    parser.add_argument("--llm-base-url", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default=None)


def _add_semantic_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--extracted-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cluster-method", choices=["label_based", "kmeans", "llm"], default="label_based")
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--no-figure-semantics", action="store_true")
    parser.add_argument("--rule-merge", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--llm-api-key", type=str, default=None)
    parser.add_argument("--llm-base-url", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default=None)


def main() -> None:
    setup_logging()
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        cfg = PreprocessConfig(
            project_root=args.project_root,
            jsonl_path=getattr(args, "jsonl", None),
            latex_dir=args.latex_dir,
            pdf_dir=args.pdf_dir,
            extracted_dir=args.extracted_dir,
            run_annotation=not args.no_annotation,
            run_figure_table_semantics=not args.no_figure_table_semantics,
            refresh_semantic_scholar=not args.no_semantic_scholar,
            skip_done=not args.no_skip_done,
            llm_api_key=args.llm_api_key,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
        )
        pipeline = PreprocessPipeline(cfg)
        if args.mode == "one":
            out = pipeline.process_one(args.arxiv_id)
            logger.info("Preprocess finished: %s", out)
            return
        outputs = pipeline.process_batch_from_jsonl()
        logger.info("Batch preprocess finished: %d papers", len(outputs))
        return

    if args.command == "semantic-units":
        cfg = SemanticUnitConfig(
            project_root=args.project_root,
            extracted_dir=args.extracted_dir,
            output_dir=args.output_dir,
            use_figure_semantics=not args.no_figure_semantics,
            cluster_method=args.cluster_method,
            n_clusters=args.n_clusters,
            use_llm_label_merge=not args.rule_merge,
            skip_existing=not args.no_skip_existing,
            llm_api_key=args.llm_api_key,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
        )
        pipeline = SemanticUnitReleasePipeline(cfg)
        if args.mode == "one":
            out = pipeline.run_one(args.arxiv_id)
            logger.info("Semantic units finished: %s", out)
            return
        outputs, ids = pipeline.run_batch_from_extracted()
        logger.info("Batch semantic units finished: %d/%d papers", len(outputs), len(ids))
        return

    if args.mode == "ingest":
        from scientific_resource_release.retrieval import run_ingestion

        run_ingestion(
            data_root=args.data_root,
            arxiv_ids=args.arxiv_ids or None,
            skip_existing_semantic_units=args.skip_existing_semantic_units,
            skip_existing_title_abstract_embedding=args.skip_existing_title_abstract_embedding,
            skip_existing_fulltext_chunks=args.skip_existing_fulltext_chunks,
            fulltext_chunk_size=args.fulltext_chunk_size,
        )
        return

    from scientific_resource_release.retrieval import RetrievalKnowledgeService
    from scientific_resource_release.retrieval.schemas import SearchRequest

    service = RetrievalKnowledgeService()
    request = SearchRequest(
        query=args.query,
        top_k=args.top_k,
        sparse_weight=args.sparse_weight,
        doc_level_weight=args.doc_level_weight,
        semantic_top_n=args.semantic_top_n,
        unit_source=args.unit_source,
        intent_decomposer=args.intent_decomposer,
    )
    if args.with_knowledge:
        payload = service.search_with_knowledge(request, top_k=args.top_k, reason_top_k=args.reason_top_k)
    else:
        payload = service.search(request)
    logger.info("%s", json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
