from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any, Generator, List, Optional

from sqlalchemy import Date, DateTime, ForeignKey, Index, Integer, String, Text, create_engine, func, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

try:
    from pgvector.sqlalchemy import Vector

    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False
    Vector = None  # type: ignore


DEFAULT_DATABASE_URL = "postgresql://localhost:5432/scientific_resource"


class Base(DeclarativeBase):
    pass


class PaperMetadata(Base):
    __tablename__ = "papers_metadata"

    arxiv_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    abstract: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    authors: Mapped[Any] = mapped_column(JSONB, default=list, nullable=False)
    published_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    fields_of_study: Mapped[List[str]] = mapped_column(ARRAY(Text), default=list, nullable=False)
    venue: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    citation_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[List["SemanticChunk"]] = relationship(
        "SemanticChunk", back_populates="paper", cascade="all, delete-orphan"
    )
    doc_embedding: Mapped[Optional["PaperDenseEmbedding"]] = relationship(
        "PaperDenseEmbedding", back_populates="paper", cascade="all, delete-orphan", uselist=False
    )


class SemanticChunk(Base):
    __tablename__ = "semantic_chunks"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    arxiv_id: Mapped[str] = mapped_column(String(32), ForeignKey("papers_metadata.arxiv_id", ondelete="CASCADE"))
    role: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    keywords: Mapped[List[str]] = mapped_column(ARRAY(Text), default=list, nullable=False)
    embedding: Mapped[List[float]] = mapped_column(
        Vector(1024) if HAS_PGVECTOR and Vector else None, nullable=True  # type: ignore
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    paper: Mapped["PaperMetadata"] = relationship("PaperMetadata", back_populates="chunks")


class PaperDenseEmbedding(Base):
    __tablename__ = "paper_dense_embeddings"

    arxiv_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("papers_metadata.arxiv_id", ondelete="CASCADE"), primary_key=True
    )
    embedding: Mapped[List[float]] = mapped_column(
        Vector(1024) if HAS_PGVECTOR and Vector else None, nullable=True  # type: ignore
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    paper: Mapped["PaperMetadata"] = relationship("PaperMetadata", back_populates="doc_embedding")


class PaperFulltextChunk(Base):
    __tablename__ = "paper_fulltext_chunks"

    chunk_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    arxiv_id: Mapped[str] = mapped_column(String(32), ForeignKey("papers_metadata.arxiv_id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    section_title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[List[float]] = mapped_column(
        Vector(1024) if HAS_PGVECTOR and Vector else None, nullable=True  # type: ignore
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_paper_fulltext_chunks_arxiv_id", "arxiv_id"),
        Index("idx_paper_fulltext_chunks_arxiv_chunk", "arxiv_id", "chunk_index", unique=True),
    )


def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_engine(database_url: Optional[str] = None):
    return create_engine(database_url or get_database_url(), echo=False)


def init_db(engine=None) -> None:
    eng = engine or get_engine()
    with eng.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    Base.metadata.create_all(eng)

    with eng.connect() as conn:
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_semantic_chunks_embedding
                ON semantic_chunks
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_papers_metadata_fts ON papers_metadata
                USING GIN (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(abstract, '')))
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_paper_dense_embeddings_embedding
                ON paper_dense_embeddings
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_paper_fulltext_chunks_embedding
                ON paper_fulltext_chunks
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )
        )
        conn.commit()


def get_session_factory(engine=None):
    eng = engine or get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=eng)


def get_session() -> Generator[Session, None, None]:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()
