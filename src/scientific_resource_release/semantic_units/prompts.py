from __future__ import annotations

from typing import List, Optional


SYSTEM_PROMPT_FACT_UNITS = """You are an expert assistant for analyzing academic papers.
Your task is to extract atomic fact units from a paper section and any figure/table descriptions provided, and assign each fact a semantic label.

Overall goal: the extracted facts will later support retrieval over scientific content. Focus on technical, conceptual, and empirical information that helps users find and understand the paper.

Requirements for each fact:
1. Each fact must be an independent, self-contained statement in English. Do not use pronouns or vague references such as \"this method\", \"we\", or \"it\"; state the subject explicitly.
2. If the content refers to figures or tables, rewrite that information as complete declarative sentences.
3. Assign exactly one semantic label per fact. Prefer the canonical labels from the rules below; if the fact does not fit any of them, invent a short lowercase label that best describes the semantic type.

Do not extract facts about:
- acknowledgments, funding sources, grants, affiliations, author biographies;
- references or descriptions of other papers;
- boilerplate about licenses, ethics statements, or dataset terms of use, unless they are central to the scientific contribution and likely to be searched for.

Semantic label rules:
- Preferred canonical labels: problem, definition, method, evaluation, experiment, result, contribution.
- Use canonical labels only when the fact clearly fits.
- When none of the above fits well, invent a new short lowercase label such as dataset, limitation, future_work, ablation, application, or assumption.

Ignore Related Work sections. Do not extract descriptions of other papers; only extract content about the current paper and its contributions.

Output format: JSON only.
{
  \"facts\": [
    { \"statement\": \"<one atomic fact in English>\", \"semantic_label\": \"<canonical label or your own short lowercase label>\" }
  ]
}"""


SYSTEM_PROMPT_SEMANTIC_UNIT = """You are an expert at summarizing academic content.
Given a list of atomic facts that have been grouped by semantic similarity, produce one semantic unit for retrieval and display.

Output a single JSON object with exactly these keys:
- semantic_role: A short label indicating the type of semantic unit, such as method, result, contribution, dataset, or limitation. If a suggested_semantic_role is provided, treat it as a hint and refine it only when the facts clearly support a better label.
- title: A short descriptive title suitable as a card title or section heading.
- content: A single coherent paragraph in English that merges the facts without repetition.
- keywords: A list of 3 to 8 English keywords or short phrases.

Do not add any extra keys. Output only the JSON object."""


SYSTEM_PROMPT_LABEL_MERGE = """You are an expert at normalizing semantic labels from academic paper fact extraction.
Given a list of unique semantic labels, merge only clearly synonymous or very similar labels while preserving genuinely distinct semantic types.

Canonical categories: problem, definition, method, evaluation, experiment, result, contribution.

Rules:
1. Map each input label to exactly one output category using short lowercase English words.
2. Labels already in the canonical set usually map to themselves; merge obvious synonyms into these categories.
3. If a label expresses a distinct semantic type such as dataset, limitation, future_work, application, or theory, keep it as its own category.
4. Do not map meaningful labels to other. Use other only for vague or noisy labels.
5. You do not need to force every canonical category to appear.
6. Output only a JSON object: { \"mapping\": { \"<input_label>\": \"<output_category>\" } }."""


def build_fact_units_user_prompt(
    section_title: str,
    section_content: str,
    figure_semantics: List[str],
    table_semantics: List[str],
) -> str:
    parts = [
        "Section title: %s\n" % section_title,
        "Section content (English):\n%s\n" % section_content,
    ]
    if figure_semantics or table_semantics:
        parts.append("Figure and table semantic descriptions:\n")
        for index, summary in enumerate(figure_semantics + table_semantics, 1):
            parts.append("%d. %s\n" % (index, summary))
    parts.append(
        "\nExtract all atomic facts and assign each a semantic_label. "
        "Use canonical labels only when the fact clearly belongs there; otherwise invent a specific short lowercase label. "
        "Output only a JSON object with a facts array; each element has statement and semantic_label."
    )
    return "\n".join(parts)


def build_label_merge_user_prompt(unique_labels: List[str]) -> str:
    labels_str = ", ".join(sorted(set(label.strip() for label in unique_labels if label and label.strip())))
    return (
        "You are given semantic labels extracted from a single paper. Merge obviously synonymous labels while preserving labels that represent distinct semantic types.\n\n"
        "Canonical categories for reference: problem, definition, method, evaluation, experiment, result, contribution.\n"
        "If a label is clearly equivalent to one of these canonical types, map it there. "
        "If a label represents a specific semantic type not covered well by the canonical list, keep it as a custom category.\n"
        "Do not force every label into the canonical list. Use other only for vague or uninformative labels.\n\n"
        "Input labels: %s\n\n"
        "Output one JSON object with key mapping: an object mapping each input label string to its chosen category string."
    ) % labels_str


def build_semantic_unit_user_prompt(
    fact_statements: List[str],
    suggested_semantic_role: Optional[str] = None,
) -> str:
    bullets = "\n".join("- %s" % statement for statement in fact_statements)
    role_hint = ""
    if suggested_semantic_role:
        role_hint = "\nSuggested semantic_role for this group (use or refine): %s." % suggested_semantic_role
    return (
        "Merge the following atomic facts into one semantic unit with one theme, one paragraph, plus title and keywords:%s\n\n%s\n\n"
        "Produce exactly one JSON object with keys: semantic_role, title, content, keywords."
    ) % (role_hint, bullets)