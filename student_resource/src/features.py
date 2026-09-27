"""
Feature Extraction for Business Entity Resolution.
Computes rich string, phonetic, token, and numeric address similarity features in DuckDB.
"""
from __future__ import annotations

import duckdb

FEATURE_COLUMNS = [
    "exact_norm_name",
    "exact_core_name",
    "exact_sorted_core",
    "exact_spaceless_core",
    "exact_norm_addr",
    "name_jaro",
    "core_jaro",
    "sorted_jaro",
    "addr_jaro",
    "core_lev_sim",
    "addr_num_overlap",
    "has_num_overlap",
    "first_num_match",
    "name_token_jaccard",
    "addr_token_jaccard",
    "name_overlap",
    "addr_overlap",
]


def create_features(
    con: duckdb.DuckDBPyConnection,
    source_table: str,
    target_table: str,
    candidate_table: str,
    output_table: str,
    include_labels: bool = False,
    ground_truth_table: str = "ground_truth",
) -> None:
    """Compute pairwise similarity features for candidate pairs in DuckDB."""
    label_expr = ""
    if include_labels:
        label_expr = f""",
            CASE WHEN EXISTS (
                SELECT 1 FROM {ground_truth_table} gt
                WHERE gt.source1_entity_id = prep.source1_entity_id
                  AND gt.matched_entity_id = prep.matched_entity_id
            ) THEN 1 ELSE 0 END AS label
        """

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {output_table} AS
        WITH prep AS (
            SELECT
                c.source1_entity_id,
                c.matched_entity_id,
                s.norm_name AS s_name,
                t.norm_name AS t_name,
                s.core_name AS s_core,
                t.core_name AS t_core,
                s.sorted_core AS s_sorted,
                t.sorted_core AS t_sorted,
                s.spaceless_core AS s_spaceless,
                t.spaceless_core AS t_spaceless,
                s.norm_addr AS s_addr,
                t.norm_addr AS t_addr,
                s.addr_nums AS s_nums,
                t.addr_nums AS t_nums,
                s.first_num AS s_first_num,
                t.first_num AS t_first_num,
                list_filter(string_split(s.core_name, ' '), x -> length(x) >= 2) AS s_core_toks,
                list_filter(string_split(t.core_name, ' '), x -> length(x) >= 2) AS t_core_toks,
                s.addr_tokens AS s_addr_toks,
                t.addr_tokens AS t_addr_toks
            FROM {candidate_table} c
            JOIN {source_table} s ON s.entity_id = c.source1_entity_id
            JOIN {target_table} t ON t.entity_id = c.matched_entity_id
        )
        SELECT
            source1_entity_id,
            matched_entity_id,
            (s_name != '' AND s_name = t_name)::DOUBLE AS exact_norm_name,
            (s_core != '' AND s_core = t_core)::DOUBLE AS exact_core_name,
            (s_sorted != '' AND s_sorted = t_sorted)::DOUBLE AS exact_sorted_core,
            (s_spaceless != '' AND s_spaceless = t_spaceless)::DOUBLE AS exact_spaceless_core,
            (s_addr != '' AND s_addr = t_addr)::DOUBLE AS exact_norm_addr,
            jaro_winkler_similarity(s_name, t_name) AS name_jaro,
            jaro_winkler_similarity(s_core, t_core) AS core_jaro,
            jaro_winkler_similarity(s_sorted, t_sorted) AS sorted_jaro,
            jaro_winkler_similarity(s_addr, t_addr) AS addr_jaro,
            1.0 - (levenshtein(s_core, t_core) / greatest(length(s_core), length(t_core), 1)::DOUBLE) AS core_lev_sim,
            len(list_intersect(s_nums, t_nums))::DOUBLE AS addr_num_overlap,
            CASE WHEN len(list_intersect(s_nums, t_nums)) > 0 THEN 1.0 ELSE 0.0 END AS has_num_overlap,
            CASE WHEN s_first_num != '' AND s_first_num = t_first_num THEN 1.0 ELSE 0.0 END AS first_num_match,
            len(list_intersect(s_core_toks, t_core_toks))::DOUBLE / greatest(list_unique(list_concat(s_core_toks, t_core_toks)), 1) AS name_token_jaccard,
            len(list_intersect(s_addr_toks, t_addr_toks))::DOUBLE / greatest(list_unique(list_concat(s_addr_toks, t_addr_toks)), 1) AS addr_token_jaccard,
            len(list_intersect(s_core_toks, t_core_toks))::DOUBLE AS name_overlap,
            len(list_intersect(s_addr_toks, t_addr_toks))::DOUBLE AS addr_overlap
            {label_expr}
        FROM prep
        """
    )
