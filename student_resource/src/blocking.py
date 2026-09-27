"""
Multi-Channel Candidate Generation (Blocking) with Frequency Caps.
"""
from __future__ import annotations
import duckdb

# Legal suffix regex pattern (no Devanagari - those are in different scripts and won't match)
LEGAL_REGEX = (
    "(^| )(private|pvt|limited|ltd|incorporated|inc|corporation|corp|company|co|"
    "llc|llp|plc|services|holdings|enterprises|group|solutions|industries|"
    "associates|partners|management|international|intl|"
    "sarl|sas|sasu|eurl|snc|sci|gie|ei|ste|societe|"
    "mr|mrs|ms|dr|shri|smt|shree)( |$)"
)


def create_normalized_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    source: str,
    country: str | None = None,
) -> None:
    """Normalize a source table or TSV file into a blocking-ready table."""
    if source.endswith(".tsv"):
        from_clause = f"read_csv_auto('{source}', delim='\\t', header=true)"
    else:
        from_clause = source

    country_filter = f"WHERE country = '{country}'" if country else ""

    # Step 1: basic normalization
    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT
            entity_id, country, business_name, business_address,
            strip_accents(lower(trim(regexp_replace(
                regexp_replace(coalesce(business_name,''), '[^\\p{{L}}\\p{{M}}\\p{{N}}\\s]',' ','g'),
                '\\s+',' ','g')))) AS norm_name,
            strip_accents(lower(trim(regexp_replace(
                regexp_replace(coalesce(business_address,''), '[^\\p{{L}}\\p{{M}}\\p{{N}}\\s]',' ','g'),
                '\\s+',' ','g')))) AS norm_addr,
            regexp_extract_all(
                strip_accents(lower(coalesce(business_address,''))), '[0-9]+'
            ) AS addr_nums,
            list_filter(
                string_split(strip_accents(lower(trim(regexp_replace(
                    regexp_replace(coalesce(business_address,''), '[^\\p{{L}}\\p{{M}}\\p{{N}}\\s]',' ','g'),
                    '\\s+',' ','g')))), ' '),
                x -> length(x) >= 3
            ) AS addr_tokens
        FROM {from_clause}
        {country_filter}
    """)

    # Step 2: derived keys (core_name, sorted, spaceless, etc.)
    legal = LEGAL_REGEX
    con.execute(f"""
        ALTER TABLE {table_name} ADD COLUMN core_name VARCHAR;
        ALTER TABLE {table_name} ADD COLUMN spaceless_core VARCHAR;
        ALTER TABLE {table_name} ADD COLUMN sorted_core VARCHAR;
        ALTER TABLE {table_name} ADD COLUMN first_2_tokens VARCHAR;
        ALTER TABLE {table_name} ADD COLUMN first_token VARCHAR;
        ALTER TABLE {table_name} ADD COLUMN first_num VARCHAR;
        ALTER TABLE {table_name} ADD COLUMN postal_code VARCHAR;
    """)

    con.execute(f"""
        UPDATE {table_name}
        SET core_name = trim(regexp_replace(
            regexp_replace(
                regexp_replace(norm_name,
                    '(\\bwww\\.|\\.com|\\.in|\\.org|\\.net|\\.co|\\.fr|\\.gov)\\b', ' ', 'g'),
                '{legal}', ' ', 'g'),
            '\\s+', ' ', 'g'))
    """)

    con.execute(f"""
        UPDATE {table_name}
        SET
            spaceless_core = regexp_replace(core_name, '\\s+', '', 'g'),
            sorted_core = array_to_string(
                list_sort(list_distinct(list_filter(string_split(core_name, ' '), x -> length(x) >= 2))), ' '),
            first_2_tokens = CASE
                WHEN len(list_filter(string_split(core_name, ' '), x -> length(x) >= 2)) >= 2
                THEN (list_filter(string_split(core_name, ' '), x -> length(x) >= 2))[1]
                     || ' '
                     || (list_filter(string_split(core_name, ' '), x -> length(x) >= 2))[2]
                ELSE core_name END,
            first_token = CASE
                WHEN len(list_filter(string_split(core_name, ' '), x -> length(x) >= 2)) >= 1
                THEN (list_filter(string_split(core_name, ' '), x -> length(x) >= 2))[1]
                ELSE '' END,
            first_num = CASE WHEN len(addr_nums) > 0 THEN addr_nums[1] ELSE '' END,
            postal_code = regexp_extract(norm_addr, '\\b[0-9]{{3,6}}\\b')
    """)


def generate_candidates(
    con: duckdb.DuckDBPyConnection,
    source_table: str,
    target_table: str,
    output_table: str,
    max_freq: int = 35,
) -> None:
    """Generate candidate pairs via 7 equi-join channels with frequency capping."""
    con.execute(f"""
        CREATE OR REPLACE TABLE {output_table} AS
        WITH
        name_freq AS (
            SELECT country, norm_name, count(*) AS cnt FROM {target_table}
            WHERE norm_name != '' GROUP BY country, norm_name HAVING count(*) <= {max_freq}
        ),
        core_freq AS (
            SELECT country, core_name, count(*) AS cnt FROM {target_table}
            WHERE length(core_name) >= 3 GROUP BY country, core_name HAVING count(*) <= {max_freq}
        ),
        sorted_freq AS (
            SELECT country, sorted_core, count(*) AS cnt FROM {target_table}
            WHERE length(sorted_core) >= 3 GROUP BY country, sorted_core HAVING count(*) <= {max_freq}
        ),
        spaceless_freq AS (
            SELECT country, spaceless_core, count(*) AS cnt FROM {target_table}
            WHERE length(spaceless_core) >= 4 GROUP BY country, spaceless_core HAVING count(*) <= {max_freq}
        ),
        first2_freq AS (
            SELECT country, first_2_tokens, count(*) AS cnt FROM {target_table}
            WHERE length(first_2_tokens) >= 5 GROUP BY country, first_2_tokens HAVING count(*) <= {max_freq}
        ),
        addr_freq AS (
            SELECT country, norm_addr, count(*) AS cnt FROM {target_table}
            WHERE length(norm_addr) >= 6 GROUP BY country, norm_addr HAVING count(*) <= {max_freq}
        ),
        post_token_freq AS (
            SELECT country, postal_code, first_token, count(*) AS cnt FROM {target_table}
            WHERE postal_code != '' AND length(first_token) >= 3
            GROUP BY country, postal_code, first_token HAVING count(*) <= {max_freq}
        )
        SELECT DISTINCT s.entity_id AS source1_entity_id, t.entity_id AS matched_entity_id
        FROM (
            SELECT s.entity_id, t.entity_id FROM {source_table} s
            JOIN name_freq f USING (country, norm_name) JOIN {target_table} t USING (country, norm_name)
            UNION ALL
            SELECT s.entity_id, t.entity_id FROM {source_table} s
            JOIN core_freq f USING (country, core_name) JOIN {target_table} t USING (country, core_name)
            UNION ALL
            SELECT s.entity_id, t.entity_id FROM {source_table} s
            JOIN sorted_freq f USING (country, sorted_core) JOIN {target_table} t USING (country, sorted_core)
            UNION ALL
            SELECT s.entity_id, t.entity_id FROM {source_table} s
            JOIN spaceless_freq f USING (country, spaceless_core) JOIN {target_table} t USING (country, spaceless_core)
            UNION ALL
            SELECT s.entity_id, t.entity_id FROM {source_table} s
            JOIN first2_freq f USING (country, first_2_tokens) JOIN {target_table} t USING (country, first_2_tokens)
            UNION ALL
            SELECT s.entity_id, t.entity_id FROM {source_table} s
            JOIN addr_freq f USING (country, norm_addr) JOIN {target_table} t USING (country, norm_addr)
            UNION ALL
            SELECT s.entity_id, t.entity_id FROM {source_table} s
            JOIN post_token_freq f USING (country, postal_code, first_token)
            JOIN {target_table} t USING (country, postal_code, first_token)
        ) pairs(entity_id, entity_id_1)
        JOIN {source_table} s ON s.entity_id = pairs.entity_id
        JOIN {target_table} t ON t.entity_id = pairs.entity_id_1
    """)
