from __future__ import annotations

import csv
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "test"
OUTPUT = ROOT / "output"


def load_table(con: duckdb.DuckDBPyConnection, name: str, filename: str) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {name} AS
        SELECT
            entity_id,
            business_name,
            business_address,
            country,
            lower(trim(regexp_replace(regexp_replace(coalesce(business_name, ''), '[^\\p{{L}}\\p{{M}}\\p{{N}}\\s]', ' ', 'g'), '\\s+', ' ', 'g'))) AS normalized_name,
            lower(trim(regexp_replace(regexp_replace(coalesce(business_address, ''), '[^\\p{{L}}\\p{{M}}\\p{{N}}\\s]', ' ', 'g'), '\\s+', ' ', 'g'))) AS normalized_address
        FROM read_csv_auto(?, delim='\\t', header=true)
        """,
        [str(DATASET / filename)],
    )
    con.execute(
        f"""
        ALTER TABLE {name} ADD COLUMN core_name VARCHAR;
        UPDATE {name}
        SET core_name = trim(regexp_replace(
            normalized_name,
            '(^| )(private|pvt|limited|ltd|incorporated|inc|corporation|corp|company|co|llc|llp|plc|sarl|sas|sasu|sa|eurl|snc|प्राइवेट|लिमिटेड|एलएलसी|एलएलपी)( |$)',
            ' ', 'g'))
        """,
    )


def export_rows(con: duckdb.DuckDBPyConnection, table: str, column: str, filename: str) -> None:
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / filename
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", column])
        source_cursor = con.cursor()
        source_cursor.execute("SELECT entity_id FROM test_source1 ORDER BY entity_id")
        match_cursor = con.cursor()
        match_cursor.execute(f"SELECT source1_entity_id, matched_entity_id FROM {table} ORDER BY source1_entity_id, matched_entity_id")
        next_match = match_cursor.fetchone()
        source_row = source_cursor.fetchone()
        while source_row is not None:
            source_id = source_row[0]
            values = []
            while next_match is not None and next_match[0] < source_id:
                next_match = match_cursor.fetchone()
            while next_match is not None and next_match[0] == source_id:
                values.append(next_match[1])
                next_match = match_cursor.fetchone()
            writer.writerow([source_id, ",".join(values)])
            source_row = source_cursor.fetchone()


def main() -> None:
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='3GB'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET temp_directory='duckdb_tmp'")
    load_table(con, "test_source1", "test_source1.tsv")
    load_table(con, "test_source2", "test_source2.tsv")
    load_table(con, "test_source3", "test_source3.tsv")
    for target in ("test_source2", "test_source3"):
        suffix = target[-2:]
        con.execute(
            f"""
            CREATE OR REPLACE TABLE candidates_{suffix} AS
            SELECT DISTINCT s.entity_id AS source1_entity_id, t.entity_id AS matched_entity_id
                        FROM test_source1 s JOIN {target} t
                            ON s.country = t.country AND s.normalized_name != ''
                         AND s.normalized_name = t.normalized_name
                        UNION
                        SELECT s.entity_id, t.entity_id
                        FROM test_source1 s JOIN {target} t
                            ON s.country = t.country AND s.core_name != ''
                         AND s.core_name = t.core_name
                        UNION
                        SELECT s.entity_id, t.entity_id
                        FROM test_source1 s JOIN {target} t
                            ON s.country = t.country AND s.normalized_address != ''
                         AND s.normalized_address = t.normalized_address
            """
        )
    con.execute("CREATE OR REPLACE TABLE candidate_all AS SELECT * FROM candidates_e2 UNION ALL SELECT * FROM candidates_e3")
    con.execute("CREATE OR REPLACE TABLE final_all AS SELECT * FROM candidate_all")
    export_rows(con, "final_all", "matched_entity_ids", "matching_results.tsv")
    export_rows(con, "candidate_all", "candidate_entity_ids", "candidate_pairs.tsv")
    print("quick output generation complete")


if __name__ == "__main__":
    main()
