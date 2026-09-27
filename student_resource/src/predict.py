"""
End-to-End Pipeline for Business Entity Resolution.
1. Trains HistGradientBoostingClassifier on training ground truth.
2. Calibrates optimal threshold optimizing macro F0.5 score.
3. Performs country-partitioned candidate generation, feature extraction, and competitive matching.
4. Generates output/matching_results.tsv and output/candidate_pairs.tsv.
5. Runs submission validation.
"""
from __future__ import annotations

import csv
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
OUTPUT = ROOT / "output"

sys.path.insert(0, str(ROOT / "src"))
from blocking import create_normalized_table, generate_candidates
from features import FEATURE_COLUMNS, create_features
from model import calibrate_threshold, train_classifier


def train_and_calibrate(con: duckdb.DuckDBPyConnection) -> tuple[object, float]:
    """Train classifier on sample of training set and calibrate threshold on held-out split."""
    print("\n--- Step 1: Training Model and Calibrating Threshold ---")
    t0 = time.time()
    train_dir = DATASET / "train"

    # Select 25,000 S1 entities for training & calibration
    con.execute(
        """
        CREATE OR REPLACE TABLE tr_s1_sample AS
        SELECT * FROM read_csv_auto(?, delim='\t', header=true)
        USING SAMPLE reservoir(25000 ROWS) REPEATABLE (42)
        """,
        [str(train_dir / "train_source1.tsv")],
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE tr_gt_sample AS
        SELECT g.source1_entity_id, trim(unnest(string_split(g.matched_entity_ids, ','))) AS m_id
        FROM read_csv_auto(?, delim='\t', header=true) g
        JOIN tr_s1_sample s ON s.entity_id = g.source1_entity_id
        WHERE g.matched_entity_ids IS NOT NULL AND trim(g.matched_entity_ids) != ''
        """,
        [str(train_dir / "train_ground_truth.tsv")],
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE tr_s2_sample AS
        SELECT * FROM read_csv_auto(?, delim='\t', header=true) s
        WHERE EXISTS (SELECT 1 FROM tr_gt_sample g WHERE g.m_id = s.entity_id)
        UNION
        (SELECT * FROM read_csv_auto(?, delim='\t', header=true) USING SAMPLE reservoir(75000 ROWS) REPEATABLE (42))
        """,
        [str(train_dir / "train_source2.tsv"), str(train_dir / "train_source2.tsv")],
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE tr_s3_sample AS
        SELECT * FROM read_csv_auto(?, delim='\t', header=true) s
        WHERE EXISTS (SELECT 1 FROM tr_gt_sample g WHERE g.m_id = s.entity_id)
        UNION
        (SELECT * FROM read_csv_auto(?, delim='\t', header=true) USING SAMPLE reservoir(75000 ROWS) REPEATABLE (42))
        """,
        [str(train_dir / "train_source3.tsv"), str(train_dir / "train_source3.tsv")],
    )

    # Normalize sampled tables directly
    create_normalized_table(con, "tr_s1", "tr_s1_sample")
    create_normalized_table(con, "tr_s2", "tr_s2_sample")
    create_normalized_table(con, "tr_s3", "tr_s3_sample")

    con.execute("CREATE OR REPLACE TABLE ground_truth AS SELECT source1_entity_id, m_id AS matched_entity_id FROM tr_gt_sample")

    # Generate candidates
    generate_candidates(con, "tr_s1", "tr_s2", "tr_cand_s2", max_freq=35)
    generate_candidates(con, "tr_s1", "tr_s3", "tr_cand_s3", max_freq=35)

    # Compute features with labels
    create_features(con, "tr_s1", "tr_s2", "tr_cand_s2", "tr_feat_s2", include_labels=True)
    create_features(con, "tr_s1", "tr_s3", "tr_cand_s3", "tr_feat_s3", include_labels=True)

    df_s2 = con.execute("SELECT * FROM tr_feat_s2").df()
    df_s3 = con.execute("SELECT * FROM tr_feat_s3").df()

    s1_all = con.execute("SELECT DISTINCT entity_id FROM tr_s1").df()["entity_id"].tolist()
    np.random.seed(42)
    np.random.shuffle(s1_all)
    split = int(len(s1_all) * 0.6)
    train_s1_set = set(s1_all[:split])
    val_s1_set = set(s1_all[split:])

    train_df = pd.concat([
        df_s2[df_s2["source1_entity_id"].isin(train_s1_set)],
        df_s3[df_s3["source1_entity_id"].isin(train_s1_set)]
    ])

    print(f"Training pairs: {len(train_df):,} (Positives: {train_df['label'].sum():,})")
    clf = train_classifier(train_df)

    # Calibrate threshold on held-out validation entities
    val_df_s2 = df_s2[df_s2["source1_entity_id"].isin(val_s1_set)]
    val_df_s3 = df_s3[df_s3["source1_entity_id"].isin(val_s1_set)]

    gt_df = con.execute("SELECT * FROM ground_truth").df()
    gt_val_df = gt_df[gt_df["source1_entity_id"].isin(val_s1_set)]
    val_gt_dict = {
        s1_id: set(grp["matched_entity_id"].values)
        for s1_id, grp in gt_val_df.groupby("source1_entity_id")
    }

    best_th, val_score = calibrate_threshold(
        val_df_s2, val_df_s3, val_gt_dict, list(val_s1_set), clf
    )
    print(f"Optimal Threshold: {best_th:.2f} | Held-out Validation Macro F0.5: {val_score:.6f}")
    print(f"Training and calibration completed in {time.time()-t0:.2f}s")

    # Clean up training tables
    con.execute("""
        DROP TABLE IF EXISTS tr_s1_sample;
        DROP TABLE IF EXISTS tr_gt_sample;
        DROP TABLE IF EXISTS tr_s2_sample;
        DROP TABLE IF EXISTS tr_s3_sample;
        DROP TABLE IF EXISTS tr_s1;
        DROP TABLE IF EXISTS tr_s2;
        DROP TABLE IF EXISTS tr_s3;
        DROP TABLE IF EXISTS tr_cand_s2;
        DROP TABLE IF EXISTS tr_cand_s3;
        DROP TABLE IF EXISTS tr_feat_s2;
        DROP TABLE IF EXISTS tr_feat_s3;
        DROP TABLE IF EXISTS ground_truth;
    """)
    gc.collect()

    return clf, best_th


def process_country_test(
    con: duckdb.DuckDBPyConnection,
    country: str,
    clf: object,
    threshold: float,
    all_candidates_dict: dict[str, list[str]],
    all_matches_dict: dict[str, list[str]],
) -> None:
    """Process one country partition for the test set."""
    print(f"\n--- Processing Test Set Partition: {country} ---")
    t0 = time.time()
    test_dir = DATASET / "test"

    # 1. Load normalized tables for this country
    print(f"[{country}] Normalizing source tables...")
    create_normalized_table(con, "ts1", str(test_dir / "test_source1.tsv"), country=country)
    create_normalized_table(con, "ts2", str(test_dir / "test_source2.tsv"), country=country)
    create_normalized_table(con, "ts3", str(test_dir / "test_source3.tsv"), country=country)

    n_s1 = con.execute("SELECT count(*) FROM ts1").fetchone()[0]
    n_s2 = con.execute("SELECT count(*) FROM ts2").fetchone()[0]
    n_s3 = con.execute("SELECT count(*) FROM ts3").fetchone()[0]
    print(f"[{country}] Loaded S1={n_s1:,}, S2={n_s2:,}, S3={n_s3:,}")

    # 2. Candidate Generation (Blocking)
    print(f"[{country}] Generating candidates for S2 and S3...")
    generate_candidates(con, "ts1", "ts2", "cand_s2", max_freq=35)
    generate_candidates(con, "ts1", "ts3", "cand_s3", max_freq=35)

    cand_s2_cnt = con.execute("SELECT count(*) FROM cand_s2").fetchone()[0]
    cand_s3_cnt = con.execute("SELECT count(*) FROM cand_s3").fetchone()[0]
    print(f"[{country}] Candidates: S2={cand_s2_cnt:,}, S3={cand_s3_cnt:,}")

    # 3. Feature Extraction
    print(f"[{country}] Computing pairwise features...")
    create_features(con, "ts1", "ts2", "cand_s2", "feat_s2", include_labels=False)
    create_features(con, "ts1", "ts3", "cand_s3", "feat_s3", include_labels=False)

    # 4. Predict & Competitive 1-to-1 Matching
    for target_src, feat_table in [("S2", "feat_s2"), ("S3", "feat_s3")]:
        print(f"[{country}] Scoring and competitive matching for {target_src}...")
        df = con.execute(f"SELECT * FROM {feat_table}").df()
        if len(df) == 0:
            continue

        # Record candidates into candidate dictionary
        for s1_id, grp in df.groupby("source1_entity_id"):
            all_candidates_dict.setdefault(s1_id, []).extend(grp["matched_entity_id"].tolist())

        # Predict match probabilities
        df["prob"] = clf.predict_proba(df[FEATURE_COLUMNS].fillna(0.0))[:, 1]

        # Filter by threshold and enforce 1-to-1 assignment per target entity
        matched = df[df["prob"] >= threshold].sort_values("prob", ascending=False).drop_duplicates(subset=["matched_entity_id"])

        # Record matches
        for s1_id, grp in matched.groupby("source1_entity_id"):
            all_matches_dict.setdefault(s1_id, []).extend(grp["matched_entity_id"].tolist())

        del df, matched
        gc.collect()

    # Clean up country tables
    con.execute("""
        DROP TABLE IF EXISTS ts1;
        DROP TABLE IF EXISTS ts2;
        DROP TABLE IF EXISTS ts3;
        DROP TABLE IF EXISTS cand_s2;
        DROP TABLE IF EXISTS cand_s3;
        DROP TABLE IF EXISTS feat_s2;
        DROP TABLE IF EXISTS feat_s3;
    """)
    gc.collect()
    print(f"[{country}] Completed in {time.time()-t0:.2f}s")


def main() -> None:
    t_start = time.time()
    OUTPUT.mkdir(exist_ok=True)
    temp_dir = ROOT / "duckdb_tmp"
    temp_dir.mkdir(exist_ok=True)

    con = duckdb.connect()
    con.execute("SET threads = 4")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET memory_limit = '8GB'")
    con.execute(f"SET temp_directory = '{temp_dir.as_posix()}'")

    # Step 1: Train & Calibrate
    clf, threshold = train_and_calibrate(con)

    # Step 2: Process Test Set by Country
    all_candidates: dict[str, list[str]] = {}
    all_matches: dict[str, list[str]] = {}

    for country in ["France", "US", "India"]:
        process_country_test(
            con, country, clf, threshold, all_candidates, all_matches
        )

    # Step 3: Stream and write final submission files
    print("\n--- Step 3: Writing Final Output Files ---")
    test_s1_path = DATASET / "test" / "test_source1.tsv"

    # Read all test S1 IDs in their sorted order
    s1_ids = []
    with test_s1_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if row and row[0].strip():
                s1_ids.append(row[0].strip())

    s1_ids = sorted(set(s1_ids))
    print(f"Total Source 1 test entities: {len(s1_ids):,}")

    match_out = OUTPUT / "matching_results.tsv"
    cand_out = OUTPUT / "candidate_pairs.tsv"

    print("Writing output/matching_results.tsv...")
    with match_out.open("w", encoding="utf-8", newline="") as f_m:
        writer_m = csv.writer(f_m, delimiter="\t", lineterminator="\n")
        writer_m.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id in s1_ids:
            matches = list(dict.fromkeys(all_matches.get(s1_id, [])))
            writer_m.writerow([s1_id, ",".join(matches)])

    print("Writing output/candidate_pairs.tsv...")
    with cand_out.open("w", encoding="utf-8", newline="") as f_c:
        writer_c = csv.writer(f_c, delimiter="\t", lineterminator="\n")
        writer_c.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id in s1_ids:
            # Candidates must include all matches to guarantee matches are a subset of candidates
            cands = list(dict.fromkeys(all_candidates.get(s1_id, []) + all_matches.get(s1_id, [])))
            writer_c.writerow([s1_id, ",".join(cands)])

    print(f"Successfully wrote matching_results.tsv ({match_out.stat().st_size / (1024**2):.2f} MB)")
    print(f"Successfully wrote candidate_pairs.tsv ({cand_out.stat().st_size / (1024**2):.2f} MB)")

    # Step 4: Validate submission files
    print("\n--- Step 4: Validating Submission Files ---")
    val_cmd = [
        sys.executable,
        str(ROOT / "utils" / "validate_submission.py"),
        "--matching", str(match_out),
        "--candidate", str(cand_out),
        "--test-dir", str(DATASET / "test"),
    ]
    res = subprocess.run(val_cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)

    print(f"\nAll tasks complete in {time.time()-t_start:.2f}s!")


if __name__ == "__main__":
    main()
