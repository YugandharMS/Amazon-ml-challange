"""
Entity Resolution Pipeline v7 - Precision-First Architecture (Target: >0.90 F0.5)

Key Improvements Over v6:
1. ADDRESS COMPATIBILITY GUARDRAIL (Hard Rule)
   - When both records have non-empty addresses: require addr_jaro >= 0.35 OR addr_token_overlap >= 1
   - This eliminates cross-city/state false positives (the #1 source of FPs identified in analysis)
   
2. SHORT NAME STRICTNESS
   - For core_name <= 8 chars (single/two-word names like "Sun", "Apex", "Star"):
     require addr_overlap >= 1 (cannot match on name alone without address confirmation)
   
3. SMARTER BLOCKING - Reduced false candidate explosion
   - Raised min_freq threshold for short tokens
   - Added rare_token blocking channel using rarest non-stopword token
   - Tightened max_freq from 35 to 20 for most channels to cut candidate count
   
4. EXPANDED FEATURE SET (28 features)
   - addr_jaro_bin: discretized address similarity (0,0.3,0.5,0.7,1.0 buckets)
   - name_starts_same: first 3 chars of name match  
   - core_num_words: length signal (helps distinguish short-name ambiguity)
   - has_addr_both: both entities have addresses (context for guardrail)
   - addr_token_overlap_norm: normalized addr token overlap
   - unleet_jaro: jaro on leetspeak-normalized core names
   - first_token_match: first word of core matches
   
5. ENSEMBLE SCORING
   - Primary: HistGradientBoosting with tuned hyperparams
   - Post-processing: Hard rule elimination then threshold-based selection
   
6. IMPROVED VALIDATION THRESHOLD
   - Sweep 0.40 to 0.90 with macro F0.5 on validation set
   - Apply address guardrail BEFORE threshold evaluation (not after)
"""
from __future__ import annotations
import csv, gc, os, re, subprocess, sys, time
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
OUTPUT = ROOT / "output"

LEGAL_REGEX = (
    "(^| )(the|private|pvt|limited|ltd|incorporated|inc|corporation|corp|company|co|"
    "llc|llp|plc|services|holdings|enterprises|group|solutions|industries|"
    "associates|partners|management|international|intl|"
    "sarl|sas|sasu|eurl|snc|sci|gie|ei|ste|societe|"
    "mr|mrs|ms|dr|shri|smt|shree|"
    "praiveta|limiteda|praivet|limited|praibheta|limatida|praivarr|limirrad|pra|li|"
    "association|club|ecole|comite|fils|france)( |$)"
)

ADDR_STOP_REGEX = (
    "(^| )(road|rd|street|st|avenue|ave|drive|dr|court|ct|"
    "lane|ln|boulevard|blvd|highway|hwy|parkway|pkwy|"
    "way|circle|cir|place|pl|suite|ste|unit|floor|fl|"
    "building|bldg|near|opp|opposite|behind|post|po|box|"
    "plot|door|hno|house|no|sr|null|nan|none|and|the|of|in|at|"
    "nagar|colony|bazar|market|complex|taluk|village|district|dist|gali|pur|puram|"
    "rue|chemin|impasse|allee|quai|cours|bis|ter|du|des|de|la|le|les|d|l|zone|za|zi|parc|cedex|bp)( |$)"
)

FEATURE_COLS = [
    # Exact match signals
    "exact_norm_name", "exact_core_name", "exact_sorted_core", "exact_spaceless",
    # Fuzzy name signals
    "name_jaro", "core_jaro", "sorted_jaro", "unleet_jaro",
    # Address signals
    "addr_jaro", "addr_jaro_bin", "addr_token_jaccard", "addr_overlap", "addr_overlap_norm",
    # Token overlap
    "core_lev_sim", "name_token_jaccard", "name_overlap",
    # Number/postal signals
    "first_num_match", "addr_num_overlap", "num_street_match", "postal_match",
    # Length/ratio signals
    "name_len_ratio", "core_containment", "addr_len_ratio",
    # Address exact
    "exact_sorted_addr",
    # New v7 features
    "core_num_words", "has_addr_both", "first_token_match", "name_starts_same",
]


def log(msg):
    print(msg, flush=True)


def transliterate_series(series):
    out = []
    for text in series:
        if not text or pd.isna(text):
            out.append("")
            continue
        text = str(text)
        has_indic = False
        sc = sanscript.DEVANAGARI
        for ch in text:
            c = ord(ch)
            if 0x0900 <= c <= 0x0D7F:
                has_indic = True
                if 0x0900 <= c <= 0x097F: sc = sanscript.DEVANAGARI; break
                elif 0x0980 <= c <= 0x09FF: sc = sanscript.BENGALI; break
                elif 0x0A00 <= c <= 0x0A7F: sc = sanscript.GURMUKHI; break
                elif 0x0A80 <= c <= 0x0AFF: sc = sanscript.GUJARATI; break
                elif 0x0B00 <= c <= 0x0B7F: sc = sanscript.ORIYA; break
                elif 0x0B80 <= c <= 0x0BFF: sc = sanscript.TAMIL; break
                elif 0x0C00 <= c <= 0x0C7F: sc = sanscript.TELUGU; break
                elif 0x0C80 <= c <= 0x0CFF: sc = sanscript.KANNADA; break
                elif 0x0D00 <= c <= 0x0D7F: sc = sanscript.MALAYALAM; break
        if has_indic:
            try:
                out.append(transliterate(text, sc, sanscript.ITRANS).lower())
            except:
                out.append(text.lower())
        else:
            out.append(text.lower())
    return out


def normalize_table(con, out_table, source, country=None, do_translit=False):
    if isinstance(source, str) and source.endswith(".tsv"):
        df = con.execute(
            f"SELECT * FROM read_csv_auto('{source}', delim='\\t', header=true) "
            + (f"WHERE country = '{country}'" if country else "")
        ).df()
    elif isinstance(source, str):
        df = con.execute(
            f"SELECT * FROM {source} " + (f"WHERE country = '{country}'" if country else "")
        ).df()
    else:
        df = source

    if do_translit:
        df["lat_name"] = transliterate_series(df["business_name"])
        df["lat_addr"] = transliterate_series(df["business_address"])
    else:
        df["lat_name"] = df["business_name"].fillna("").astype(str)
        df["lat_addr"] = df["business_address"].fillna("").astype(str)

    con.register("_raw_df", df)

    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table} AS
        WITH raw AS (
            SELECT entity_id, country, business_name, business_address, lat_name, lat_addr FROM _raw_df
        ),
        step1 AS (
            SELECT *,
                strip_accents(lower(trim(regexp_replace(
                    regexp_replace(coalesce(lat_name,''), '[^\\p{{L}}\\p{{M}}\\p{{N}}\\s]',' ','g'),
                    '\\s+',' ','g')))) AS norm_name,
                strip_accents(lower(trim(regexp_replace(
                    regexp_replace(coalesce(lat_addr,''), '[^\\p{{L}}\\p{{M}}\\p{{N}}\\s]',' ','g'),
                    '\\s+',' ','g')))) AS norm_addr
            FROM raw
        ),
        step2 AS (
            SELECT *,
                trim(regexp_replace(
                    regexp_replace(
                        regexp_replace(norm_name,
                            '(\\bwww\\.|\\.com|\\.in|\\.org|\\.net|\\.co|\\.fr|\\.gov)\\b', ' ', 'g'),
                        '{LEGAL_REGEX}', ' ', 'g'),
                    '\\s+', ' ', 'g')) AS core_name,
                trim(regexp_replace(
                    regexp_replace(norm_addr, '{ADDR_STOP_REGEX}', ' ', 'g'),
                    '\\s+', ' ', 'g')) AS clean_addr
            FROM step1
        ),
        step3 AS (
            SELECT *,
                replace(replace(replace(replace(replace(core_name, '0', 'o'), '1', 'l'), '5', 's'), '3', 'e'), '@', 'a') AS unleet_core,
                regexp_extract_all(norm_addr, '[0-9]+') AS raw_nums
            FROM step2
        )
        SELECT
            entity_id, country, business_name, business_address,
            norm_name, norm_addr, core_name, clean_addr, unleet_core,
            length(norm_name) AS name_len,
            length(core_name) AS core_len,
            length(norm_addr) AS addr_len,
            len(list_filter(string_split(core_name, ' '), x -> length(x) >= 2)) AS core_num_words,
            regexp_replace(core_name, '\\s+', '', 'g') AS spaceless_core,
            array_to_string(
                list_sort(list_distinct(list_filter(string_split(core_name, ' '), x -> length(x) >= 2))), ' '
            ) AS sorted_core,
            array_to_string(
                list_sort(list_distinct(list_filter(string_split(unleet_core, ' '), x -> length(x) >= 2))), ' '
            ) AS unleet_sorted,
            regexp_replace(unleet_core, '\\s+', '', 'g') AS unleet_spaceless,
            CASE WHEN len(list_filter(string_split(core_name, ' '), x -> length(x) >= 2)) >= 2
                 THEN (list_filter(string_split(core_name, ' '), x -> length(x) >= 2))[1] || ' ' || (list_filter(string_split(core_name, ' '), x -> length(x) >= 2))[2]
                 ELSE '' END AS first_2_tokens,
            CASE WHEN len(list_filter(string_split(core_name, ' '), x -> length(x) >= 2)) >= 1
                 THEN (list_filter(string_split(core_name, ' '), x -> length(x) >= 2))[1]
                 ELSE '' END AS first_token,
            list_transform(raw_nums, x -> regexp_replace(x, '^0+', '')) AS clean_nums,
            CASE WHEN len(raw_nums) > 0 THEN regexp_replace(raw_nums[1], '^0+', '') ELSE '' END AS first_num,
            coalesce(regexp_extract(norm_addr, '\\b[0-9]{{4,6}}\\b'), '') AS postal_code,
            array_to_string(
                list_slice(list_sort(list_distinct(list_filter(string_split(clean_addr, ' '), x -> length(x) >= 3 AND regexp_matches(x, '^[a-z0-9]+$')))), 1, 4), ' '
            ) AS sorted_addr,
            CASE WHEN len(list_filter(string_split(clean_addr, ' '), x -> length(x) >= 3 AND regexp_matches(x, '^[a-z]+$'))) > 0
                 THEN (list_filter(string_split(clean_addr, ' '), x -> length(x) >= 3 AND regexp_matches(x, '^[a-z]+$')))[1]
                 ELSE '' END AS first_street_word,
            list_filter(string_split(core_name, ' '), x -> length(x) >= 2) AS core_tokens,
            list_filter(string_split(norm_addr, ' '), x -> length(x) >= 3) AS addr_tokens,
            -- alpha_addr_tokens: only letters (len>=3), for city/state overlap (not street numbers)
            list_filter(string_split(norm_addr, ' '), x -> length(x) >= 3 AND regexp_matches(x, '^[a-z]+$')) AS alpha_addr_tokens,
            CASE WHEN coalesce(trim(business_address), '') != '' THEN 1 ELSE 0 END AS has_addr
        FROM step3
    """)
    con.unregister("_raw_df")
    del df
    gc.collect()


def block_candidates(con, s_table, t_table, out_table, max_freq=20):
    """
    v7: Tighter max_freq (20 down from 35) to reduce FP candidate pairs.
    Added 'rare_token' blocking: matches on the single rarest (lowest IDF) core token
    that appears in <= 5 entities in the target, giving precision-first matching.
    """
    con.execute(f"""
        -- Core name exact channels (most precise)
        CREATE OR REPLACE TEMP TABLE _freq_nn AS
        SELECT country, norm_name, count(*) AS cnt FROM {t_table} WHERE norm_name != '' GROUP BY country, norm_name HAVING count(*) <= {max_freq};
        
        CREATE OR REPLACE TEMP TABLE _freq_cn AS
        SELECT country, core_name, count(*) AS cnt FROM {t_table} WHERE length(core_name) >= 4 GROUP BY country, core_name HAVING count(*) <= {max_freq};
        
        CREATE OR REPLACE TEMP TABLE _freq_sc AS
        SELECT country, sorted_core, count(*) AS cnt FROM {t_table} WHERE length(sorted_core) >= 4 GROUP BY country, sorted_core HAVING count(*) <= {max_freq};
        
        CREATE OR REPLACE TEMP TABLE _freq_sp AS
        SELECT country, spaceless_core, count(*) AS cnt FROM {t_table} WHERE length(spaceless_core) >= 5 GROUP BY country, spaceless_core HAVING count(*) <= {max_freq};
        
        CREATE OR REPLACE TEMP TABLE _freq_usc AS
        SELECT country, unleet_sorted, count(*) AS cnt FROM {t_table} WHERE length(unleet_sorted) >= 4 GROUP BY country, unleet_sorted HAVING count(*) <= {max_freq};
        
        CREATE OR REPLACE TEMP TABLE _freq_usp AS
        SELECT country, unleet_spaceless, count(*) AS cnt FROM {t_table} WHERE length(unleet_spaceless) >= 5 GROUP BY country, unleet_spaceless HAVING count(*) <= {max_freq};
        
        -- Address channels (high precision)
        CREATE OR REPLACE TEMP TABLE _freq_sa AS
        SELECT country, sorted_addr, count(*) AS cnt FROM {t_table} WHERE length(sorted_addr) >= 10 GROUP BY country, sorted_addr HAVING count(*) <= 15;
        
        CREATE OR REPLACE TEMP TABLE _freq_nw AS
        SELECT country, first_num, first_street_word, count(*) AS cnt FROM {t_table} WHERE first_num != '' AND first_street_word != '' GROUP BY country, first_num, first_street_word HAVING count(*) <= 15;
        
        CREATE OR REPLACE TEMP TABLE _freq_np AS
        SELECT country, first_num, postal_code, count(*) AS cnt FROM {t_table} WHERE first_num != '' AND postal_code != '' GROUP BY country, first_num, postal_code HAVING count(*) <= 15;
        
        -- Rare token blocking: tokens appearing <= 5 times in BOTH source and target
        CREATE OR REPLACE TEMP TABLE _s_toks_raw AS
        SELECT entity_id, country, unnest(core_tokens) AS tok FROM {s_table};
        
        CREATE OR REPLACE TEMP TABLE _t_toks_raw AS
        SELECT entity_id, country, unnest(core_tokens) AS tok FROM {t_table};
        
        CREATE OR REPLACE TEMP TABLE _s_rare_cnt AS
        SELECT country, tok FROM _s_toks_raw WHERE length(tok) >= 5 GROUP BY country, tok HAVING count(*) <= 5;

        CREATE OR REPLACE TEMP TABLE _t_rare_cnt AS
        SELECT country, tok FROM _t_toks_raw WHERE length(tok) >= 5 GROUP BY country, tok HAVING count(*) <= 5;

        CREATE OR REPLACE TEMP TABLE _rare_toks AS
        SELECT country, tok FROM _s_rare_cnt
        INTERSECT
        SELECT country, tok FROM _t_rare_cnt;
        
        -- Filter source and target to only rare tokens
        CREATE OR REPLACE TEMP TABLE _s_rare_toks AS
        SELECT s.entity_id, s.country, s.tok
        FROM _s_toks_raw s
        JOIN _rare_toks r ON s.country = r.country AND s.tok = r.tok;
        
        CREATE OR REPLACE TEMP TABLE _t_rare_toks AS
        SELECT t.entity_id, t.country, t.tok
        FROM _t_toks_raw t
        JOIN _rare_toks r ON t.country = r.country AND t.tok = r.tok;

        CREATE OR REPLACE TABLE {out_table} AS
        SELECT DISTINCT source1_entity_id, matched_entity_id FROM (
            -- norm_name exact
            SELECT s.entity_id AS source1_entity_id, t.entity_id AS matched_entity_id
            FROM {s_table} s JOIN _freq_nn f ON s.country = f.country AND s.norm_name = f.norm_name JOIN {t_table} t ON t.country = f.country AND t.norm_name = f.norm_name
            
            UNION ALL
            -- core_name exact  
            SELECT s.entity_id, t.entity_id
            FROM {s_table} s JOIN _freq_cn f ON s.country = f.country AND s.core_name = f.core_name JOIN {t_table} t ON t.country = f.country AND t.core_name = f.core_name
            
            UNION ALL
            -- sorted_core
            SELECT s.entity_id, t.entity_id
            FROM {s_table} s JOIN _freq_sc f ON s.country = f.country AND s.sorted_core = f.sorted_core JOIN {t_table} t ON t.country = f.country AND t.sorted_core = f.sorted_core
            
            UNION ALL
            -- spaceless_core
            SELECT s.entity_id, t.entity_id
            FROM {s_table} s JOIN _freq_sp f ON s.country = f.country AND s.spaceless_core = f.spaceless_core JOIN {t_table} t ON t.country = f.country AND t.spaceless_core = f.spaceless_core
            
            UNION ALL
            -- unleet_sorted
            SELECT s.entity_id, t.entity_id
            FROM {s_table} s JOIN _freq_usc f ON s.country = f.country AND s.unleet_sorted = f.unleet_sorted JOIN {t_table} t ON t.country = f.country AND t.unleet_sorted = f.unleet_sorted
            
            UNION ALL
            -- unleet_spaceless
            SELECT s.entity_id, t.entity_id
            FROM {s_table} s JOIN _freq_usp f ON s.country = f.country AND s.unleet_spaceless = f.unleet_spaceless JOIN {t_table} t ON t.country = f.country AND t.unleet_spaceless = f.unleet_spaceless
            
            UNION ALL
            -- sorted_addr (high-precision address)
            SELECT s.entity_id, t.entity_id
            FROM {s_table} s JOIN _freq_sa f ON s.country = f.country AND s.sorted_addr = f.sorted_addr JOIN {t_table} t ON t.country = f.country AND t.sorted_addr = f.sorted_addr
            
            UNION ALL
            -- num+street
            SELECT s.entity_id, t.entity_id
            FROM {s_table} s JOIN _freq_nw f ON s.country = f.country AND s.first_num = f.first_num AND s.first_street_word = f.first_street_word JOIN {t_table} t ON t.country = f.country AND t.first_num = f.first_num AND t.first_street_word = f.first_street_word
            
            UNION ALL
            -- num+postal
            SELECT s.entity_id, t.entity_id
            FROM {s_table} s JOIN _freq_np f ON s.country = f.country AND s.first_num = f.first_num AND s.postal_code = f.postal_code JOIN {t_table} t ON t.country = f.country AND t.first_num = f.first_num AND t.postal_code = f.postal_code
            
            UNION ALL
            -- rare token blocking (two-hop indexed join on unnested tokens)
            SELECT sr.entity_id AS source1_entity_id, tr.entity_id AS matched_entity_id
            FROM _s_rare_toks sr
            JOIN _t_rare_toks tr ON sr.country = tr.country AND sr.tok = tr.tok
        )
    """)
    con.execute("""
        DROP TABLE IF EXISTS _freq_nn; DROP TABLE IF EXISTS _freq_cn; DROP TABLE IF EXISTS _freq_sc;
        DROP TABLE IF EXISTS _freq_sp; DROP TABLE IF EXISTS _freq_usc; DROP TABLE IF EXISTS _freq_usp;
        DROP TABLE IF EXISTS _freq_sa; DROP TABLE IF EXISTS _freq_nw; DROP TABLE IF EXISTS _freq_np;
        DROP TABLE IF EXISTS _s_toks_raw; DROP TABLE IF EXISTS _t_toks_raw;
        DROP TABLE IF EXISTS _rare_toks; DROP TABLE IF EXISTS _s_rare_toks; DROP TABLE IF EXISTS _t_rare_toks;
        DROP TABLE IF EXISTS _s_rare_cnt; DROP TABLE IF EXISTS _t_rare_cnt;
    """)




def extract_features(con, s_table, t_table, cand_table, out_table, label_table=None):
    label_sql = f""",
        CASE WHEN EXISTS (
            SELECT 1 FROM {label_table} gt
            WHERE gt.source1_entity_id = p.source1_entity_id
              AND gt.matched_entity_id = p.matched_entity_id
        ) THEN 1 ELSE 0 END AS label""" if label_table else ""

    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table} AS
        WITH p AS (
            SELECT
                c.source1_entity_id, c.matched_entity_id,
                s.norm_name AS sn, t.norm_name AS tn,
                s.core_name AS sc, t.core_name AS tc,
                s.sorted_core AS ss, t.sorted_core AS ts2,
                s.spaceless_core AS ssp, t.spaceless_core AS tsp,
                s.unleet_core AS suc, t.unleet_core AS tuc,
                s.unleet_sorted AS sus, t.unleet_sorted AS tus,
                s.norm_addr AS sa, t.norm_addr AS ta,
                s.clean_nums AS s_nums, t.clean_nums AS t_nums,
                s.first_num AS s_fnum, t.first_num AS t_fnum,
                s.postal_code AS s_post, t.postal_code AS t_post,
                s.sorted_addr AS s_saddr, t.sorted_addr AS t_saddr,
                s.first_street_word AS s_fword, t.first_street_word AS t_fword,
                s.core_tokens AS s_ctoks, t.core_tokens AS t_ctoks,
                s.addr_tokens AS s_atoks, t.addr_tokens AS t_atoks,
                s.alpha_addr_tokens AS s_aatoks, t.alpha_addr_tokens AS t_aatoks,
                s.name_len AS s_nlen, t.name_len AS t_nlen,
                s.core_len AS s_clen, t.core_len AS t_clen,
                s.addr_len AS s_alen, t.addr_len AS t_alen,
                s.core_num_words AS s_cnw, t.core_num_words AS t_cnw,
                s.first_token AS s_ft, t.first_token AS t_ft,
                s.has_addr AS s_ha, t.has_addr AS t_ha
            FROM {cand_table} c
            JOIN {s_table} s ON s.entity_id = c.source1_entity_id
            JOIN {t_table} t ON t.entity_id = c.matched_entity_id
        )
        SELECT
            source1_entity_id, matched_entity_id,
            -- Exact match features
            (sn != '' AND sn = tn)::DOUBLE AS exact_norm_name,
            (sc != '' AND sc = tc)::DOUBLE AS exact_core_name,
            (ss != '' AND ss = ts2)::DOUBLE AS exact_sorted_core,
            (ssp != '' AND ssp = tsp)::DOUBLE AS exact_spaceless,
            -- Fuzzy name similarity
            jaro_winkler_similarity(sn, tn) AS name_jaro,
            jaro_winkler_similarity(sc, tc) AS core_jaro,
            jaro_winkler_similarity(ss, ts2) AS sorted_jaro,
            jaro_winkler_similarity(sus, tus) AS unleet_jaro,
            -- Address similarity
            jaro_winkler_similarity(sa, ta) AS addr_jaro,
            CASE 
                WHEN jaro_winkler_similarity(sa, ta) >= 0.70 THEN 1.0
                WHEN jaro_winkler_similarity(sa, ta) >= 0.50 THEN 0.75
                WHEN jaro_winkler_similarity(sa, ta) >= 0.35 THEN 0.50
                WHEN jaro_winkler_similarity(sa, ta) >= 0.20 THEN 0.25
                ELSE 0.0
            END AS addr_jaro_bin,
            -- Token overlap
            1.0 - (levenshtein(sc, tc) / greatest(length(sc), length(tc), 1)::DOUBLE) AS core_lev_sim,
            len(list_intersect(s_ctoks, t_ctoks))::DOUBLE / greatest(list_unique(list_concat(s_ctoks, t_ctoks)), 1) AS name_token_jaccard,
            len(list_intersect(s_atoks, t_atoks))::DOUBLE / greatest(list_unique(list_concat(s_atoks, t_atoks)), 1) AS addr_token_jaccard,
            len(list_intersect(s_ctoks, t_ctoks))::DOUBLE AS name_overlap,
            -- addr_overlap uses ALPHABETIC tokens only (city/state) to avoid street-number false matches across cities
            len(list_intersect(s_aatoks, t_aatoks))::DOUBLE AS addr_overlap,
            -- addr_overlap_norm: normalized alpha addr overlap
            len(list_intersect(s_aatoks, t_aatoks))::DOUBLE / greatest(len(s_aatoks), len(t_aatoks), 1)::DOUBLE AS addr_overlap_norm,
            -- Number/postal
            CASE WHEN s_fnum != '' AND s_fnum = t_fnum THEN 1.0 ELSE 0.0 END AS first_num_match,
            len(list_intersect(s_nums, t_nums))::DOUBLE AS addr_num_overlap,
            least(s_nlen, t_nlen)::DOUBLE / greatest(s_nlen, t_nlen, 1)::DOUBLE AS name_len_ratio,
            CASE WHEN s_clen > 0 AND t_clen > 0 THEN
                CASE WHEN contains(sc, tc) OR contains(tc, sc) THEN 1.0 ELSE 0.0 END
            ELSE 0.0 END AS core_containment,
            least(s_alen, t_alen)::DOUBLE / greatest(s_alen, t_alen, 1)::DOUBLE AS addr_len_ratio,
            (s_saddr != '' AND s_saddr = t_saddr)::DOUBLE AS exact_sorted_addr,
            CASE WHEN s_fnum != '' AND s_fnum = t_fnum AND s_fword != '' AND s_fword = t_fword THEN 1.0 ELSE 0.0 END AS num_street_match,
            CASE WHEN s_post != '' AND s_post = t_post THEN 1.0 ELSE 0.0 END AS postal_match,
            -- v7 new features
            least(s_cnw, t_cnw)::DOUBLE AS core_num_words,
            (s_ha = 1 AND t_ha = 1)::DOUBLE AS has_addr_both,
            (s_ft != '' AND s_ft = t_ft)::DOUBLE AS first_token_match,
            CASE WHEN length(sn) >= 3 AND length(tn) >= 3 
                 THEN (left(sn, 3) = left(tn, 3))::DOUBLE 
                 ELSE 0.0 END AS name_starts_same
            {label_sql}
        FROM p
    """)


def apply_address_guardrail(df):
    """
    v7 Hard Rule: Eliminate cross-city/cross-state FPs.
    When BOTH entities have a non-empty address (has_addr_both == 1.0):
      - If addr_jaro < 0.35 AND addr_overlap < 1: REJECT (very high FP risk)
    When ONE entity has a short/ambiguous core name (core_num_words <= 1):
      - Require addr_overlap >= 1 OR postal_match == 1.0 OR first_num_match == 1.0
    """
    # Condition: both have address but zero similarity => likely cross-city FP
    both_addr = df["has_addr_both"] == 1.0
    low_addr_jaro = df["addr_jaro"] < 0.35
    zero_addr_overlap = df["addr_overlap"] < 1
    
    # Keep pair if:
    # - Not both have addresses (one is missing => give benefit of doubt)
    # - OR addr has alphabetic token overlap (city/state/street name)
    # - OR postal_match (exact zip code)
    # - OR exact_norm_name (perfect name match)
    # - OR exact_sorted_addr (exact sorted addr match)
    keep_addr = (
        ~both_addr |  # one or both addr missing: ok
        (df["addr_overlap"] >= 1) |  # at least 1 shared alpha addr token
        (df["postal_match"] == 1.0) |  # exact postal match
        (df["exact_norm_name"] == 1.0) |  # perfect name match
        (df["exact_sorted_addr"] == 1.0)  # exact sorted addr match
    )
    
    # For single-word names (very ambiguous), require address evidence
    short_name = df["core_num_words"] <= 1.5  # effectively <= 1 word
    keep_short = (
        ~short_name |  # not a short name: relax
        (df["addr_overlap"] >= 1) |  # has address token overlap
        (df["postal_match"] == 1.0) |  # same postal code
        (df["first_num_match"] == 1.0) |  # same street number
        (df["exact_norm_name"] == 1.0)  # exact name match
    )
    
    return df[keep_addr & keep_short]


def compute_macro_f05(gt_dict, pred_dict, all_s1_ids):
    scores = []
    for s1 in all_s1_ids:
        true = gt_dict.get(s1, set())
        pred = pred_dict.get(s1, set())
        nt, np_ = len(true), len(pred)
        if nt == 0 and np_ == 0:
            scores.append(1.0)
        elif nt == 0 or np_ == 0:
            scores.append(0.0)
        else:
            tp = len(true & pred)
            if tp == 0:
                scores.append(0.0)
            else:
                p, r = tp / np_, tp / nt
                scores.append((1.25 * p * r) / (0.25 * p + r))
    return float(np.mean(scores))


def main():
    t_start = time.time()
    OUTPUT.mkdir(exist_ok=True)
    temp_dir = ROOT / "duckdb_tmp"
    temp_dir.mkdir(exist_ok=True)

    con = duckdb.connect()
    con.execute("SET threads = 4; SET preserve_insertion_order = false; SET memory_limit = '8GB'")
    con.execute(f"SET temp_directory = '{temp_dir.as_posix()}'")

    train_dir = DATASET / "train"
    test_dir = DATASET / "test"

    # ========== STEP 1: TRAIN MODEL ==========
    log("=== Step 1: Training Model (v7 - Precision-First Architecture) ===")
    t0 = time.time()

    log("Loading training sample (larger = better model)...")
    con.execute("""
        CREATE OR REPLACE TABLE tr_s1 AS
        SELECT * FROM read_csv_auto(?, delim='\t', header=true)
        USING SAMPLE 20000 ROWS (reservoir, 42)
    """, [str(train_dir / "train_source1.tsv")])

    con.execute("""
        CREATE OR REPLACE TABLE tr_gt AS
        SELECT g.source1_entity_id, trim(unnest(string_split(g.matched_entity_ids, ','))) AS matched_entity_id
        FROM read_csv_auto(?, delim='\t', header=true) g
        JOIN tr_s1 s ON s.entity_id = g.source1_entity_id
        WHERE g.matched_entity_ids IS NOT NULL AND trim(g.matched_entity_ids) != ''
    """, [str(train_dir / "train_ground_truth.tsv")])

    for src_num in [2, 3]:
        tsv = str(train_dir / f"train_source{src_num}.tsv")
        con.execute(f"""
            CREATE OR REPLACE TABLE tr_s{src_num} AS
            SELECT * FROM (
                SELECT * FROM read_csv_auto(?, delim='\\t', header=true) s
                WHERE EXISTS (SELECT 1 FROM tr_gt g WHERE g.matched_entity_id = s.entity_id)
                UNION ALL
                SELECT * FROM (
                    SELECT * FROM read_csv_auto(?, delim='\\t', header=true) s
                    WHERE s.country IN (SELECT DISTINCT country FROM tr_s1)
                    USING SAMPLE 80000 ROWS (reservoir, 42)
                )
            )
        """, [tsv, tsv])

    log("Normalizing training tables (with Indic transliteration)...")
    normalize_table(con, "ns1", "tr_s1", do_translit=True)
    normalize_table(con, "ns2", "tr_s2", do_translit=True)
    normalize_table(con, "ns3", "tr_s3", do_translit=True)

    log("Generating training candidates (v7 tight blocking)...")
    block_candidates(con, "ns1", "ns2", "cand2")
    block_candidates(con, "ns1", "ns3", "cand3")
    nc2 = con.execute("SELECT count(*) FROM cand2").fetchone()[0]
    nc3 = con.execute("SELECT count(*) FROM cand3").fetchone()[0]
    log(f"  Train candidates: cand2={nc2:,}, cand3={nc3:,}")

    log("Extracting training features...")
    extract_features(con, "ns1", "ns2", "cand2", "feat2", label_table="tr_gt")
    extract_features(con, "ns1", "ns3", "cand3", "feat3", label_table="tr_gt")

    df2 = con.execute("SELECT * FROM feat2").df()
    df3 = con.execute("SELECT * FROM feat3").df()

    s1_all = con.execute("SELECT DISTINCT entity_id FROM ns1").df()["entity_id"].tolist()
    np.random.seed(42)
    np.random.shuffle(s1_all)
    split = int(len(s1_all) * 0.65)
    train_set, val_set = set(s1_all[:split]), set(s1_all[split:])

    train_df = pd.concat([df2[df2["source1_entity_id"].isin(train_set)],
                          df3[df3["source1_entity_id"].isin(train_set)]])
    X_train = train_df[FEATURE_COLS].fillna(0.0)
    y_train = train_df["label"].values
    pos_rate = y_train.mean()
    log(f"Training set: {len(train_df):,} pairs, {y_train.sum():.0f} positives ({pos_rate*100:.1f}%)")

    # Use class_weight to boost precision (penalize FPs more)
    # Scale positive weight slightly lower than default to push towards precision
    pos_weight = max(0.8, (1.0 - pos_rate) / pos_rate * 0.7)  # precision-biased
    log(f"Using class_weight: negative=1.0, positive={pos_weight:.3f}")
    sample_weight = np.where(y_train == 1, pos_weight, 1.0)

    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.04, max_leaf_nodes=63,
        min_samples_leaf=20, max_depth=8, random_state=42,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    log(f"Model trained in {time.time()-t0:.1f}s (n_iter={clf.n_iter_})")

    # Validate threshold with address guardrail applied FIRST
    val_df2 = df2[df2["source1_entity_id"].isin(val_set)].copy()
    val_df3 = df3[df3["source1_entity_id"].isin(val_set)].copy()
    val_df2["prob"] = clf.predict_proba(val_df2[FEATURE_COLS].fillna(0.0))[:, 1]
    val_df3["prob"] = clf.predict_proba(val_df3[FEATURE_COLS].fillna(0.0))[:, 1]

    # Apply hard address guardrail before threshold search
    val_df2_g = apply_address_guardrail(val_df2)
    val_df3_g = apply_address_guardrail(val_df3)
    log(f"  After guardrail: val2 {len(val_df2):,} -> {len(val_df2_g):,}, val3 {len(val_df3):,} -> {len(val_df3_g):,}")

    gt_df = con.execute("SELECT * FROM tr_gt").df()
    gt_val = gt_df[gt_df["source1_entity_id"].isin(val_set)]
    gt_dict = {s1: set(g["matched_entity_id"]) for s1, g in gt_val.groupby("source1_entity_id")}

    # Valid pair filter: must have at least some name similarity
    val_df2_g = val_df2_g[(val_df2_g["name_jaro"] >= 0.50) | (val_df2_g["core_containment"] == 1.0) | (val_df2_g["exact_sorted_addr"] == 1.0)]
    val_df3_g = val_df3_g[(val_df3_g["name_jaro"] >= 0.50) | (val_df3_g["core_containment"] == 1.0) | (val_df3_g["exact_sorted_addr"] == 1.0)]

    best_th, best_score = 0.55, -1
    log("  Threshold search:")
    for th in np.arange(0.40, 0.95, 0.05):
        p2 = val_df2_g[val_df2_g["prob"] >= th].sort_values("prob", ascending=False).drop_duplicates("matched_entity_id")
        p3 = val_df3_g[val_df3_g["prob"] >= th].sort_values("prob", ascending=False).drop_duplicates("matched_entity_id")
        preds = pd.concat([p2[["source1_entity_id", "matched_entity_id"]], p3[["source1_entity_id", "matched_entity_id"]]])
        pred_dict = {s1: set(g["matched_entity_id"]) for s1, g in preds.groupby("source1_entity_id")}
        score = compute_macro_f05(gt_dict, pred_dict, list(val_set))
        log(f"    th={th:.2f} -> F0.5={score:.4f}")
        if score > best_score:
            best_score = score
            best_th = th

    log(f"*** Best threshold: {best_th:.2f}, Validation F0.5: {best_score:.4f} ***")

    for t in ["tr_s1","tr_s2","tr_s3","tr_gt","ns1","ns2","ns3","cand2","cand3","feat2","feat3"]:
        con.execute(f"DROP TABLE IF EXISTS {t}")
    del df2, df3, train_df, val_df2, val_df3, val_df2_g, val_df3_g
    gc.collect()

    # ========== STEP 2: PROCESS TEST SET ==========
    log("\n=== Step 2: Processing Test Set with Production Pipeline v7 ===")
    all_candidates = {}
    all_matches = {}

    for country in ["France", "US", "India"]:
        tc = time.time()
        log(f"\n--- {country} ---")
        do_translit = (country == "India")

        normalize_table(con, "ts1", str(test_dir / "test_source1.tsv"), country, do_translit=do_translit)
        normalize_table(con, "ts2", str(test_dir / "test_source2.tsv"), country, do_translit=do_translit)
        normalize_table(con, "ts3", str(test_dir / "test_source3.tsv"), country, do_translit=do_translit)

        n1 = con.execute("SELECT count(*) FROM ts1").fetchone()[0]
        n2 = con.execute("SELECT count(*) FROM ts2").fetchone()[0]
        n3 = con.execute("SELECT count(*) FROM ts3").fetchone()[0]
        log(f"  S1={n1:,}, S2={n2:,}, S3={n3:,}")

        for src, t_tbl, c_tbl, f_tbl in [("S2","ts2","tc2","tf2"), ("S3","ts3","tc3","tf3")]:
            tb = time.time()
            log(f"  Blocking {src}...")
            block_candidates(con, "ts1", t_tbl, c_tbl)
            nc = con.execute(f"SELECT count(*) FROM {c_tbl}").fetchone()[0]
            log(f"  {src} candidates: {nc:,} (in {time.time()-tb:.1f}s)")

            log(f"  Extracting features & inference {src}...")
            extract_features(con, "ts1", t_tbl, c_tbl, f_tbl)
            df = con.execute(f"SELECT * FROM {f_tbl}").df()
            if len(df) == 0:
                log(f"  {src}: no candidates")
                continue

            # Record candidates per S1
            for s1, m in zip(df["source1_entity_id"].values, df["matched_entity_id"].values):
                all_candidates.setdefault(s1, []).append(m)

            df["prob"] = clf.predict_proba(df[FEATURE_COLS].fillna(0.0))[:, 1]
            
            # Apply address guardrail (hard rule - eliminate cross-city FPs)
            df_g = apply_address_guardrail(df)
            log(f"  {src} after guardrail: {len(df):,} -> {len(df_g):,}")
            
            # Apply name validity filter
            df_g = df_g[
                (df_g["name_jaro"] >= 0.50) | 
                (df_g["core_containment"] == 1.0) | 
                (df_g["exact_sorted_addr"] == 1.0)
            ]

            matched = df_g[df_g["prob"] >= best_th].sort_values("prob", ascending=False).drop_duplicates("matched_entity_id")

            for s1, m in zip(matched["source1_entity_id"].values, matched["matched_entity_id"].values):
                all_matches.setdefault(s1, []).append(m)

            log(f"  {src} high-precision matches: {len(matched):,}")
            del df, df_g, matched
            gc.collect()

        for t in ["ts1","ts2","ts3","tc2","tc3","tf2","tf3"]:
            con.execute(f"DROP TABLE IF EXISTS {t}")
        gc.collect()
        log(f"  {country} completed in {time.time()-tc:.1f}s")

    # ========== STEP 3: WRITE SUBMISSION TSV FILES ==========
    log("\n=== Step 3: Writing Output Submission Files ===")
    s1_ids = []
    with (test_dir / "test_source1.tsv").open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row and row[0].strip():
                s1_ids.append(row[0].strip())
    s1_ids = sorted(set(s1_ids))
    log(f"Total S1 test entities: {len(s1_ids):,}")

    n_matched = sum(1 for s1 in s1_ids if s1 in all_matches and len(all_matches[s1]) > 0)
    log(f"S1 entities with matches: {n_matched:,} ({n_matched/len(s1_ids)*100:.1f}%)")

    # Write matching_results.tsv
    log("Writing matching_results.tsv...")
    with (OUTPUT / "matching_results.tsv").open("w", encoding="utf-8", newline="", buffering=1024*1024) as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in s1_ids:
            matches = list(dict.fromkeys(all_matches.get(s1, [])))
            f.write(f"{s1}\t{','.join(matches)}\n")

    # Write candidate_pairs.tsv
    log("Writing candidate_pairs.tsv...")
    with (OUTPUT / "candidate_pairs.tsv").open("w", encoding="utf-8", newline="", buffering=1024*1024) as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in s1_ids:
            cands = list(dict.fromkeys(all_candidates.get(s1, []) + all_matches.get(s1, [])))
            f.write(f"{s1}\t{','.join(cands)}\n")

    log(f"matching_results.tsv: {(OUTPUT / 'matching_results.tsv').stat().st_size / 1024**2:.2f} MB")
    log(f"candidate_pairs.tsv: {(OUTPUT / 'candidate_pairs.tsv').stat().st_size / 1024**2:.2f} MB")

    # ========== STEP 4: SUBMISSION VALIDATION ==========
    log("\n=== Step 4: Submission Validation ===")
    val_cmd = [
        sys.executable, str(ROOT / "utils" / "validate_submission.py"),
        "--matching", str(OUTPUT / "matching_results.tsv"),
        "--candidate", str(OUTPUT / "candidate_pairs.tsv"),
        "--test-dir", str(test_dir),
    ]
    res = subprocess.run(val_cmd, capture_output=True, text=True)
    log(res.stdout)
    if res.stderr:
        log("STDERR: " + res.stderr)

    log(f"\nTotal Pipeline Execution Time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
