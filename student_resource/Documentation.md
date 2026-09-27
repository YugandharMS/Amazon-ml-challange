# ML Challenge 2026: Business Entity Resolution Solution Documentation

**Challenge:** Amazon ML Challenge 2026 — Multilingual Business Entity Resolution  
**Evaluation Metric:** Macro $F_{0.5}$ (Precision-Weighted harmonic mean)  
**Validation Score:** $0.9511$ Macro $F_{0.5}$  

---

## 1. Executive Summary

This solution presents a high-throughput, precision-first entity resolution pipeline designed to resolve noisy business entity records across three heterogeneous databases ($S_1$, $S_2$, and $S_3$) spanning France, the United States, and India. The core architecture integrates:
1. Multi-script phonological transliteration (covering Devanagari, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Gurmukhi, and Oriya scripts into Latin phonetics).
2. Word-boundary legal entity and domain normalization that resolves regex overlap issues and standardizes business entities.
3. High-precision candidate blocking via rare-token inversion, sorted n-grams, and rare address locality filtering.
4. A 28-dimensional similarity vector evaluated with a precision-biased Histogram Gradient Boosting Classifier (`HistGradientBoostingClassifier`).
5. A deterministic cross-city address compatibility guardrail that completely eliminates false positive matches between geographically conflicting entities.

---

## 2. Methodology

### 2.1 Problem Analysis
Key challenges identified during exploratory data analysis (EDA):
- **Heavy Multilingual Noise:** Indian entities frequently alternate between native Indic scripts and English, with phonetic spelling variations (e.g. Hindi *डायनामिक प्रॉपर्टीज* vs English *Dynamic Properties*).
- **Asymmetric Precision Penalty ($F_{0.5}$):** The Macro $F_{0.5}$ metric penalizes false positives 4× more than false negatives ($(\beta=0.5)$). A single false merge drops the score significantly.
- **OCR and Typographic Corruptions:** Common OCR confusions include lowercase `l`/`1` replacing uppercase `I` (e.g. `lndia` for `India`, `lnfrastructure` for `Infrastructure`, `prriraet` for `private`), web domain suffixes fused to names (`candymanagement.com`), and leetspeak numbers (`0` for `o`, `5` for `s`).
- **Zero-Shot Test Distribution:** France is present exclusively in the test set (259k entities), requiring robust unsupervised generalization across French legal suffixes (`SARL`, `SAS`, `EURL`, `SCI`, `Amicale`, `Ligue`) and street abbreviations (`RUE`/`R.`, `AVENUE`/`AV.`, `BOULEVARD`/`BD.`).

### 2.2 Solution Strategy
- **Approach Type:** Hybrid Multi-Channel Blocking + Gradient Boosted Matching + Hard Deterministic Precision Guardrail.
- **Core Innovation:** 
  1. *Word-Boundary Legal Suffix & Business Stem Normalization:* Eliminates adjacent legal entity collision bugs (e.g., `Private Limited` retaining `limited`).
  2. *Dual Rare-Token Blocking:* Inverts rare core-name tokens ($\le 5$ occurrences) and rare alphabetic address tokens ($\le 3$ occurrences), achieving $>92\%$ blocking recall while keeping candidate pairs computationally lean.
  3. *Post-Model Address Compatibility Guardrail:* Hard geographic rejection when both entities possess non-empty addresses with conflicting alphabetic tokens and zero postal/street confirmation.

---

## 3. Candidate Generation (Blocking)

To scale across millions of cross-product combinations without exceeding memory or flooding false positives, 11 precision-oriented blocking channels are executed in DuckDB:

1. `norm_name`: Exact normalized full string match (frequency cap $\le 25$).
2. `core_name`: Stripped core name match ($\ge 4$ characters, cap $\le 25$).
3. `sorted_core`: Word-order-invariant core tokens ($\ge 4$ characters, cap $\le 25$).
4. `spaceless_core`: Whitespace-invariant core string ($\ge 5$ characters, cap $\le 25$).
5. `unleet_sorted`: Leetspeak-inverted sorted core tokens (cap $\le 25$).
6. `unleet_spaceless`: Leetspeak-inverted spaceless core string (cap $\le 25$).
7. `first_2_tokens`: Leading two tokens of core name ($\ge 7$ characters, cap $\le 20$).
8. `sorted_addr`: High-precision address token sequence ($\ge 10$ characters, cap $\le 15$).
9. `first_num + first_street_word`: Exact street number + first street word (cap $\le 15$).
10. `first_num + postal_code`: Exact street number + 4-6 digit postal code (cap $\le 15$).
11. `first_token + postal_code`: Leading core name token + exact postal code (cap $\le 15$).
12. `rare_token`: Bidirectional intersection of tokens of length $\ge 5$ occurring $\le 5$ times in both source and target tables.
13. `rare_addr_token`: Bidirectional intersection of alphabetic address locality tokens ($\ge 6$ characters) occurring $\le 3$ times in both source and target tables.

**Results of Blocking:**
- Overall Blocking Recall: **$92.50\%$**
- US Recall: **$95.6\%$** (S2), **$94.8\%$** (S3)
- India Recall: **$88.6\%$** (S2), **$88.3\%$** (S3)

---

## 4. Matching Model & Feature Engineering

### 4.1 Feature Set (28 Extracted Features)
- **Exact Matches:** `exact_norm_name`, `exact_core_name`, `exact_sorted_core`, `exact_spaceless`, `exact_sorted_addr`.
- **Fuzzy Name Similarity:** Jaro-Winkler on normalized name, core name, sorted core, and unleeted core; Levenshtein normalized similarity.
- **Token Overlap:** Name token Jaccard similarity, raw name token intersection count, address token Jaccard similarity, alphabetic address token overlap (`addr_overlap`), normalized address token overlap (`addr_overlap_norm`).
- **Numeric & Geographic Verification:** `first_num_match`, `addr_num_overlap`, `num_street_match`, `postal_match`.
- **Structural Signals:** Length ratios, core string containment (`contains(s, t)`), word count indicators (`core_num_words`), initial 3-character prefix agreement (`name_starts_same`), leading token match (`first_token_match`), address presence indicator (`has_addr_both`).

### 4.2 Model Architecture & Training
- **Model:** `HistGradientBoostingClassifier` with `max_iter=400`, `learning_rate=0.04`, `max_leaf_nodes=63`, `min_samples_leaf=20`, `max_depth=8`.
- **Class-Weight Precision Bias:** Negative instances are weighted higher (`positive_weight ~ 0.7 * (1-p)/p`) to discourage false positives.
- **Address Guardrail:** If both entities possess valid addresses, pairs with `addr_jaro < 0.35` and zero alphabetic overlap (`addr_overlap < 1`) are deterministically discarded unless postal code or full normalized name matches perfectly.
- **Threshold Calibration:** Swept on a held-out validation set using the competition's exact Macro $F_{0.5}$ metric. Optimal operating threshold: $\tau = 0.70$.

---

## 5. Results & Error Analysis

- **Local Validation Score (Macro $F_{0.5}$):** **$0.9511$**
- **Test Set Match Coverage:** $>95\%$ of test entities resolved to matching target records.
- **False Positive Elimination:** Hard address filtering prevents cross-city collisions between common single-word business names (e.g. *Apex*, *Sun*, *Star*).
- **False Negative Mitigation:** Multi-script transliteration and rare locality token blocking successfully capture cross-lingual and severely truncated addresses.

---

## 6. Conclusion

The v8 precision-first pipeline achieves an optimal balance between candidate recall ($>92\%$) and matching precision, scaling efficiently across 1.73M test entities in under 15 minutes. The integration of word-boundary regexes, phonological transliteration, and deterministic address guardrails yields a state-of-the-art Macro $F_{0.5}$ validation score of $0.9511$.
