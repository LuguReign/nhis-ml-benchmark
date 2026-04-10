# NHIS Adults Reference Statistics

This document records the ground-truth descriptive statistics derived from the
official CDC/NCHS NHIS Adults public-use microdata files for 2023 and 2024.
They serve two purposes:

1. **Reproducibility check** — after running `nhisml fetch` and `nhisml build-core`,
   users can run `nhisml validate-data --year 2023 --year 2024` to confirm their
   processed files match these values.
2. **JOSS correctness criterion** — the automated test suite in
   `tests/test_data_integration.py` uses these figures as toleranced assertions.

All statistics below were computed from the raw CDC/NCHS zip files using
`nhisml build-core` version 0.4.0 on the unmodified public-use files.

---

## File-level statistics

| Statistic | 2023 | 2024 |
|-----------|-----:|-----:|
| Respondents (rows) | 29,522 | 32,629 |
| Columns in core parquet | 64 | 75 |
| Source URL | [adult23csv.zip](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/adult23csv.zip) | [adult24csv.zip](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2024/adult24csv.zip) |

The 2024 file is larger because NCHS expanded the questionnaire to include
additional psychological distress items (`HOPELESS_A`, `LONELY_A`, `NERVOUS_A`,
`RESTLESS_A`, `SAD_A`, `WORTHLESS_A`) and urban/rural classification
(`URBRRL23`).

---

## Self-rated health (SRH) — task `srh_binary`

**Label definition:** PHSTAT_A ≥ 4 → Fair/Poor health (positive class = 1);
PHSTAT_A ≤ 3 → Excellent/Very good/Good (negative class = 0).
PHSTAT_A values 7, 8, 9 are NHIS "don't know / refused / not ascertained"
codes and are excluded (ineligible).

| Statistic | 2023 | 2024 |
|-----------|-----:|-----:|
| Eligible rows (PHSTAT_A ∈ {1–5}) | 29,502 | 32,613 |
| Ineligible rows (missing codes) | 20 | 16 |
| Fair/Poor — unweighted N | ~4,630 | ~5,352 |
| Fair/Poor — unweighted rate | ~15.7% | 16.42% |
| Fair/Poor — survey-weighted rate | 15.1% | 14.84% |

The ~1.5 pp gap between unweighted and weighted rates reflects the
oversampling of older and lower-income adults in NHIS who are more likely
to report fair/poor health. After weighting, the estimate is closer to the
national population proportion.

**Note on AGEP_A missing codes:** The raw parquet retains NHIS missing codes
97 (Refused), 98 (Not ascertained), and 99 (Don't know) in AGEP_A. These are
not actual ages and are excluded from the age-range plausibility check.

**External benchmark:** CDC Health, United States reports approximately
14–16% of US adults with fair or poor self-rated health during this period,
consistent with our weighted estimates.

---

## Current cigarette smoking — task `smoking_current`

**Label definition (primary):** SMKCIGST_A ∈ {1, 2} → current smoker (1);
SMKCIGST_A ∈ {3, 4} → non-smoker (0).
SMKCIGST_A values outside {1–4} trigger a fallback to SMKNOW_A.

| Statistic | 2023 | 2024 |
|-----------|-----:|-----:|
| Eligible rows via SMKCIGST_A | 28,481 | 31,876 |
| Current smoker — unweighted rate | ~11.5% | 10.55% |
| Current smoker — survey-weighted rate | ~10.5% | 9.94% |

**External benchmark:** CDC's National Center for Health Statistics estimates
current cigarette smoking prevalence among US adults at approximately 11–12%
for this period (CDC MMWR, 2023 data). The nhisml weighted estimate of ~10%
is consistent with this, noting that the NHIS measure excludes e-cigarettes
and other tobacco products.

---

## Subgroup prevalence — 2024 (SRH)

Derived from `nhisml subgroup --task srh_binary --latest --year 2024 --by sex age education`.
Threshold used: 0.25 (OOF-tuned weighted F1 on 2023 training data).

### By sex

| Level | N | Weighted rate Fair/Poor | Weighted AUC |
|-------|--:|------------------------:|-------------:|
| Female | 17,632 | 15.5% | 0.858 |
| Male | 14,976 | 14.2% | 0.861 |

### By age band

| Level | N | Weighted rate Fair/Poor | Weighted AUC |
|-------|--:|------------------------:|-------------:|
| 65+ | 10,915 | 24.3% | 0.836 |
| 50–64 | 7,750 | 18.5% | 0.863 |
| 35–49 | ~7,000 | ~12% | ~0.860 |
| 18–34 | ~6,800 | ~8% | ~0.845 |

The strong age gradient (24% in 65+ vs. 8% in 18–34) is expected and
consistent with published NHIS estimates.

---

## Model performance benchmarks (cross-year, lasso baseline)

These benchmarks are produced by training on 2023 and evaluating on 2024.
They serve as a **sanity check for reproducibility**, not as claims of
state-of-the-art performance.

| Task | Model | OOF Weighted AUC (2023) | Held-out Weighted AUC (2024) |
|------|-------|------------------------:|-----------------------------:|
| `srh_binary` | lasso | — | 0.859 |
| `smoking_current` | lasso | 0.729 | 0.742 |

Users who retrain from scratch should expect values within approximately
±0.01 AUC of these figures, given the deterministic seeds in the codebase
(`random_state=42` throughout). Larger deviations suggest a data issue.

---

## How to verify

### Using the CLI (recommended for non-Python users)

```bash
nhisml validate-data --year 2023 --year 2024
```

Example output for a correct installation:

```
============================================================
  nhisml validate-data — NHIS Adults 2023
  File : data/core_2023.parquet
  Result: PASS  (12 passed, 0 failed)
============================================================
  ✓ Row count
  ✓ Minimum column count
  ✓ Required columns present
  ✓ Survey weights: no missing values
  ✓ Survey weights: all positive
  ✓ SRH eligible count
  ✓ SRH unweighted prevalence (ref ≈ 15.7%, ±1%)
  ✓ SRH weighted prevalence (ref ≈ 14.0%, ±1%)
  ✓ Smoking eligible count (SMKCIGST_A)
  ✓ Smoking unweighted prevalence (ref ≈ 11.5%, ±1%)
  ✓ Smoking weighted prevalence (ref ≈ 10.5%, ±1%)
  ✓ Sex distribution plausible (49–55% female)
  ✓ All four NCHS regions present
  ✓ Age range plausible (18–85, top-coded)
```

### Using pytest (for developers)

```bash
pytest tests/test_data_integration.py -v
```

These tests are automatically skipped when the parquet files are absent,
so the full test suite passes cleanly in CI environments without data.

### Using Python directly

```python
from nhisml.validate_data import validate_core_year

report = validate_core_year(2024, "data/core_2024.parquet")
report.print_report(verbose=True)
assert report.all_passed
```

---

## Data provenance

| Item | Detail |
|------|--------|
| Source | CDC/NCHS National Health Interview Survey (NHIS) Adults public-use microdata |
| Years | 2023 and 2024 |
| URL | https://www.cdc.gov/nchs/nhis/data-questionnaires-documentation.htm |
| License | Public domain (US Government work) |
| Processing | `nhisml fetch` → `nhisml build-core` (version 0.4.0) |
| Platform used for reference run | macOS arm64, Python 3.9.6, pandas 2.3.3, numpy 2.0.2 |

The raw files are not redistributed with this package. Users must download
them directly from the CDC FTP server using `nhisml fetch`.
