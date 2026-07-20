---
title: "nhisml: Fostering Infrastructural Collaboration Between Survey Research and AI/ML"
authors:
  - name: Nicholas Lugu Reign
    affiliation: 1
  - name: Catherine Lamoreaux
    affiliation: 1
  - name: Jan Simson
    affiliation: "2, 3"
  - name: Christoph Kern
    affiliation: "2, 3"
  - name: Frauke Kreuter
    affiliation: "1, 2, 3"
affiliations:
  - name: University of Maryland, United States
    index: 1
  - name: LMU Munich, Germany
    index: 2
  - name: Munich Center for Machine Learning (MCML), Germany
    index: 3
date: 20 July 2026
bibliography: paper.bib
---

# Summary

Trustworthy and accurate modeling requires benchmarking to the best data
available. While the United States (U.S) government administers hundreds
of high-quality surveys with publicly available datasets, this data is
underused in model alignment and benchmarking beyond the survey research
community. Part of the reason for this is the survey-specific knowledge
necessary for understanding the complex documentation needed to convert
raw formats into analyzable data.

To lower this barrier to entry, we developed `nhisml`, a Python package
with a flexible and accessible interface for data from the National
Health Interview Survey (NHIS), the U.S.' largest and longest-running
household health survey. Our package includes pre-specified prediction
tasks, including the prediction of self-rated health, and associated
data processing and evaluation pipelines that are mindful of the NHIS
design and weighting procedures.

# Statement of Need

Publicly available high-quality U.S. government survey data has been
historically designed to be consumed by survey statisticians. However,
the audience for high quality datasets has evolved beyond those who
produce official statistics to include those from data science, machine
learning (ML), computer science, and now Artificial Intelligence (AI).
Bridging the gap between these research communities requires
"infrastructural collaboration", which involves speaking the same
language, programmatically, and removing potential barriers to
understanding [@rothschild2025successfully].

While these survey datasets might be ideal in theory for AI benchmarking
and model fine tuning, potential barriers to their use include: a lack
of understanding complex survey sampling and survey design-aware
analysis, standard code available in programming languages AI/ML
researchers don't use, and a general unawareness of the vast set of
high-quality data products that survey institutions have to offer for
improving ML model training, evaluation and benchmarking.

To help bridge this gap, we created `nhisml`. Our development of this
package was inspired by the widely-used `folktables` Python package,
which transforms U.S. Census microdata into a superset, making it
accessible to AI/ML researchers [@ding2021]. While folktables' superset
contains a multitude of features, the focus is more on socioeconomic
factors than health-related outcomes. In the spirit of folktables, we
hope that our package makes the NHIS more accessible to health data
users outside the survey research community and survey researchers who
want to become more involved with AI/ML research. By incorporating the
NHIS weighting procedures into the package and writing the package in
Python, the default language of choice for AI/ML researchers, we built a
tool for those who wish to benchmark their models to high-quality
population-representative U.S. health survey data.

# Software Design

We developed an open-source Python package (`nhisml`)[^1] aimed at
integrating machine learning functions into NHIS data seamlessly.
`nhisml` has five modules that form a linear data processing pipeline.
Each module is independently callable via the `nhisml` commands and
produces a structured artifact used at the next stage. The pipeline
proceeds from raw data acquisition (`fetch`), through column selection
and normalization (`build_core`), preprocessing (`preprocess`),
weight-aware model training (`train`), and finally population-weighted
evaluation and subgroup auditing (`evaluate`, `subgroup`).

The architecture follows a registry pattern for its two primary
extension points: Task objects and FeatureSet objects. Both are
registered by name in module-level dictionaries and retrieved via
factory functions (`maketask()`, `getfeatureset()`). This means that
adding a new prediction task or a new feature set requires editing a
single file and adding a single registration call, with no modifications
to any other module.

#### Preprocessing Pipeline

The preprocessing pipeline is implemented as a scikit-learn pipeline
with two steps: a custom `PrepareFrame` transformer followed by a
`ColumnTransformer`. This design ensures that all transformations are
fit exclusively on training data and applied without re-fitting at
evaluation time, preventing any form of data leakage.

PrepareFrame encodes two categories of transformation. Survey-specific
operations require no configuration from the user: NHIS non-response 
codes are mapped to `NaN`, binary yes/no items are one-hot encoded, and 
rare categorical levels are collapsed into a single "rare" category.
At evaluation time, unseen levels are
automatically routed to this category. Missingness indicator flags are
generated for all ordinal and categorical columns.

#### Tasks and Feature Sets

A Task is a frozen dataclass that encodes the following: the source
variable name, the label derivation function, the set of NHIS columns
that must be present in the dataset, and a human-readable description.
The label derivation function accepts a DataFrame and returns a tuple of
`(y, eligiblemask)`, where `eligiblemask` identifies which respondents
have valid responses for this particular outcome. This design ensures
that eligibility filtering -- which requires knowing that code 9 means
"not ascertained" rather than a genuine response -- is included in the
task. The task registry is designed to be extensible, adding a new task
requires only defining a task dataclass instance with the appropriate
label function.

The current release ships two pre-defined tasks, specified in Table 1. The
feature set summary is given in Table 2.


  Task ID             Source Variable          Positive Class                             Negative Class                               Problem Type  Eligibility
  ------------------- ------------------------ ------------------------------------------ ------------------------------------------- -------------- ----------------------
  Self-rated Health   `PHSTAT_A`               Fair, Poor (codes 4--5)                    Excellent / Very Good / Good (codes 1--3)       Binary     Codes 1--5 only
  Smoking Current     `SMKCIGST_A` (primary)   Every-day / Some-day smoker (codes 1--2)   Former / Never smoker (codes 3--4)              Binary     Codes 1--4 (primary)

  : Table 1. Pre-defined Benchmark Tasks


  Class                   $k$   Domains                                                                                                                                                             Transformations
  ----------------------- ----- ------------------------------------------------------------------------------------------------------------------------------------------------------------------- ---------------------------------------------------------------------------------------
  Binary                  43    Employment, disability, mental health medication, chronic conditions (hypertension, diabetes, CVD, asthma, arthritis, COPD, cancer), insurance, healthcare access   NHIS 1/2 recode; mode imputation
  Ordinal                 20    Income-to-poverty ratio, educational attainment, food security, Kessler K6 psychological distress items, healthcare utilization recency                             Median imputation; missingness flag
  Categorical (nominal)   6     Marital status, urban-rural classification, US Census region, employment status                                                                                     Rare-level bucketing ($<50$ obs); mode imputation; one-hot encoding; missingness flag

  : Table 2. Feature Sets by Measurement Class


#### Survey Weights, Model Training and Evaluation

Survey weights are treated as a first-class software concern throughout
`nhisml`. NHIS person-level annual weights (`WTFA_A`) are normalized
before use, ensuring numerical stability without altering their relative
structure.

The two baseline models in `nhisml`, an elastic-net penalized logistic
regression and a random forest classifier, are trained with NHIS 
survey weights passed as observation-level sample weights, applied 
automatically across the full-data fit and all cross-validation folds.

For model evaluation, all metrics (AUC, PR-AUC, F1, Brier score,
log-loss, and ECE) are computed using the survey weights of the NHIS. To
assess differential model performance across sociodemographic subgroups,
stratified evaluations are build for sex, age, and educational
attainment and can be extended to additional NHIS variables of interest.
Within each subgroup level, the same suite of weighted metrics is
computed as in the overall evaluation. To quantify disparities relative
to overall model performance, fairness delta metrics are computed as the
difference between each subgroup's weighted AUC, F1, Brier score, and
expected calibration error and the corresponding overall metric.

#### Reproducibility

Every `nhisml` train invocation produces a self-contained, timestamped
run directory. The manifest.json file records the task name, model name,
featureset name, absolute input path, effective sample size, weight
usage flag, all artifact paths, and the exact software versions used.

# State of the Field

The Python package `folktables` inspired the development of this
package. The developers behind `folktables` created a collection of
datasets derived from US Census data and programmed prediction tasks
corresponding to various sociodemographic domains, such as income and
housing [@ding2021].

In the public health domain, the National Health and Nutrition
Examination Survey (NHANES) is another US health survey, covering both
clinical measures alongside self-reported survey data. Similar to the US
Census, NHANES has several user-developed Python packages associated
with downloading and analyzing its data, including `nhanes`
[@poldrack2020nhanes], `nhanes-dl` [@butcher2022nhanesdl], `pynhanes`
[@pyrkov2022pynhanes], and `NHANES-pyTOOL-API` [@rusere2023nhanes].
However, the danger to this accessibility is misuse.
[@suchak2025explosion] explore the phenomenon of a recent influx of
formulaic NHANES papers in academic publishing, due to AI and paper
mills. Readying a dataset like the NHANES for AI use, partially through
the development of R and Python packages, engenders "the ability to
generate large numbers of machine-learning models," which in turn allow
for "rapid post-hoc investigation of alternative hypotheses"
[@suchak2025explosion]. While `nhisml` includes guardrails such as
pre-specified feature sets, we caution against an exploratory use of
NHIS data through our package.

While NHANES has clinical and survey data, the NHIS has a much higher
number of variables or features or questions, thus making it a more
natural fit for machine learning methods. There are numerous instances
in the literature of other researchers using ML to analyze NHIS data:
[@shilane2023machine] predicted telehealth utilization;
[@miller2023determinants] predicted life satisfaction; [@guan2025breast]
predicted breast cancer risk; [@seixas20180] created sleep and physical
activity profiles to analyze racial disparities in diabetes risk;
[@akter2025optimizing] used the NHIS to evaluate heart disease
prediction. Creating a package to streamline ML analysis of the NHIS
would enable faster and more accessible analysis, which will allow
subject matter experts more time to devote thorough methodology and
rigorous validation of results.

# Research Impact Statement

`nhisml` provides a flexible interface for data from the National Health
Interview Survey (NHIS), the U.S.' largest and longest-running household
health survey, and includes pre-specified prediction tasks as well as
survey-design-aware data processing and evaluation pipelines. It is
designed for substantive researchers in public health, epidemiology, and
health services, as well as for AI/ML researchers aiming to test novel
prediction algorithms with real-world health data.

As a concrete validation of the pipeline, cross-year application of
`nhisml` to the 2023 and 2024 NHIS Adults files reproduces external
benchmarks: survey-weighted estimates of fair/poor self-rated health are
15.1% (2023) and 14.8% (2024), matching the estimates reported by the
NHIS Adult Summary Health Statistics for these years [@nchs2026adult].

Our package is available on [PyPI](https://pypi.org/project/nhisml/) and
has been released under the MIT license.

# AI Usage Disclosure

Claude Code (Sonnet 4.6 and 5) was used for writing the scripts for 
automated testing of the package, as well as for debugging coding errors. 
All contributions from Claude models were verified by the authors to 
ensure accuracy.

# Acknowledgments

We are a team of university students from the University of Maryland's
Joint Program in Survey Methodology (JPSM) and researchers from LMU
Munich. We are not affiliated with the CDC, NCHS, or the NHIS.

[^1]: `nhisml` is available on PyPI (<https://pypi.org/project/nhisml/>)
    and GitHub (<https://github.com/LuguReign/nhis-ml-benchmark>).
