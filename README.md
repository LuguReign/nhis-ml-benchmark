# nhisml
```text
Survey-aware machine learning toolkit for NHIS Adults data.
```
## Features
```text
- Task-aware training (SRH, smoking, etc.)
- Survey-weighted metrics
- Cross-year evaluation (2023 → 2024)
- Subgroup fairness analysis
- Publication-ready outputs
```
## Installation
```bash

pip install nhisml
```
## Quick start
```bash

#Download and cache raw NHIS Adults public-use files.
nhisml fetch --year 2023 --year 2024
#Builds a clean, analysis-ready core parquet with harmonized variable names.
nhisml build-core --year 2023
```
### Self-rated health
```bash

#Train
nhisml train --in data/core_2023.parquet --task srh_binary
#Evaluate
nhisml evaluate --task srh_binary --latest --year 2024
# Subgroup analysis
nhisml subgroup --task srh_binary --latest --year 2024 --by sex age education
```

### Current smoking
```bash
#Train
nhisml train --in data/core_2023.parquet --task smoking_current
#Evaluate
nhisml evaluate --task smoking_current --latest --year 2024
# Subgroup analysis
nhisml subgroup --task smoking_current --latest --year 2024 --by sex age education
```
### FIGURES & TABLES
```bash

python scripts/make_paper_outputs.py \
  --tasks srh_binary smoking_current \
  --run-for-task srh_binary=runs/<srh_run_dir> \
  --run-for-task smoking_current=runs/<smoking_run_dir>
```
## Others
```bash

nhisml list-tasks # list all available predicton tasks
nhisml describe-task # Describe a specific task.
nhisml list-featuresets # List available feature sets.
nhisml describe-featureset  # Describe feature set contents.
```

## LICENSE
```text
This project is licensed under the MIT License.
See the LICENSE file for details.
...