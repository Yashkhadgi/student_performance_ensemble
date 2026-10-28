# Data provenance

## Real dataset

- Cortez, Paulo (2008). *Student Performance*. UCI Machine Learning Repository.
- DOI: https://doi.org/10.24432/C5TG7T
- Landing page: https://archive.ics.uci.edu/dataset/320/student+performance
- License: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/
- The original dataset concerns two Portuguese secondary schools; it contains Mathematics (`student-mat.csv`, 395 records) and Portuguese language (`student-por.csv`, 649 records).
- `G1` and `G2` are earlier-period grades; `G3` is final-period grade, scored 0 to 20.
- Original academic reference: Cortez, P. & Silva, A. (2008). Using data mining to predict secondary school student performance. *Proceedings of 5th Future Business Technology Conference*.

### Redistribution and reproducibility

Original CSV files are **not bundled** in this ZIP. Use `python -m src.data --download`, or download the archive from UCI manually. Credit the above dataset when reporting results. The synthetic demo file included here was programmatically generated and contains no real individual data.

## Synthetic example data

The `data/demo/student-demo-SYNTHETIC.csv` file is completely generated using seeded random numbers and a fictional grade formula in `src/data.py`. It is strictly for installation checks, running the app without internet and automated tests; it is not empirical evidence of academic outcomes. Pre-trained demo artifacts are likewise synthetic-only. Re-train all scripts on real UCI CSVs before submission.
