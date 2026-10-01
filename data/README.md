# Data

| Folder / file | Source | Contents |
|---|---|---|
| `liar/{train,valid,test}.tsv` | Wang (2017), [LIAR](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) — official splits, 14 tab-separated columns, no header | 12,791 PolitiFact statements, 6-way truth label, speaker metadata |
| `fakenewsnet/{politifact,gossipcop}_{fake,real}.csv` | Shu et al. (2018), [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) | `id, news_url, title` per article (the original `tweet_ids` column is dropped here — it isn't used; the full files come from `scripts/download_data.py`) |
| `unified.parquet` | produced by `backend/ml/data.py` | the cleaned, de-duplicated, labelled training set: `text, label (1=fake), fine_label, dataset, domain, split, speaker, party, context, subject` — 34,534 rows |

LIAR columns: `id, label, statement, subject, speaker, job_title, state_info, party_affiliation,
barely_true_counts, false_counts, half_true_counts, mostly_true_counts, pants_on_fire_counts, context`.

The preprocessing is documented step by step in `notebooks/01_data_preprocessing.ipynb`.
`data/raw/` (git-ignored) is where `scripts/download_data.py` puts the untouched originals.
