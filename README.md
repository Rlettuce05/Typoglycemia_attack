# Typoglycemia_attack

Generate typoglycemia-poisoned captions for research datasets.

The current script only shuffles candidate words tagged as nouns or verbs
(`NN*` or `VB*` POS tags). `Typoglycemia` accepts a custom `pos_tagger`
callable for stricter tagging. When none is supplied, it uses NLTK if available
and otherwise falls back to a conservative built-in tagger.

## Baseline attacks

`poisoning_baselines.py` provides a shared framework for additional poisoning
baselines:

- `CharmerBaseline`: character-level perturbations with swap, delete, insert,
  and keyboard-neighbor replacement candidates.
- `TextFoolerBaseline`: word-level synonym replacement with importance-ranked
  token selection and pluggable synonym providers.

Both baselines expose `poison_text()` and `poison_dataframe()`:

```python
from poisoning_baselines import CharmerBaseline, TextFoolerBaseline

charmer = CharmerBaseline(seed=42)
print(charmer.poison_text("A person rides a bicycle.", max_changed_words=2))

textfooler = TextFoolerBaseline(seed=42)
poisoned_df = textfooler.poison_dataframe(
    df,
    text_column="Caption",
    image_column="File Path",
    max_changed_words=2,
)
```

## CLIPScore summaries

`clip_score_summary.py` converts paired CLIPScore result rows into chapter-ready
delta tables and a short Markdown report for slide summaries. Input files can be
CSV, TSV, JSON, or JSONL and should include a sample id, method label, prompt or
caption, and CLIPScore column.

Example:

```bash
python clip_score_summary.py results.tsv --output-prefix clip_results --original-label original
```

The command writes:

- `clip_results_clip_delta_rows.csv`: one row per perturbed prompt paired with
  its original prompt.
- `clip_results_clip_summary.csv`: method-level count, mean delta, variance,
  mean absolute delta, and representative sample ids.
- `clip_results_clip_report.md`: a paper table candidate, one-slide summary
  bullets, and representative examples.

## Poison-rate CLIPScore summaries

`poison_rate_summary.py` aggregates paired CLIPScore rows by poisoning rate and
changed-word count. It pairs perturbed rows with the original row for the same
`sample_id` and split, then writes reviewable tables without rerunning image or
text generation.

Example:

```bash
python poison_rate_summary.py scored_baseline_compare.csv \
  --output-prefix poison_rate_scores \
  --dataset mscoco \
  --split-column split \
  --attack-setting typoglycemia \
  --model clip-vit-base-patch32
```

If the input has no explicit `poison_rate` column, the script computes
`poison_rate = changed_words / original_word_count`, using an explicit word-count
column when present and otherwise the original prompt word count. Outputs are not
overwritten unless `--overwrite` is passed.

The command writes:

- `poison_rate_scores_poison_rate_delta_rows.csv`: per-sample CLIPScore deltas
  with split, changed words, and poisoning rate.
- `poison_rate_scores_poison_rate_summary.csv`: mean CLIPScore deltas by split,
  method, and poisoning rate.
- `poison_rate_scores_changed_words_summary.csv`: mean CLIPScore deltas by split,
  method, and changed-word count.
- `poison_rate_scores_poison_rate_report.md` and
  `poison_rate_scores_poison_rate_metadata.json`: a short report plus
  reproducibility metadata.

## Baseline comparison workflow

`baseline_comparison.py` connects the Typoglycemia pipeline with the Charmer and
TextFooler baselines on the same caption rows. It first writes one prompt table
with `original`, `typoglycemia`, `charmer`, and `textfooler` rows plus changed
word counts:

```bash
python baseline_comparison.py captions.tsv \
  --output-prefix baseline_compare \
  --text-column Caption \
  --image-column "File Path"
```

After running CLIPScore on that prompt table and appending a `clip_score` column,
rerun with `--scored-results` to create chapter-ready comparison outputs:

```bash
python baseline_comparison.py captions.tsv \
  --output-prefix baseline_compare \
  --scored-results scored_baseline_compare.csv
```

The scored run writes per-sample CLIPScore deltas, method-level summaries with
mean changed words, and a Markdown report with representative examples.

## DF-Impact CLIP token features

`df_impact_features.py` extracts CLIP text-token IDs, character offsets, and
hidden-state vectors for each prompt token. It keeps `transformers` and `torch`
optional at import time, so the helper functions and tests can run without a
downloaded model.

Example:

```bash
python df_impact_features.py captions.tsv \
  --text-column Caption \
  --sample-id-column "File Path" \
  --output-jsonl df_impact_clip_hidden_states.jsonl
```

The JSONL output is token-level and includes the sample id, prompt, token id,
token text, optional character offsets, special-token flag, and selected
hidden-state vector. Use `--hidden-layer` to export a layer other than the final
hidden state.

## DF-Impact token matching

`df_impact_matching.py` DP-aligns clean and poisoned token feature files produced
by `df_impact_features.py`. It emits token-level `match`, `substitute`, `insert`,
and `delete` rows with token ids, token indices, character offsets, and hidden
state distances.

Example:

```bash
python df_impact_matching.py clean_clip_hidden_states.jsonl poisoned_clip_hidden_states.jsonl \
  --output-csv df_impact_token_alignment.csv \
  --output-jsonl df_impact_token_alignment.jsonl
```

Matching is token-only by default. Use `--hidden-weight` to include hidden-state
L2 distance in the DP pair cost, or `--allow-unpaired` to skip sample ids that are
present in only one input file.

## DF-Impact word scoring

`df_impact_scoring.py` aggregates token alignments into word-level impact rows and
a DF-integrated word ranking. It reads the CSV/TSV/JSON/JSONL output from
`df_impact_matching.py`; JSONL preserves prompt text and offsets, so it gives the
most precise word grouping.

Example:

```bash
python df_impact_scoring.py df_impact_token_alignment.jsonl \
  --df-table all_words_typoglycemia.tsv \
  --output-prefix df_impact_words
```

The command writes:

- `df_impact_words_word_impact_rows.csv`: per-sample word impact rows.
- `df_impact_words_word_impact_summary.csv`: corpus-level word ranking with
  impact, DF, and DF-Impact scores.
- `df_impact_words_df_impact_report.md`: a short top-word table for research
  notes or slides.
