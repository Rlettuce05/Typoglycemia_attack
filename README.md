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
