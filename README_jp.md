# Typoglycemia_attack

研究用データセット向けに、タイポグリセミアで汚染したキャプションを生成します。

`typoglycemia.py` は、POS で絞り込まない `Typoglycemia` ベースラインを提供し、
有効なアルファベット候補単語をシャッフルします。`pos_filter.py` はそのベースラインを
継承する `PosFilteredTypoglycemia` を追加し、名詞または動詞としてタグ付けされた候補単語
（`NN*` または `VB*` の品詞タグ）のみをシャッフルします。POS フィルタは、より厳密な
タグ付けのためにカスタムの `pos_tagger` 呼び出し可能オブジェクトを受け取れます。指定しない
場合は、利用可能であれば NLTK を使用し、そうでなければ保守的な組み込みタガーへ
フォールバックします。

## ベースライン攻撃

`poisoning_baselines.py` は、追加の汚染ベースライン向けの共通フレームワークを提供します。

- `CharmerBaseline`: swap、delete、insert、キーボード隣接文字への置換候補を使う文字レベルの摂動。
- `TextFoolerBaseline`: 重要度で順位付けしたトークン選択と差し替え可能な同義語プロバイダーを使う単語レベルの同義語置換。

どちらのベースラインも `poison_text()` と `poison_dataframe()` を公開しています。

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

## CLIPScore サマリー

`clip_score_summary.py` は、対応する CLIPScore 結果行を、章で使える差分テーブルと
スライド要約向けの短い Markdown レポートに変換します。入力ファイルは CSV、TSV、
JSON、JSONL に対応しており、サンプル ID、手法ラベル、プロンプトまたはキャプション、
CLIPScore 列を含める必要があります。

例:

```bash
python clip_score_summary.py results.tsv --output-prefix clip_results --original-label original
```

このコマンドは以下を書き出します。

- `clip_results_clip_delta_rows.csv`: 元のプロンプトと対応付けられた、摂動済みプロンプトごとの行。
- `clip_results_clip_summary.csv`: 手法ごとの件数、平均差分、分散、平均絶対差分、代表サンプル ID。
- `clip_results_clip_report.md`: 論文用テーブルの候補、1 スライド要約の箇条書き、代表例。

## 混入率別 CLIPScore サマリー

`poison_rate_summary.py` は、対応する CLIPScore 行を混入率と変更単語数ごとに集計します。
同じ `sample_id` と split の元行に摂動済み行を対応付け、画像生成やテキスト生成を再実行せずに
レビュー可能な表を書き出します。

例:

```bash
python poison_rate_summary.py scored_baseline_compare.csv \
  --output-prefix poison_rate_scores \
  --dataset mscoco \
  --split-column split \
  --attack-setting typoglycemia \
  --model clip-vit-base-patch32
```

入力に明示的な `poison_rate` 列がない場合は、明示された単語数列があればそれを使い、
なければ元プロンプトの単語数を分母として `poison_rate = changed_words / original_word_count`
を計算します。既存の出力は `--overwrite` を渡さない限り上書きしません。

このコマンドは以下を書き出します。

- `poison_rate_scores_poison_rate_delta_rows.csv`: split、変更単語数、混入率を含むサンプルごとの CLIPScore 差分。
- `poison_rate_scores_poison_rate_summary.csv`: split、手法、混入率ごとの平均 CLIPScore 差分。
- `poison_rate_scores_changed_words_summary.csv`: split、手法、変更単語数ごとの平均 CLIPScore 差分。
- `poison_rate_scores_poison_rate_report.md` と
  `poison_rate_scores_poison_rate_metadata.json`: 短いレポートと再現性メタデータ。

## ベースライン比較ワークフロー

`baseline_comparison.py` は、同じキャプション行に対して Typoglycemia パイプラインと
Charmer、TextFooler ベースラインを接続します。まず、`original`、`typoglycemia`、
`charmer`、`textfooler` の各行と変更単語数を含むプロンプトテーブルを 1 つ書き出します。

```bash
python baseline_comparison.py captions.tsv \
  --output-prefix baseline_compare \
  --text-column Caption \
  --image-column "File Path"
```

そのプロンプトテーブルに対して CLIPScore を実行し、`clip_score` 列を追加したあと、
`--scored-results` を付けて再実行すると、章で使える比較出力を作成できます。

```bash
python baseline_comparison.py captions.tsv \
  --output-prefix baseline_compare \
  --scored-results scored_baseline_compare.csv
```

スコア付き実行では、サンプルごとの CLIPScore 差分、平均変更単語数を含む手法ごとのサマリー、
代表例を含む Markdown レポートを書き出します。

## DF-Impact CLIP トークン特徴量

`df_impact_features.py` は、各プロンプトトークンについて CLIP のテキストトークン ID、
文字オフセット、隠れ状態ベクトルを抽出します。`transformers` と `torch` は
インポート時には任意依存のままなので、モデルをダウンロードしていなくてもヘルパー関数と
テストを実行できます。

例:

```bash
python df_impact_features.py captions.tsv \
  --text-column Caption \
  --sample-id-column "File Path" \
  --output-jsonl df_impact_clip_hidden_states.jsonl
```

JSONL 出力はトークン単位で、サンプル ID、プロンプト、トークン ID、トークン文字列、
任意の文字オフセット、特殊トークンフラグ、選択した隠れ状態ベクトルを含みます。
最終隠れ状態以外の層を書き出すには `--hidden-layer` を使用してください。

## DF-Impact トークンマッチング

`df_impact_matching.py` は、`df_impact_features.py` が生成したクリーン版と汚染版の
トークン特徴量ファイルを DP でアラインメントします。トークン ID、トークンインデックス、
文字オフセット、隠れ状態距離を含む、トークン単位の `match`、`substitute`、`insert`、
`delete` 行を出力します。

例:

```bash
python df_impact_matching.py clean_clip_hidden_states.jsonl poisoned_clip_hidden_states.jsonl \
  --output-csv df_impact_token_alignment.csv \
  --output-jsonl df_impact_token_alignment.jsonl
```

デフォルトではトークンのみでマッチングします。DP のペアコストに隠れ状態の L2 距離を
含めるには `--hidden-weight` を、片方の入力ファイルにしか存在しないサンプル ID を
スキップするには `--allow-unpaired` を使用してください。

## DF-Impact 単語スコアリング

`df_impact_scoring.py` は、トークンアラインメントを単語レベルの impact 行と
DF 統合単語ランキングに集計します。`df_impact_matching.py` の CSV、TSV、JSON、JSONL
出力を読み込みます。JSONL はプロンプト本文とオフセットを保持するため、最も精密に単語を
グループ化できます。

例:

```bash
python df_impact_scoring.py df_impact_token_alignment.jsonl \
  --df-table all_words_typoglycemia.tsv \
  --output-prefix df_impact_words
```

このコマンドは以下を書き出します。

- `df_impact_words_word_impact_rows.csv`: サンプルごとの単語 impact 行。
- `df_impact_words_word_impact_summary.csv`: impact、DF、DF-Impact スコアを含むコーパス単位の単語ランキング。
- `df_impact_words_df_impact_report.md`: 研究メモやスライド向けの短い上位単語表。
