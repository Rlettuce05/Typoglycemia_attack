# Typoglycemia_attack

研究用データセット向けに、タイポグリセミアで汚染したキャプションを生成します。

現在のスクリプトは、名詞または動詞としてタグ付けされた候補単語のみをシャッフルします
（`NN*` または `VB*` の品詞タグ）。`Typoglycemia` は、より厳密なタグ付けのために
カスタムの `pos_tagger` 呼び出し可能オブジェクトを受け取れます。指定しない場合は、
利用可能であれば NLTK を使用し、そうでなければ保守的な組み込みタガーへフォールバックします。

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
