# Fine-tuning Qwen2.5-3B for Text-to-SQL with LoRA

Fine-tuned a 3B open-source model to convert natural language questions
into SQL queries. Exact-match accuracy improved from 3% to 75% on a held-out
test set, trained entirely on free Kaggle GPUs for $0.

## Results

| Model | Exact match (100 held-out questions) |
|---|---|
| Qwen2.5-3B-Instruct (base) | 3/100 |
| Qwen2.5-3B + LoRA fine-tune | **75/100** |

Only 0.96% of parameters (29.9M of 3.1B) were trained, using QLoRA
via Unsloth on a single free Kaggle T4 session (~40 min of training).

## The iteration story (v1 to v2)

**v1:** Trained on Spider questions without database schemas.
Result: 2 to 14 exact matches. Better, but capped: the model had to
guess table and column names it had never seen.

**v2:** Switched to a schema-in-prompt setup (b-mc2/sql-create-context),
so every prompt includes the relevant CREATE TABLE statements.
Result: 3 to 75. Diagnosing the v1 bottleneck and fixing the data
pipeline was worth far more than any hyperparameter change.

## Example

**Question:** What is the number of party in the arkansas 1 district

**Base model:**
```sql
SELECT party FROM table_1341930_5 WHERE district = 'Arkansas 1'
```
Returns the party values themselves, but the question asked for a count.

**Fine-tuned model:**
```sql
SELECT COUNT(party) FROM table_1341930_5 WHERE district = "Arkansas 1"
```
Exact match with the reference query.

### Why 75% understates accuracy

Strict exact match counts functionally-equivalent SQL as a miss. For
example, on "which contestant got the least votes?", the fine-tuned model
produced a correct JOIN + GROUP BY + ORDER BY query that differs from the
reference only in which table is aliased T1 vs T2. Manual inspection of
the 25 misses shows several such cases (reordered WHERE clauses, swapped
JOIN aliases), so exact match is a conservative floor.

## Setup

- Model: unsloth/Qwen2.5-3B-Instruct (4-bit)
- Method: QLoRA (r=16, alpha=16) on all attention + MLP projections
- Training: 300 steps, batch 8 (2 x 4 grad accum), lr 2e-4, ~40 min on T4 x2
- Eval: strict exact-match after normalization (whitespace, case,
  markdown fences, trailing semicolons)

## Files

- `finetune.ipynb` - full notebook (data, training, before/after eval)
- `results/` - v1_baseline_results.json, v1_finetuned_results.json,
  v2_baseline_results.json, v2_finetuned_results.json
