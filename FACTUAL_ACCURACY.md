# Factual-accuracy improvement

The chemistry v1 checkpoint produced correct values and units in 21 of 64 source-grounded questions and zero complete correct profiles. Template overlap and low teacher-forced perplexity did not establish factual accuracy. Questions already contain source values, so these failures cannot be attributed to missing access to a database.

The v2 run continues the same native 18,880,896-parameter checkpoint. It preserves the tokenizer, source archive, train/validation/test partitions and prior weights. No external pretrained model or inference API is used. Entity names, numerical/formula values and units receive a 12-fold token-loss weight; other supervised answer tokens retain weight one. Prompts and padding have no supervised weight. Reported validation/test perplexity remains unweighted for comparability.

Checkpoint selection uses generated entity/value/unit matches on all 64 validation questions, with unweighted validation loss as a tie-breaker. Test questions are excluded from optimizer updates and selection. The final evaluation repeats the existing 64-question benchmark; it is not a fresh blind test. No additional source values are promoted to verified engineering records.

```powershell
python scripts/train_split_dataset.py --archive data/chemistry_expansion/property_sft.zip --checkpoint runs/chemistry_properties_v1/export/ownllm.pt --tokenizer runs/chemistry_properties_v1/export/tokenizer.json --out runs/chemistry_properties_v2 --provenance data/chemistry_expansion/PROVENANCE.json --epochs 2 --seq-len 416 --batch 4 --threads 4 --lr 0.00008 --max-new-tokens 224 --critical-field-weight 12 --factual-selection
python scripts/check_property_predictions.py --run runs/chemistry_properties_v2
python scripts/evaluate_factual_readiness.py --run runs/chemistry_properties_v2
```

Use a new output directory for subsequent training. Exact answers and conditions are checked independently of lexical overlap. Complete-profile scoring includes entity, property name, scientific description, value, unit, conditions, data kind and structure. Readiness requires at least 64 test cases and 95% complete-profile accuracy plus a passing reviewed-chat audit. Passing permits a supervised source-grounded trial; it does not certify closed-book recall, scientific reasoning or engineering safety. The readiness script records results without activating weights or changing server configuration.

The reviewed-chat audit compares each material profile against the local collection, checks that unreviewed fields are excluded, and exercises both generation and streaming HTTP routes without invoking the native model. This is a reproduction audit, not an independent review of the source science.

During this audit, Sodium's verified symbol `Na` was incorrectly omitted as an `NA` placeholder. The placeholder filter now respects that case-sensitive chemical symbol, while continuing to omit `NA` and `N/A` fillers.

The candidate export includes weights, tokenizer, configuration, evaluation, prediction grading, readiness results and source attribution. Its checksums are regenerated after adding these artifacts. Until readiness passes, the current chat configuration remains in place and materials answers continue through the verified-record policy.

## Completed v2 results

Two weighted epochs completed; validation selected epoch two. Validation entity/value/unit correctness improved from 0/64 initially to 4/64 after epoch one and 9/64 after epoch two.

| Repeated source-grounded test metric | v1 | v2 |
| --- | ---: | ---: |
| Correct value and unit | 21/64 | 23/64 |
| Correct entity | 5/64 | 27/64 |
| Complete correct profile | 0/64 | 9/64 |
| Correct unit | 62/64 | 60/64 |
| Correct conditions | 62/64 | 60/64 |

The gains are concentrated in net charge, periodic-table periods and groups. Chemical formulas were correct in 0/17 cases, calculated molecular masses in 0/5, and reported atomic masses in 0/4. All six density cases failed. Aggregate overlap or perplexity therefore must not be interpreted as reliable chemical or engineering knowledge. Unit and condition accuracy also regressed slightly.

The candidate failed the 95% complete-profile readiness rule and was not activated. It is retained for review under `runs/chemistry_properties_v2/export/`. The reviewed-chat audit passed all 349 material profiles, preserving 959 source-checked fields and excluding 878 unreviewed ones. Unknown fallbacks passed 3/3; generation and streaming HTTP checks passed 6/6. The complete software suite passed 69 tests with one existing Starlette/httpx deprecation warning.
