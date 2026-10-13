# Teaching materials knowledge and decisions

## Current state

The materials domain prompt already lists all 66 requested property groups. A prompt is a task specification, not training data. The existing 2.36 million parameter CPU checkpoint is an experimental baseline: its recorded evaluation produced 2/48 exact answers and 0/8 correct numerical answers. Do not treat its generated engineering values as verified.

This guide is a curriculum, not a claim that the 66 groups have been trained or validated. No new weights are produced by this document.

## What to collect for every property group

Build a coverage table with property ID (1–66), definition, units, applicable material classes, formula or empirical test method, assumptions, verified book/page or datasheet reference, and evaluation cases. Some properties require measurement or qualitative evidence rather than a universal formula. Record missing coverage explicitly.

For material values, preserve grade/composition, processing or heat treatment, temperature, pressure, loading direction, test method, uncertainty or range, and source edition/date. Use separate fields for energy and carbon footprint. Never supply a numerical default when the source gives no value. Keep current prices and regulatory information in a dated retrieval collection rather than relying on remembered weights.

Use these training task types across all 66 groups:

1. Define and explain a property, its units, and limitations.
2. Select a formula, explain its applicability, and calculate with unit conversion.
3. Compare materials using supplied, sourced values.
4. Reject candidates that violate hard requirements, then rank feasible choices using explicit user priorities.
5. Ask for missing conditions or state that evidence is insufficient.
6. Identify erroneous premises, incompatible units, and unavailable measurements.

## A worked training example

This example uses supplied hypothetical values; it asserts no real material property. It teaches constraint checking and a concise decision rationale.

```json
{"id":"decision-example-001","instruction":"Select a candidate. Require yield strength at least 250 MPa and cost at most 5 units/kg. Among feasible candidates, minimize density.","input":"Hypothetical data: A: yield=300 MPa, cost=4 units/kg, density=7800 kg/m3. B: yield=200 MPa, cost=3 units/kg, density=2700 kg/m3. C: yield=280 MPa, cost=4.5 units/kg, density=4500 kg/m3.","output":"Choose C. B fails the 250 MPa yield requirement. A and C satisfy both limits. C has the lower density: 4500 versus 7800 kg/m3. This conclusion applies only to the supplied criteria; service temperature, corrosion, fatigue and processing constraints still need evaluation.","module":"materials_selection","concept":"hard_constraints_then_ranking","source_id":"hypothetical-decision-001"}
```

Vary numbers, candidate names, requirements and wording. Include cases with no feasible candidate and cases where the preferred candidate changes. Have calculations independently checked. A thousand paraphrases of one example do not provide a thousand independent lessons.

## Dataset and training workflow

The existing trainer expects this layout:

```text
data/materials_66/
  books/                        # authorized readable PDF, EPUB, TXT, DOCX, ZIP
  dataset/
    sft_train.jsonl              # instruction/input/output/module/concept/source_id/id
    sft_validation.jsonl
    sft_test.jsonl
    evaluation.csv              # id,question,reference_answer,module,concept,difficulty,source_id
    numerical_problems.jsonl    # id,module,question,formula,working,answer
```

Keep documents/case families and near-duplicate questions in the same split. Hold out independent cases and source families where possible. The existing trainer checks exact prompt overlap, but source-family and near-duplicate separation need a dataset audit before training. Its document split alone does not guarantee that SFT and books have no shared test facts.

Prepare and inspect a new output directory first:

```powershell
python scripts/train_books.py --input data/materials_66 --out runs/materials_66_v1 --prepare-only
```

After reviewing readable-document coverage, labels, units and split integrity:

```powershell
python scripts/train_books.py --input data/materials_66 --out runs/materials_66_v1 --reuse-prepared --steps 1000 --sft-epochs 20
```

These commands use the existing small-model defaults. They demonstrate the pipeline; they are not a recipe for reliable engineering reasoning. Training currently starts from random initialization, rather than continuing the earlier checkpoint. A different model size can be supplied with `--config`; choosing it requires checking hardware and training resources.

The trainer exports weights, tokenizer, configuration, predictions and evaluation results under the selected run directory. Its exact-match/token-overlap metrics are insufficient to establish engineering correctness. Before serving a replacement, evaluate formula selection, numerical tolerance, units, conditions, evidence citations, hard-constraint satisfaction and appropriate abstention separately for every property group. Use validation for model selection and reserve the test set for final evaluation.

## Answering with evidence and calculations

For useful decisions, retrieve the relevant book/datasheet passages, compute with validated functions, and generate an explanation tied to that evidence. Require the answer to state the result, formula and units when relevant, a brief rationale, assumptions and sources. Training helps the response behavior; retrieval supplies traceable facts and calculators supply arithmetic.

The repository has retrieval components, but the chat API currently does not automatically retrieve this book collection or call a verified materials calculator. Those integrations remain implementation work. Do not assume that uploading books or training weights enables them.

## References

- NIST materials data repository: https://materialsdata.nist.gov/
- NIST thermodynamics research (properties depend on composition, temperature and pressure): https://www.nist.gov/thermodynamics
