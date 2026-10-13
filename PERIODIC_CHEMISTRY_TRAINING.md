# Local periodic-table and compound training

This workflow fine-tunes the project's existing native OwnLLM checkpoint. It does not download another model or add an external inference service.

The bounded dataset covers all 118 elements (name, symbol and atomic number) and 20 common compounds (ChEBI identity and chemical formula). It does not cover every known compound or measured engineering properties. Community element data is not treated as a verified engineering database.

Sources and attribution:

- Bowserinator/Periodic-Table-JSON, CC BY-SA 3.0: https://github.com/Bowserinator/Periodic-Table-JSON . The original license and source bytes are retained locally.
- ChEBI, EMBL-EBI, CC BY 4.0: https://www.ebi.ac.uk/chebi/about/ . Formula records were acquired through the Ontology Lookup Service; source identifiers and original formula notation are retained. Conventional formula ordering was checked for identical elemental counts.

Downloads are used during dataset preparation. Training and inference run locally. Source licenses apply to their respective derived QA records; they do not establish a license for earlier user-supplied training corpora.

```powershell
python scripts/prepare_periodic_chemistry.py
python scripts/train_split_dataset.py --archive data/periodic_chemistry/chemistry_sft.zip --checkpoint runs/materials_sft_v4_v10/export/ownllm.pt --tokenizer runs/materials_sft_v4_v10/export/tokenizer.json --out runs/periodic_chemistry_v1 --provenance data/periodic_chemistry/PROVENANCE.json --epochs 4 --seq-len 256 --batch 4 --threads 4 --lr 0.0001 --max-new-tokens 160
python scripts/check_chemistry_predictions.py --run runs/periodic_chemistry_v1
python scripts/import_reviewed_chemistry.py
```

Use a new output directory for another training run; the trainer refuses to overwrite a previous run.

There are 788 training prompts, 27 validation prompts and 53 test prompts. All 138 entities occur in training. Validation/test prompts are distinct but concern known facts, so their results measure held-out wording, recall and response format, not unseen-entity generalization. The generation checks report identity fields separately because boilerplate can inflate token F1.

The selected checkpoint, exact tokenizer, configuration, source provenance, attribution, evaluation and SHA-256 manifest are saved under `runs/periodic_chemistry_v1/export/`. Previous checkpoints are preserved. The running server is not automatically switched to these weights.

The API's strict materials route continues to use reviewed local engineering records and its unknown-material fallback. Training does not automatically mark source records as verified or bypass that policy. A saved model's ability to reproduce a number does not certify that number.

The separate source-reviewed import cross-checks all 118 elemental identities against PubChem and checks the 20 compound formulas against the downloaded curated ChEBI terms. It adds identity-only records to the offline knowledge base. Query them explicitly as `element hydrogen`, `element aluminium`, or `compound sodium chloride`. These aliases preserve the existing grade-specific and generic-reference profiles. No physical-property values are inferred from identity.
