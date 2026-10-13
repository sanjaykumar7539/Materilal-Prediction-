# Chemistry properties, reading references and local training

This expansion uses the project's native 18,880,896-parameter OwnLLM model. No external pretrained model or inference API is integrated. Network access is used only for source acquisition.

The prepared collection contains 118 elements with 878 available source-aggregated property entries, and 200 curated ChEBI compounds with 600 descriptors. Element entries include reported atomic mass, electron configuration, period/group, available melting/boiling or triple-point temperatures, density and molar heat capacity. Missing values are omitted. Source documentation establishes Kelvin temperatures, phase-dependent density units and gas-density conditions. Measurement conditions that are not established remain explicit; 72 unconfirmed entries are not presented as verified experimental data.

ChEBI descriptors are formula, calculated average molecular mass and formal net charge. These do not imply solubility, toxicity, phase-transition temperatures or mechanical performance. Existing conventional formulas such as NaCl are retained in the offline profiles while original ChEBI notation remains in the source archive.

Two exact OpenStax editions are stored locally: *Chemistry* (1,355 PDF pages) and *Chemistry: Atoms First* (1,357 PDF pages). Their embedded licenses declare CC BY 4.0. The first credits Paul Flowers, Klaus Theopold, Richard Langley and William R. Robinson; the second also credits Edward J. Neth. Later website editions may have different licenses. Exact URLs, licenses, hashes and attribution are in `data/chemistry_expansion/PROVENANCE.json` and `ATTRIBUTION.md`.

The element-to-book index distinguishes general chemistry background from keyword occurrences. There are keyword matches for 103 elements; the other 15 have general references only. A keyword match is not proof of substantive element-specific discussion or a source for a particular numerical property. This collection does not supply 118 separate books.

The offline knowledge base now has 349 records and 1,837 property entries. It includes 959 source-checked/reference descriptors and 878 unreviewed element-property entries. The latter retain `verified: false` and are omitted by the strict materials policy. Element profiles include related reading within the General Overview. Query `element hydrogen`, `compound water`, or `compound sodium chloride`. Unsupported engineering parameters still receive the existing fallback.

```powershell
python scripts/prepare_chemistry_expansion.py
python scripts/import_chemistry_expansion.py
python scripts/train_corpus_jsonl.py --input data/chemistry_expansion/book_corpus.jsonl --base-export runs/periodic_chemistry_v1/export --new-corpus --exclude-seen-input data/corpus_v4_v10/combined.jsonl --out runs/chemistry_books_v1 --steps 250 --seq-len 256 --batch 1 --threads 4 --lr 0.00005 --eval-batches 32
python scripts/train_split_dataset.py --archive data/chemistry_expansion/property_sft.zip --checkpoint runs/chemistry_books_v1/export/ownllm.pt --tokenizer runs/chemistry_books_v1/export/tokenizer.json --out runs/chemistry_properties_v1 --provenance data/chemistry_expansion/PROVENANCE.json --epochs 2 --seq-len 416 --batch 4 --threads 4 --lr 0.0001 --max-new-tokens 224
python scripts/check_property_predictions.py --run runs/chemistry_properties_v1
```

Use fresh output directories for additional runs; previous checkpoints are preserved.

Book continuation uses 250 sampled batches of 256 tokens (64,000 target-token presentations), rather than a full epoch over both books. Source groups are chapters within related books, not independent books; chapter splits can share facts and near-duplicate passages. Exact prior-corpus and within-collection duplicate chunks are removed where detected. Perplexity evaluation uses 32 sampled windows per held-out split.

Property fine-tuning uses 1,350 training, 64 validation and 64 test questions. Every prepared example fits the 416-token limit. Each question supplies its source value, unit and conditions; evaluation therefore measures source-grounded reading and formatting, not closed-book recall or scientific reasoning. Generated value/unit and complete-profile checks are reported separately from lexical overlap.

Final model weights, exact tokenizer, configuration and evaluations are saved under `runs/chemistry_properties_v1/export/`. The server is not automatically switched to this checkpoint. The verified offline knowledge path remains active independently of model quality.
