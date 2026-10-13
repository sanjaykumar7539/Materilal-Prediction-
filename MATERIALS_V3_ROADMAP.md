# Materials intelligence V3: implementation and evidence

The original user specification is retained in `domains/materials/MASTER_UPGRADE_V3.md`. This is an incremental implementation, not a completed autonomous scientist.

## Inspected before implementation

- Native OwnLLM: RMSNorm/RoPE/GQA/SwiGLU, tied embeddings, 18,880,896 parameters; active chemistry_properties_v4 tokenizer vocabulary 16,384, 384 hidden dimensions, eight layers, 2,048 context.
- Existing reviewed local property records, six-family qualitative recommendation catalog, 66-parameter taxonomy, reference imports and training/evaluation scripts.
- FastAPI generate/stream paths, security perimeter, React request construction and Markdown rendering; Gradio remains compatible with the same API.
- Latest Drive training: 450 training / 11 validation / 19 test examples; six epochs, selected epoch five. Generated exact answers 0/19. The candidate remains inactive. Lower loss does not demonstrate engineering accuracy.
- No independently validated composition/process-to-property predictor or installed simulation integration was found in the application. The reviewed KB is not a homogeneous supervised experimental training table. Element identity, calculated molecular mass and literature bounds must not be used as interchangeable measured performance targets.

## Implemented now (phases 1 and 2)

- Chemical symbols/formulas resolve from reviewed descriptors, respecting capitalization (Co differs from CO), with Unicode subscript normalization. Unknown compositions and ambiguous grades fail closed.
- Registered names, aliases, focused property questions and existing automatic improvement reports remain supported. Close ordinary-name spellings produce suggestions, never silent substitutions.
- Comparisons of up to four named materials show existing local properties with definition, value, unit, exact record and conditions. No invented mixture or percentage ranking.
- Reviewed mechanism questions use a small exact-variant corpus. Initial chromium/passivation explanation is an original factual synopsis of Outokumpu's corrosion-resistance page, reviewed 2026-10-12. This is limited question coverage, not general semantic reasoning.
- Explicit experiment-plan requests produce local proposed validation steps. No laboratory action is executed.
- The latest user template supersedes the earlier layout: broad reports use twelve numbered sections with applicable content only. Overview, properties and final recommendation are initially expanded; other details use accessible native disclosure controls. Tables render through remark-gfm. Unknown fields remain omitted. Unsupported structural, fire, cost or numerical prediction claims are omitted.
- Prior-material follow-ups support `improve it/this [for objective]`. React passes the preceding user message as an identity hint; the backend accepts only registered material names, never previous claims as evidence. Longer contextual inference remains future work.
- Sources appear as explanatory answer text with optional plain evidence titles; source URLs remain stored locally. The 66-property filtering and original focused-property format remain intact.

## Modular next stages and acceptance gates

| Phase | Remaining work | Required evidence before activation |
| --- | --- | --- |
| 1?2 | Broader reviewed materials, direct-question variants, composition parsing and longer conversational references | Reviewed provenance, ambiguity and unsupported-question regressions |
| 3 | Independent local property regression | Licensed measured rows with composition, process, test conditions, units, sample/source groups; grouped train/validation/test splits; MAE/RMSE, residuals, calibrated intervals and out-of-domain evaluation |
| 4 | Inverse design and Pareto exploration | Validated phase-3 domain, explicit constraints and reproducible evaluated candidate set; no global-optimum claim |
| 5 | Reliability, phase and simulation adapters | Installed compatible solver, material models, boundary/loading conditions and comparison with experimental benchmarks |
| 6 | SEM/XRD/curve interpretation, active learning and research | Appropriate labelled datasets, tool-specific validation and human-reviewed experimental protocols |
| 7 | Scientific acceptance and deployment | Independent held-out engineering questions, quantitative checks, regression/security tests, checkpoint provenance and explicit activation decision |

No prediction, lifetime, simulation, optimization or microscopy capability is advertised as implemented. No new weights were trained or activated in this V3 reporting change. Preserve the existing tokenizer, checkpoints, datasets and security controls.

## Verification

`tests/test_materials_v3.py` checks symbol/formula ambiguity, unknown grades/compositions, typo handling, condition-preserving comparison, direct reviewed answers, unreviewed-data rejection, contextual identity, evidence and API parity. These are software/grounding checks, not scientific validation of a predictor.

Examples: `Cu`; `Compare copper and aluminium`; `304 stainless steel vs 441 stainless steel`; `Why does chromium improve corrosion resistance in steel?`; `plan experiments for glass`; `Improve glass for safety`.
