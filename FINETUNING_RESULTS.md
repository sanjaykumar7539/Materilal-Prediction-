# Local fine-tuning results ? 2026-10-12

Native OwnLLM: 18,880,896 parameters; 16,384 vocabulary; 2,048-token configured context. No external pretrained model or inference service.

This campaign completed five chemistry epochs in v3, stopped a partial sixth after validation declined, then two synthetic literal-copy epochs and two additional chemistry epochs in v4. The synthetic exercises contain no chemical facts and were not added to the knowledge base. Chemistry training has 1,350 examples, validation 64, test 64. Selection uses generated validation entity/value/unit matches, then validation loss; v4 epoch 2 wins with 29/64 matches. Earlier v2 validation was 9/64.

| Existing source-provided test | v2 | v3 | v4 |
|---|---:|---:|---:|
| Correct value and unit / 64 | 23 | 26 | 26 |
| Complete correct profile / 64 | 9 | 21 | 22 |

This repeated test supplies source facts in the prompt. It does not prove independent factual recall or generalization. v4 test loss is 0.0726386204, perplexity 1.07534186. Synthetic copy test was only 12/64 exact.

Direct native-model diagnostic, without source facts: 0/8 known value/unit answers correct; 0/3 unknown exact fallbacks correct. This small diagnostic is not representative, but demonstrates that unrestricted native chat is not ready. The candidate fails the 95% complete-profile readiness threshold and was not activated in chat.

Verified offline chat separately reproduced all 349 stored material records and 959 source-checked fields, excluded 878 unreviewed fields, passed 3 unknown fallbacks and 6 generate/stream cases. This verifies local record reproduction, not a new independent review of the science. Actual Node proxy checks passed six questions. HTTP chat is available at http://127.0.0.1:3000 while the local processes run. It uses the reviewed-record policy with model_loaded=false. Start using scripts/serve_verified.cmd and frontend/start.cmd.

Software: 71 tests passed in the isolated .tools/qa_python environment. Core source and API behavior were tested; the final direct-question evaluator was separately syntax checked and executed. Earlier global Python dependency and relocated editable-install issues were corrected in this isolated environment.

Weights and parent exports are preserved. Attribution is retained; this does not assign a new license to all earlier training data or the resulting weights. Training and evaluation remained local on CPU.
