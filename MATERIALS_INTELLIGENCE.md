# Materials intelligence chat

Bare material names use the latest twelve-section master template, implemented by materials_report.py. Only supported content is included. The React UI initially expands overview, properties and final recommendation, with other sections expandable and property/comparison tables rendered as tables. Specific property questions keep the focused two-section response. Explicit improvement requests such as `Improve glass for safety` prioritize the stated objective through qualitative candidate screening.

Reviewed strategy coverage: Iron, Copper, Glass, Aluminium/Aluminum, Steel and Titanium. Other exact known names receive their stored baseline and honest gaps in improvement evidence. Unknown grades remain unknown; a steel family shows explicitly labelled 304 and 441 stainless examples, never universal steel values. Generic reference values retain their original test conditions.

No validated engineering-property predictor is installed or executed. There are no generated numerical predictions, uncertainty percentages, simulated results or claimed percentage improvements. The OwnLLM checkpoint remains loaded, while material reports use reviewed local evidence. This update changes the prompt, evidence catalogue and response routing; it does not retrain the model.

The catalogue contains short manually reviewed factual synopses from the sources below. Full articles were not imported and no automatic training-dataset rights are asserted. Provenance remains in the catalogue; chat presents explanatory content without link lists. No network access occurs while answering these material questions.

## Verification

The earlier integrated-rule results are retained in runs/integrated_rules_live_check.json. Current trial reports are in runs/materials_template_trial; the capability roadmap is MATERIALS_V3_ROADMAP.md. Tests cover all six families, both generate and stream endpoints, focused properties, ambiguous grades, unreviewed catalogue rejection, objective-sensitive ordering, unavailable predictors and missing strategies.

## Evidence reviewed 2026-10-12

- Copper Development Association: Conductivity of Specialty Alloys: https://www.copper.org/applications/industrial/DesignGuide/selection/conductspec02.php
- Copper Information Center: Copper-chromium-zirconium CuCr1Zr: https://help.copper.fyi/hc/en-us/articles/360021212740-Copper-chromium-zirconium-CuCr1Zr
- Pilkington Toughened Glass: https://www.pilkington.com/en/zz/products/product-categories/safety-security/pilkington-toughened-glass
- Pilkington Glass FAQs: toughened versus laminated glass: https://my.pilkington.com/en/knowledge-base/helpandadvice/faqs/glass-faqs
- Hydro Alloy 6061 datasheet: https://www.hydro.com/Document/Index?id=560718&name=Alloy+6061.pdf
- TWI Heat Treatment Services: https://www.twi-global.com/what-we-do/services-and-support/manufacturing-production-support/heat-treatment
- American Galvanizers Association: Corrosion Protection for Steel: https://galvanizeit.org/hot-dip-galvanizing/why-specify-galvanizing/corrosion-protection
- TIMET Designing with Titanium: https://www.timet.com/assets/local/documents/technicalmanuals/DesignandFabrication.pdf
- ORNL: High-Temperature Galling Characteristics of Ti-6Al-4V with and without Surface Treatments: https://www.ornl.gov/publications/high-temperature-galling-characteristics-of-ti-6al-4v-with-and-without-surface-treatments/
- NASA: Wear Behavior of TiC/Ti-6Al-4V with solid-film lubricant counterparts: https://ntrs.nasa.gov/citations/20070012401

## Integrated request and earlier rules

The attached proactive-analysis request is retained in domains/materials/PROACTIVE_ANALYSIS_REQUEST.md. SYSTEM_PROMPT.md merges it with the original 66-parameter taxonomy, property field structure, relevance filtering, offline fallback, verification limits and unverified-feedback handling.

The latest twelve-section format takes precedence for broad analysis; older embedded layouts are historical. Focused property questions retain the two-section profile. Unsupported property rows and table columns are omitted, while important gaps are stated once in the validation section. Known materials without reviewed improvement strategies use a shorter baseline/validation response instead of repeated filler sections.

Representative steel entries are retrieved from the existing reviewed knowledge base. They are not newly measured properties, universal steel constants or predictions for the proposed modifications. No new training or engineering predictor was added.

## Latest template trial

Iron, Glass, Copper and Titanium were rendered through the actual local policy into twelve applicable sections. Those outputs are saved in runs/materials_template_trial. The sample iron density, melting point and approximate heat capacity in the user presentation were not imported as verified values. The existing reference heat-capacity record retains its own value and conditions. Ranking for unspecified balanced objectives explicitly declines to claim one best candidate.
