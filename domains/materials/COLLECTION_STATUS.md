# Source collection status — 2026-10-09

The current local collection contains 349 records and 1,837 property entries: 14 manufacturer stainless-steel grades, 17 generic textbook references, 118 elements and 200 ChEBI compounds. Of these entries, 959 are source-checked/reference descriptors and 878 elemental properties remain unverified and are omitted from verified answers. Calculated molecular mass and formal charge are not measured engineering-performance data. This is not a complete export of all referenced databases.

| Source | Status | Remaining work |
|---|---|---|
| Outokumpu Core datasheet | All 14 grade rows represented; selected mechanical and physical properties checked from Tables 5 and 7 | Other product forms, tables, curves and narrative claims require additional condition-specific review |
| OpenStax University Physics, stored 2018 editions | 17 generic reference materials, 29 selected properties reviewed from Volume 1 Table 12.1 and Volume 2 Tables 1.3/1.5; embedded CC BY 4.0 declarations and PDF hashes checked | Textbook approximations only; use `generic copper`, `generic aluminium`, etc. No alloy grade inferred; unknown composition, test method and elastic measurement temperature remain explicit |
| PubChem periodic-table export | All 118 names, symbols and atomic numbers cross-checked against the training identity collection; only these identity fields imported | Use `element copper`, `element aluminium`, etc. No physical values or predicted properties imported. Source: https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/JSON |
| ChEBI via EMBL-EBI OLS | 200 curated compounds; formulas, calculated average masses and formal charges checked against non-obsolete 3_STAR terms; CC BY 4.0 | Use `compound water`, `compound sodium chloride`, etc. Original formula notation retained; no phase-dependent physical values inferred. Source: https://www.ebi.ac.uk/chebi/about/ |
| Community element-property collection | 878 source-aggregated entries covering all 118 elements, including 72 with unconfirmed measurement/confirmation status; CC BY-SA 3.0 | Stored for labelled training/reference use; verified flag remains false. Gas-density conditions and carbon/arsenic triple-point exceptions retained. Source: https://github.com/Bowserinator/Periodic-Table-JSON |
| OpenStax Chemistry and Chemistry: Atoms First, exact stored editions | Two PDFs, 2,712 pages, embedded CC BY 4.0 declarations checked; per-element reading index | Keyword matches for 103 elements; 15 have general background references only. A keyword match is not a verified discussion, and these are not 118 separate element-specific books. See element_book_index.json |
| KME copper datasheets | Inspected but excluded from structured ingestion | Legal notices restrict offered downloads to original personal-use copies and require prior written consent for other electronic-system use: https://www.kme.com/en/legal-notices/ |
| NIST Chemistry WebBook | Site and source documentation inspected; no bulk export imported | Obtain an authorized dataset/export and preserve original references, uncertainties and property conditions; SRD copyright applies |
| NIST Alloy Data | Experimental-data provenance described by the publisher; JavaScript interface | Obtain the supported export or published datasets; preserve composition, uncertainty, experimental method and temperature |
| Special Metals | Document catalog and terms inspected; no bulk collection performed | Publisher terms prohibit automated copying; permission or a suitable licensed export is required for this bulk acquisition |
| NIST general repository | Public repository inspected | Review each collection's license, provenance and review status individually; repository membership alone is not verification |
| Materials Project | Not imported | Keep computational predictions distinct from measured engineering data; requires its own acquisition and license review |
| MatWeb | Excluded from AI ingestion | Current terms prohibit AI training, evaluation and embeddings |

Source checks establish faithful transcription, not independent physical certification. Importing a raw record never automatically sets its verified flag. Broad identifiers shared by different variants intentionally produce the existing ambiguity fallback.

Source URLs: https://webbook.nist.gov/chemistry/ ; https://trc.nist.gov/MetalsAlloyUI/ ; https://www.specialmetals.com/terms-of-use.html ; https://materialsdata.nist.gov/ ; https://www.matweb.com/reference/terms.aspx
