"""Conservative local material intent handling; no numerical prediction or online lookup."""
from __future__ import annotations

import json
import re
from difflib import get_close_matches

from .materials_policy import _meaningful, _safe, fallback


def valid_properties(record):
    return [p for p in record.properties if p.verified and p.applicable
            and all(_meaningful(x) for x in (p.name, p.description, p.value, p.unit, p.source, p.conditions))
            and (not p.formula_applicable or (p.formula and _meaningful(p.formula)))]


def normalize(text):
    return re.sub(r"\s+", " ", text.strip().rstrip(" ?.!")).casefold().replace('aluminium', 'aluminum')


def resolve_name(policy, query):
    """Resolve registered symbols/formulas case-sensitively; never infer compositions."""
    query = query.strip().rstrip(' ?.!').translate({0x2080 + i: ord('0') + i for i in range(10)})
    records = policy.records()
    chemical = {}
    for record in records:
        for prop in valid_properties(record):
            if prop.name in ('Chemical Symbol', 'Chemical Formula'):
                chemical.setdefault(prop.value, set()).add(record.name)
    if query in chemical:
        matches = chemical[query]
        if len(matches) != 1:
            return None
        return re.sub(r' \((?:element|compound)\)$', '', next(iter(matches)))
    # Preserve formula capitalization: Co and CO cannot be interchanged.
    if re.fullmatch(r'(?:[A-Z][a-z]?\d*)+(?:[+-])?', query):
        return None
    return query


def registered(policy, name):
    from .materials_intelligence import _load_catalog
    catalog, _ = _load_catalog(policy)
    names = {normalize(x) for key, entry in catalog.items() for x in [key, *entry.get('aliases', [])]}
    for record in policy.records():
        names.update(normalize(x) for x in [record.name, *record.aliases,
                     re.sub(r' \((?:element|compound|generic reference)\)$', '', record.name)])
    return normalize(name) in names


def comparison(policy, names):
    from .materials_intelligence import _load_catalog
    catalog, _ = _load_catalog(policy)
    rows, overviews = [], []
    records = policy.records()
    for original in names:
        name = resolve_name(policy, original)
        if not name or not registered(policy, name):
            return fallback(original)
        candidates = [r for r in records if normalize(name) in {normalize(r.name),
                      normalize(re.sub(r' \((?:element|compound|generic reference)\)$', '', r.name)),
                      *[normalize(a) for a in r.aliases]}]
        # A generic reference is explicitly labelled, rather than silently mixing grades.
        preferred = [r for r in candidates if r.name.endswith('(generic reference)')]
        explicit = [r for r in candidates if normalize(r.name) == normalize(name)]
        candidates = explicit or preferred or candidates
        if len(candidates) != 1:
            return fallback(original)
        record = candidates[0]
        overviews.append(f"**{_safe(record.name)}:** {_safe(record.overview)}")
        for prop in valid_properties(record):
            rows.append('| ' + ' | '.join(_safe(x) for x in
                        (record.name, prop.name, prop.description, prop.value, prop.unit, prop.conditions)) + ' |')
    if not rows:
        return fallback(', '.join(names))
    return ('### Material comparison\n\n' + '\n\n'.join(overviews) +
            '\n\n| Material / grade | Property | Knowledge | Value / range | Unit / scale | Conditions |\n'
            '| --- | --- | --- | --- | --- | --- |\n' + '\n'.join(rows) +
            '\n\n**Evidence:** Retrieved local reference values; identity and calculated descriptors are not experimental performance measurements. '
            'This is a comparison of available records, not a prediction of a mixture. '
            'Different temperatures, product forms, treatments and value types prevent direct ranking. '
            'No percentage improvement or overall winner is inferred.')


def focused_answer(policy, prompt):
    """Only explicitly reviewed question variants are answered by the mechanism corpus."""
    try:
        entries = json.loads(policy.records_path.with_name('reviewed_questions.json').read_text(encoding='utf-8'))['questions']
        for entry in entries:
            if (entry.get('reviewed') is True and entry.get('source_url') and entry.get('review_date')
                    and isinstance(entry.get('answer'), str) and _meaningful(entry['answer'])
                    and normalize(prompt) in {normalize(q) for q in entry.get('questions', [])}):
                return _safe(entry['answer']) + '\n\n**Evidence:** Reviewed literature explanation; no property prediction was executed.'
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return None


def query_response(policy, prompt, material=None, parameter=None, previous_material=None):
    from .materials_intelligence import analysis_response, _load_catalog
    if material or parameter:
        return None
    direct = focused_answer(policy, prompt)
    if direct:
        return direct
    text = prompt.strip().rstrip(' ?.!')
    # Only an explicitly supplied, registered prior material can resolve a pronoun.
    followup = re.fullmatch(r'(?:improve|analyse|analyze) (?:it|this)(?: for (.+))?', text, re.I)
    if followup:
        prior = resolve_name(policy, previous_material) if previous_material else None
        if not prior or not registered(policy, prior):
            return 'Please name the material or exact grade for this follow-up.'
        return analysis_response(policy, 'improve ' + prior +
                                 (' for ' + followup[1] if followup[1] else ''))
    body = re.sub(r'^compare\s+', '', text, flags=re.I)
    parts = re.split(r'\s+(?:versus|vs\.?|and)\s+|\s*,\s*', body, flags=re.I)
    if len(parts) > 1:
        if len(parts) > 4:
            return 'Compare up to four named materials at a time, with their grades where available.'
        return comparison(policy, parts)
    experiment = re.fullmatch(r'(?:plan experiments for|experiment plan for|how to test)\s+(.+)', text, re.I)
    if experiment:
        name = resolve_name(policy, experiment[1])
        catalog, _ = _load_catalog(policy)
        entries = [(key, entry) for key, entry in catalog.items() if name and normalize(name) in
                   {normalize(key), *[normalize(a) for a in entry.get('aliases', [])]}]
        if len(entries) != 1:
            return fallback(experiment[1])
        key, entry = entries[0]
        return ('### Proposed validation plan: ' + _safe(key) + '\n\n' + _safe(entry['validation']) +
                '\n\n1. Fix the grade, composition, processing, geometry and intended exposure.\n'
                '2. Compare an untreated baseline with each selected candidate under matching conditions.\n'
                '3. Choose an applicable test method, calibration, replicates and acceptance criteria before testing.\n'
                '4. Record raw observations, units, uncertainty and failed specimens; retain provenance.\n'
                '5. Review the results before accepting a design change.\n\n'
                '**Status:** Proposed plan only. No experiment or simulation has been executed; a qualified reviewer must approve the actual protocol.')
    resolved = resolve_name(policy, text)
    if resolved is None:
        return fallback(text)
    if resolved != text:
        return analysis_response(policy, resolved) or policy.profile(resolved)
    # Never silently autocorrect a chemical formula or grade. Suggest only ordinary names.
    if re.fullmatch(r'[a-zA-Z]{5,25}', text) and not registered(policy, text):
        catalog, _ = _load_catalog(policy)
        vocabulary = {normalize(re.sub(r' \((?:element|compound|generic reference)\)$', '', r.name))
                      for r in policy.records()}
        vocabulary.update(normalize(x) for x in catalog)
        suggestions = get_close_matches(normalize(text), sorted(vocabulary), n=2, cutoff=0.84)
        if suggestions:
            return fallback(text) + '\n\nPossible spelling: ' + ', '.join(_safe(x) for x in suggestions) + '. Please confirm the material name.'
    return None
