"""Offline evidence-based material improvement reports; no property predictor is installed."""
from __future__ import annotations

import json
import re

from .materials_policy import _safe, fallback

from .materials_report import HEADINGS, build_report, property_tables

DEFAULT_OBJECTIVE = ('Balanced mechanical performance, environmental resistance, relevant thermal '
                     'performance, durability, manufacturability and reasonable cost.')



def _normal(text):
    return re.sub(r'\s+', ' ', text.strip().rstrip(' ?.!')).casefold()


def _load_catalog(policy):
    try:
        catalog = json.loads(policy.records_path.with_name('intelligence_catalog.json').read_text(encoding='utf-8'))
        if not isinstance(catalog, dict) or not isinstance(catalog.get('sources'), dict):
            return {}, {}
        entries = catalog.get('materials', {})
        if not isinstance(entries, dict):
            return {}, {}
        sources = catalog['sources']
        checked = {}
        for name, entry in entries.items():
            if not isinstance(entry, dict) or entry.get('reviewed') is not True:
                continue
            if not all(isinstance(entry.get(k), str) and entry[k].strip()
                       for k in ('overview','weaknesses','validation')):
                continue
            refs = entry.get('sources', [])
            if not refs or not all(isinstance(sources.get(ref), dict) and sources[ref].get('url')
                                   and sources[ref].get('review_date') for ref in refs):
                continue
            candidates = []
            for candidate in entry.get('candidates', []):
                if not isinstance(candidate, dict) or candidate.get('reviewed') is not True:
                    continue
                if not all(isinstance(candidate.get(k), str) and candidate[k].strip()
                           for k in ('name','change','benefit','tradeoff','mechanism','conditions','evidence_kind')):
                    continue
                if not candidate.get('sources') or not all(ref in sources for ref in candidate['sources']):
                    continue
                candidates.append(candidate)
            checked[name] = {**entry, 'candidates': candidates}
        return checked, sources
    except (OSError, ValueError, TypeError, AttributeError):
        return {}, {}


def analysis_response(policy, prompt: str, material=None, parameter=None):
    """Return a report for a bare name or explicit improvement request, else None.

    None means use the existing focused property/profile path. Unknown entities
    still use the exact offline fallback. No unreviewed prompt facts are admitted.
    """
    if parameter:
        return None
    query = (material or prompt).strip()
    objective = None
    improve = re.fullmatch(
        r'(?:how (?:can|do) (?:i|we) improve|how to improve|improve|analyse|analyze)\s+(.+?)(?:\s+for\s+(.+))?',
        prompt.strip().rstrip(' ?.!'), re.I)
    if improve:
        query = material or improve[1]
        objective = improve[2]
    elif material and _normal(prompt) != _normal(material):
        return None
    catalog, sources = _load_catalog(policy)
    selected = [(name, entry) for name,entry in catalog.items()
                if _normal(query) in {_normal(name), *[_normal(a) for a in entry.get('aliases', [])]}]
    entry = selected[0][1] if len(selected)==1 else None
    if entry:
        name = selected[0][0]
        baseline = entry.get('baseline')
        profile = policy.profile(baseline, material=baseline) if baseline else None
    else:
        # Recognize exact registered names only; never infer an alloy from a substring.
        records = policy.records()
        names = { _normal(n) for record in records for n in
                  [record.name, *record.aliases,
                   re.sub(r' \((?:element|compound|generic reference)\)$','',record.name)] }
        if _normal(query) not in names:
            return fallback(query) if improve else None
        name = query
        profile = policy.profile(query)
        if profile.startswith('I currently do not have verified'):
            return profile
    overview = entry['overview'] if entry else profile.split('### 1. General Overview\n\n',1)[-1].split('\n\n### 2.',1)[0]
    property_blocks = []
    if profile and '### 2. Engineering Profile\n\n' in profile:
        property_blocks.append(property_tables(profile))
    # Representative grades are labelled examples, never a substituted universal baseline.
    if entry:
        records = {r.name: r for r in policy.records()}
        for example in entry.get('representatives', []):
            grade = example.get('grade')
            if grade not in records:
                continue
            report = policy.profile(grade, material=grade)
            if '### 2. Engineering Profile\n\n' not in report:
                continue
            body = property_tables(report)
            property_blocks.append('#### Representative grade: '+_safe(grade)+'\n\n'+
                                   _safe(example['reason'])+' These values apply only to this grade and the conditions below.\n\n'+body)
    if property_blocks:
        property_blocks[0] = ('Retrieved local reference data; values retain their definitions, units and conditions. '
                              'Representative grades are examples, not universal family properties.\n\n' + property_blocks[0])
    from .materials_queries import valid_properties
    identity_fields = []
    categories = {'Iron': 'Metal', 'Copper': 'Metal', 'Aluminium': 'Metal',
                  'Titanium': 'Metal', 'Steel': 'Alloy family', 'Glass': 'Glass; composition unspecified'}
    if name in categories:
        identity_fields.append(('Material Category', categories[name]))
    identities = [r for r in policy.records() if r.name.endswith(('(element)', '(compound)'))
                  and _normal(name) in {_normal(re.sub(r' \((?:element|compound)\)$', '', r.name)),
                                       _normal(r.name), *[_normal(a) for a in r.aliases]}]
    if len(identities) == 1:
        for prop in valid_properties(identities[0]):
            if prop.name in ('Chemical Formula', 'Chemical Symbol'):
                identity_fields.append((prop.name, prop.value))
    return build_report(name, overview, property_blocks, entry, sources, objective, DEFAULT_OBJECTIVE, identity_fields)
