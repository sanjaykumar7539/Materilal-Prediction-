"""Twelve-section response schema for grounded materials reports.

The schema fixes order and labels, not scientific coverage. Empty sections are omitted.
"""
from dataclasses import dataclass
import re

from .materials_policy import _safe

HEADINGS = (
    'Material Overview', 'Complete Material Properties', 'Strengths and Weaknesses',
    'Automatic Material Improvement Recommendations', 'Intelligent Material Combination Design',
    'AI Property Prediction', 'Scientific Reasoning and Explanation',
    'Original vs Modified Material Comparison', 'Engineering Optimization',
    'Failure, Reliability and Lifetime Analysis',
    'Real-World Applications and Laboratory Validation', 'Final Engineering Recommendation',
)


@dataclass(frozen=True)
class ReportSection:
    number: int
    body: str

    def __post_init__(self):
        if not 1 <= self.number <= 12 or not self.body.strip():
            raise ValueError('A report section needs a known number and nonempty evidence.')


@dataclass(frozen=True)
class MaterialReport:
    material: str
    sections: tuple[ReportSection, ...]

    def render(self):
        numbers = [s.number for s in self.sections]
        if numbers != sorted(set(numbers)):
            raise ValueError('Report sections must be unique and in template order.')
        intro = 'CAMPUSMIND MATERIALS INTELLIGENCE\n\n# ' + _safe(self.material)
        return intro + '\n\n' + '\n\n'.join(
            f'## {s.number:02d}. {HEADINGS[s.number - 1]}\n\n{s.body}' for s in self.sections)


def property_tables(profile):
    """Adapt the existing reviewed profile renderer without relaxing its filters.

    The legacy focused-question format remains stable. Each match must contain
    all three required reviewed fields; conditions remain attached to the value.
    """
    groups = {}
    pattern = (r'\* \*\*(.+?):\*\*\n  \* \*\*Knowledge:\*\* (.+)\n'
               r'  \* \*\*Typical Value / Range:\*\* (.+)\n  \* \*\*Unit / Scale:\*\* (.+)')
    for name, knowledge, value, unit in re.findall(pattern, profile):
        lower = name.lower()
        if any(w in lower for w in ('strength','hardness','modulus','ductility','elongation','toughness','fatigue')):
            group = 'Mechanical properties'
        elif any(w in lower for w in ('thermal','heat','melting','boiling','temperature')):
            group = 'Thermal properties'
        elif any(w in lower for w in ('symbol','atomic number','formula','molecular mass','charge')):
            group = 'Identity and calculated descriptors'
        else:
            group = 'Other reviewed properties'
        groups.setdefault(group, []).append(f'| {name} | {knowledge} | {value} | {unit} |')
    result = []
    for group, rows in groups.items():
        result.append('### ' + group + '\n\n| Property | Knowledge | Typical Value / Range and conditions | Unit / Scale |\n'
                      '| --- | --- | --- | --- |\n' + '\n'.join(rows))
    formulas = re.findall(r'  \* \*\*Formula / Equation:\*\* (.+)', profile)
    if formulas:
        result.append('**Reviewed equations:**\n\n' + '\n\n'.join(formulas))
    return '\n\n'.join(result)


def build_report(name, overview, property_blocks, entry, sources, objective, default_objective, identity_fields=()):
    candidates = entry['candidates'] if entry else []
    sections = []
    def add(number, body):
        if body.strip():
            sections.append(ReportSection(number, body))
    add(1, '**Material Name:** ' + _safe(name) + ''.join('\n\n**' + _safe(label) + ':** ' + _safe(value) for label, value in identity_fields) + '\n\n**Description:** ' + _safe(overview) +
        '\n\n**Assumptions:** ' + ('Use only the grades, forms and conditions explicitly identified below; an unspecified grade is not inferred.' if entry else
        'Use the identified local record only; identity descriptors do not establish engineering performance.') +
        '\n\n**Objective:** ' + _safe(objective or default_objective))
    add(2, '\n\n'.join(property_blocks))
    if not candidates:
        add(12, 'No reviewed candidate combination is available for this material; the response therefore includes the supported baseline only. '
            'No validated engineering-property prediction model is installed or executed. '
            'Identify composition, processing and service requirements and obtain reviewed candidate evidence before recommending a modification.')
        return MaterialReport(name, tuple(sections)).render()
    add(3, '**Identified limitations:** ' + _safe(entry['weaknesses']))
    table = ['| Target | Recommended modification | Expected benefit | Trade-off |', '| --- | --- | --- | --- |']
    table += ['| ' + ' | '.join(_safe(x) for x in (', '.join(c['objectives']), c['name'], c['benefit'], c['tradeoff'])) + ' |' for c in candidates]
    add(4, '\n'.join(table))
    add(5, '\n\n'.join(f"### Candidate {i}: {_safe(c['name'])}\n\n**Change / processing:** {_safe(c['change'])}\n\n**Applicability:** {_safe(c['conditions'])}"
                         for i, c in enumerate(candidates, 1)))
    add(6, '**Prediction status:** No validated engineering-property prediction model is installed or executed. '
        'Numerical candidate predictions, intervals, probabilities and percentage improvements are not available. '
        'The following expectations are qualitative literature-supported directions; their transfer to this component remains a hypothesis.\n\n' +
        '\n'.join('- **' + _safe(c['name']) + ':** ' + _safe(c['benefit']) for c in candidates))
    add(7, '\n\n'.join('**' + _safe(c['name']) + ':** ' + _safe(c['mechanism']) +
                          ' **Processing and service limits:** ' + _safe(c['conditions']) for c in candidates))
    table = ['| Original baseline / scope | Modified candidate | Expected change | Important trade-off | Evidence type |',
             '| --- | --- | --- | --- | --- |']
    baseline = entry.get('baseline') or name + ' family; grade unspecified'
    table += ['| ' + ' | '.join(_safe(x) for x in (baseline, c['name'], c['benefit'], c['tradeoff'], c['evidence_kind'])) + ' |' for c in candidates]
    add(8, '\n'.join(table) + '\n\nQualitative comparison only. Candidate-specific grades and applicability in section 05 govern this comparison; no matched before/after measurements are available.')
    requested = set(re.findall(r'[a-z]+', (objective or '').lower()))
    if 'mechanical' in requested: requested.add('strength')
    if 'electrical' in requested: requested.add('conductivity')
    scores = [(len(requested.intersection(c.get('objectives', []))), i, c) for i,c in enumerate(candidates)]
    ranked = sorted(scores, key=lambda item: (-item[0], item[1]))
    winner = ranked[0][2] if objective and ranked[0][0] > 0 and (len(ranked) == 1 or ranked[0][0] > ranked[1][0]) else None
    if objective and not ranked[0][0]:
        selection = 'The stated objective has no matching reviewed candidate evidence. The options address other objectives.'
    elif winner:
        selection = ('**Candidate to investigate first:** ' + _safe(winner['name']) +
                     '\n\nThe objective matches its reviewed qualitative benefit. This is a screening suggestion, not a validated optimal design.\n\n' +
                     '\n'.join('- **' + _safe(c['name']) + ':** relevant to ' + _safe(', '.join(c['objectives'])) + '.' for _,_,c in ranked))
    else:
        selection = ('No unique best candidate is established for the balanced or tied objectives. Define the service priorities and constraints before ranking.\n\n' +
                     '\n'.join('- **' + _safe(c['name']) + ':** relevant to ' + _safe(', '.join(c['objectives'])) + '.' for c in candidates))
    add(9, '**Primary engineering objective:** ' + _safe(objective or default_objective) + '\n\n' + selection +
        '\n\nQuantitative cost and multi-objective optimization have not been performed.')
    add(10, '**Relevant limitations:** ' + _safe(entry['weaknesses']) + '\n\n' +
        '\n'.join('- **' + _safe(c['name']) + ':** ' + _safe(c['tradeoff']) for c in candidates) +
        '\n\n**Operating constraints:** ' + ' '.join(_safe(c['conditions']) for c in candidates) +
        '\n\nService life is not estimated from these data. Corrosion, oxidation, thermal transport and fire performance require distinct assessments.')
    add(11, '**Recommended validation:** ' + _safe(entry['validation']) +
        '\n\nCompare matched baseline and candidate specimens. Record grade, composition, processing, geometry, exposure, '
        'test method, units and uncertainty. Choose acceptance criteria and replicates before testing. '
        'This is a proposed plan; no experiment or simulation has been executed.\n\n'
        '**Needed for stronger conclusions:** Candidate-specific measurements under the intended operating conditions.')
    refs = dict.fromkeys(ref for c in candidates for ref in c['sources'])
    evidence = '\n'.join('- ' + _safe(sources[ref].get('title', ref)) + ' (reviewed ' +
                         _safe(sources[ref].get('review_date', '')) + ')' for ref in refs)
    final = ('**Best supported next step:** ' + (_safe(winner['change']) if winner else
             'Select the application and its acceptance criteria, then compare the applicable options above. A universal winner is not established.') +
             '\n\n**Confidence and evidence:** Baseline values are retrieved local reference data, including identity or calculated descriptors where labelled. '
             'Candidate mechanisms are reviewed literature summaries; they are not measured improvements in this component.\n\n' + evidence +
             '\n\n**Next step:** ' + _safe(entry['validation']))
    add(12, final)
    return MaterialReport(name, tuple(sections)).render()
