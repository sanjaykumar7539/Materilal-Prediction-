"""Offline materials responses rendered from reviewed records, never model guesses."""
from __future__ import annotations

import ast
import json
import math
import operator
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class PropertyRecord(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    value: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    source: str = Field(min_length=1)
    conditions: str = Field(min_length=1)
    formula: str | None = Field(default=None, min_length=1)
    formula_applicable: bool = Field(default=False, strict=True)
    applicable: bool = Field(default=False, strict=True)
    verified: bool = Field(default=False, strict=True)


class BookReference(BaseModel):
    title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    relationship: str = Field(min_length=1)


class MaterialRecord(BaseModel):
    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    overview: str = Field(min_length=1)
    overview_source: str = Field(min_length=1)
    verified: bool = Field(default=False, strict=True)
    properties: list[PropertyRecord] = Field(default_factory=list)
    related_books: list[BookReference] = Field(default_factory=list)


def fallback(material: str) -> str:
    return (f"I currently do not have verified engineering data for {material} in my offline "
            "knowledge base. Please verify the material name or provide additional context.")


def _meaningful(text: str) -> bool:
    # Chemical symbols are case-sensitive: Na is sodium, not an NA placeholder.
    if text.strip() == 'Na':
        return True
    return text.strip().casefold() not in {'', 'n/a', 'na', 'unknown', 'not available', 'none', '-'}


def _safe(text: str) -> str:
    # Keep reviewed data inside its Markdown field, rather than allowing injected headings.
    return re.sub(r'([\\`*_{}\[\]<>#|])', r'\\\1', ' '.join(text.split()))



def _arithmetic_checks(text: str) -> list[str]:
    """Check explicitly written scalar arithmetic, without executing user code."""
    operations = {ast.Add:operator.add, ast.Sub:operator.sub,
                  ast.Mult:operator.mul, ast.Div:operator.truediv}

    def calculate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int,float):
            value = float(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            value = calculate(node.operand) * (-1 if isinstance(node.op,ast.USub) else 1)
        elif isinstance(node,ast.BinOp) and type(node.op) in operations:
            value = operations[type(node.op)](calculate(node.left),calculate(node.right))
        else:
            raise ValueError('Unsupported expression')
        if not math.isfinite(value) or abs(value)>1e15:
            raise ValueError('Expression outside check bounds')
        return value

    findings = []
    pattern = r'(?<![\w.])([+-]?\d+(?:\.\d+)?(?:\s*[+*/-]\s*[+-]?\d+(?:\.\d+)?)+)\s*=\s*([+-]?\d+(?:\.\d+)?)(?![\w]|\.\d)'
    for match in list(re.finditer(pattern,text))[:30]:
        expression, claimed = match.groups()
        if len(expression)>128:
            continue
        try:
            actual = calculate(ast.parse(expression,mode='eval').body)
            if not math.isclose(actual,float(claimed),rel_tol=1e-4,abs_tol=1e-6):
                findings.append(f'Arithmetic inconsistency: {expression.strip()} = {actual:g}, rather than {claimed}. Check rounding and units separately.')
        except ZeroDivisionError:
            findings.append(f'Invalid arithmetic: {expression.strip()} divides by zero.')
        except (ValueError,SyntaxError,OverflowError):
            continue
    return findings


class MaterialsPolicy:
    def __init__(self, records_path: Path, feedback_path: Path):
        self.records_path = Path(records_path)
        self.feedback_path = Path(feedback_path)

    def records(self) -> list[MaterialRecord]:
        if not self.records_path.exists():
            return []
        try:
            raw = json.loads(self.records_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return []  # Fail closed if the collection cannot be read.
        if not isinstance(raw, dict) or not isinstance(raw.get('materials'), list):
            return []
        records = []
        for item in raw['materials']:
            if not isinstance(item,dict):
                continue
            try:
                record = MaterialRecord.model_validate({**item, 'properties': []})
            except ValidationError:
                continue
            entries = item.get('properties',[])
            if isinstance(entries,list):
                for entry in entries:
                    try:
                        record.properties.append(PropertyRecord.model_validate(entry))
                    except ValidationError:
                        continue
            if record.verified and all(_meaningful(s) for s in
                                      (record.name, record.overview, record.overview_source)):
                records.append(record)
        return records

    def profile(self, prompt: str, material: str | None = None,
                parameter: str | None = None) -> str:
        query = (material or prompt).strip()
        def normalize(text):
            text = re.sub(r'\s+', ' ', text.strip()).casefold()
            return re.sub(r'\baluminium\b', 'aluminum', text)

        candidate = query.rstrip(' ?.!')
        requested = None
        property_list = False
        if not material:
            candidate = re.sub(r'^please\s+', '', candidate, flags=re.I)
            # Strip question wrappers before parsing a property, so "What are the
            # properties of copper" is not treated as an unknown property name.
            candidate = re.sub(
                r'^(?:(?:what is|what are|tell me about|describe|explain|give me|show me|report)\s+)(?:the\s+)?',
                '', candidate, flags=re.I)
            general = re.fullmatch(
                r'(?:properties|material properties|engineering properties|engineering profile|details)\s+(?:of|for|about)\s+(.+)',
                candidate, re.I)
            if general:
                candidate = general[1].strip()
                property_list = True
            else:
                specific = re.fullmatch(r'(.+?)\s+(?:of|for)\s+(.+)', candidate, re.I)
                if specific:
                    requested, candidate = specific[1].strip(), specific[2].strip()
            candidate = re.sub(r'^stainless steel\s+(\d{3,4}[A-Za-z]*)$',
                               r'\1 stainless steel', candidate, flags=re.I)
        aliases = {'formula':'Chemical Formula','molecular mass':'Calculated Molecular Mass',
                   'molecular weight':'Calculated Molecular Mass','symbol':'Chemical Symbol',
                   'young modulus':"Young's Modulus", 'youngs modulus':"Young's Modulus",
                   'elastic modulus':"Young's Modulus", 'specific heat':'Specific Heat Capacity'}
        parameter = parameter or requested
        if parameter:
            parameter = aliases.get(normalize(parameter), parameter.strip())
        matches = []
        explicit = []
        for record in self.records():
            names = [record.name, *record.aliases]
            bare = re.sub(r' \((?:element|compound|generic reference)\)$', '', record.name)
            is_explicit = any(normalize(candidate)==normalize(name) for name in names)
            if is_explicit or normalize(candidate)==normalize(bare):
                matches.append(record)
                if is_explicit:
                    explicit.append(record)
        # Explicit qualifiers and grades never fall through to another record.
        if explicit:
            matches = explicit
        if parameter and len(matches)>1:
            eligible = [r for r in matches if any(
                normalize(p.name)==normalize(parameter) and p.verified and p.applicable
                for p in r.properties)]
            if eligible:
                matches = eligible
            else:
                identities = [r for r in matches if r.name.endswith(('(element)', '(compound)'))]
                if len(identities)==1:
                    matches = identities
        elif len(matches)>1:
            preferred = [r for r in matches if r.name.endswith('(generic reference)')]
            if not property_list:
                preferred = [r for r in matches if r.name.endswith(('(element)', '(compound)'))]
            if len(preferred)==1:
                matches = preferred
        # Ambiguous grades and absent properties still fail closed.
        if len(matches) != 1:
            return fallback(query)
        record = matches[0]
        properties = [p for p in record.properties if p.verified and p.applicable
                      and all(_meaningful(s) for s in
                              (p.name, p.description, p.value, p.unit, p.source, p.conditions))
                      and (not p.formula_applicable or (p.formula and _meaningful(p.formula)))
                      and (not parameter or p.name.casefold() == parameter.strip().casefold())]
        if not properties:
            return fallback(record.name)
        # Reviewed explanatory text is stored locally; never fetch websites at chat time.
        overview = record.overview
        try:
            summaries = json.loads(self.records_path.with_name('reviewed_overviews.json').read_text(encoding='utf-8'))
            entry = summaries.get('materials', {}).get(record.name, {})
            if (entry.get('reviewed') is True and isinstance(entry.get('text'), str)
                    and _meaningful(entry['text']) and isinstance(entry.get('source'), str)
                    and _meaningful(entry['source'])):
                overview = entry['text']
        except (OSError, ValueError, AttributeError, TypeError):
            pass
        result = ['### 1. General Overview', _safe(overview),
                  '### 2. Engineering Profile']
        for prop in properties:
            lines = [f'* **{_safe(prop.name)}:**',
                     f'  * **Knowledge:** {_safe(prop.description)}',
                     f'  * **Typical Value / Range:** {_safe(prop.value)} (Conditions: {_safe(prop.conditions)})',
                     f'  * **Unit / Scale:** {_safe(prop.unit)}']
            if prop.formula and _meaningful(prop.formula):
                # Reviewed equations retain LaTeX syntax; the UI never executes HTML.
                lines.append('  * **Formula / Equation:** ' + ' '.join(prop.formula.split()))
            result.append('\n'.join(lines))
        return '\n\n'.join(result)

    def answer(self, prompt: str, material: str | None = None,
               parameter: str | None = None, previous_material: str | None = None) -> str:
        from .materials_queries import query_response
        routed = query_response(self, prompt, material, parameter, previous_material)
        if routed is not None:
            return routed
        from .materials_intelligence import analysis_response
        report = analysis_response(self, prompt, material, parameter)
        return report if report is not None else self.profile(prompt, material, parameter)

    def save_feedback(self, material: str, correction: str) -> int:
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.feedback_path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS feedback '
                       '(id INTEGER PRIMARY KEY, material TEXT, correction TEXT, created TEXT)')
            cursor = db.execute('INSERT INTO feedback(material,correction,created) VALUES(?,?,?)',
                                (material.strip().casefold(), correction.strip(),
                                 datetime.now(timezone.utc).isoformat()))
            return cursor.lastrowid

    def feedback(self, material: str) -> list[str]:
        if not self.feedback_path.exists():
            return []
        with sqlite3.connect(self.feedback_path) as db:
            rows = db.execute('SELECT correction FROM feedback WHERE material=? '
                              'ORDER BY id DESC LIMIT 10', (material.strip().casefold(),)).fetchall()
        return [row[0] for row in rows]

    def verify(self, text: str, material: str | None = None) -> str:
        observations = _arithmetic_checks(text)
        if not re.search(r'\b(source|reference|doi|https?://|page)\b', text, re.I):
            observations.append('No explicit source reference was detected; factual provenance is missing.')
        if re.search(r'\d', text) and not re.search(
                r'\b(MPa|GPa|Pa|kg|g|K|W|J|V|ohm|percent|dimensionless)\b|°|%', text, re.I):
            observations.append('Numerical content was detected without recognized units or a scale; check each value.')
        if not re.search(r'\b(temperature|pressure|grade|composition|condition|method)\b', text, re.I):
            observations.append('Material grade, test conditions and measurement method are not explicit.')
        if not observations:
            observations.append('These limited structural checks detected no missing fields. This does not establish factual correctness or logical validity.')
        result = ['I operate offline and cannot independently verify exact factual metrics against a live database.',
                  'Limited consistency checks:', *['- ' + item for item in observations],
                  'Scientific contradictions and calculation correctness require reviewed evidence and validated calculations; this check does not certify the supplied text.']
        if material:
            corrections = self.feedback(material)
            if corrections:
                result.extend(['Local user corrections (unverified; considered as context):',
                               *['- ' + _safe(c) for c in corrections]])
        return '\n\n'.join(result)
