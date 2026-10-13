import importlib.util
import json
from pathlib import Path
import sys


def test_grader_rejects_wrong_entity_definition_and_extra_properties(tmp_path,monkeypatch):
    path = Path(__file__).resolve().parents[1]/'scripts/check_property_predictions.py'
    spec = importlib.util.spec_from_file_location('property_grader',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = ('### 1. General Overview\nFixture: identity source record for Net Charge.\n\n'
                 '### 2. Engineering Profile\n* **Net Charge:**\n'
                 '  * **Knowledge:** Fixture definition.\n'
                 '  * **Typical Value / Range:** 0 (Conditions: Fixture condition.)\n'
                 '  * **Unit / Scale:** e')
    predictions = [reference.replace('Range:** 0 (','Range:** 0.0 ('),
                   reference.replace('Fixture: identity','Fixture derivative: identity'),
                   reference.replace('Fixture definition.','Wrong definition.'),
                   reference+'\n* **Extra Property:**\n  * **Knowledge:** Unsupported.']
    export = tmp_path/'export'
    export.mkdir()
    for folder in [tmp_path,export]:
        (folder/'REPORT.md').write_text('Fixture only.\n')
    (tmp_path/'predictions.json').write_text(json.dumps([
        {'id':str(i),'reference':reference,'prediction':prediction}
        for i,prediction in enumerate(predictions)]))
    monkeypatch.setattr(sys,'argv',['grader','--run',str(tmp_path)])
    module.main()
    result = json.loads((tmp_path/'property_evaluation.json').read_text())
    assert result['value_and_unit_correct']==4
    assert result['complete_source_profile_correct']==1
    assert not result['cases'][1]['entity_correct']
    assert not result['cases'][2]['knowledge_correct']
    assert not result['cases'][3]['format_correct']
