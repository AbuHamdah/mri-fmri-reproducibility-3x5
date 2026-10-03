from pathlib import Path
O = Path(__file__).resolve().parents[1]
f = O / 'source' / 'finalize.py'
s = f.read_text(encoding='utf-8-sig')
html = '<p>Paired mean accuracy difference: +0.7937 percentage points. Descriptive paired child bootstrap95% interval: -7.9365 to +9.5238 points, including zero. Agentic is lower for seed44; ordering is not uniform. No statistically established or independent superiority is claimed.</p><p><a href="PAIRED_COMPARISON.json">Paired child comparison</a></p><img src="PER_SEED_METRICS.png" style="max-width:100%" alt="Actual per-seed accuracy recall and F1"><img src="OPTIMIZER_CURVES.png" style="max-width:100%" alt="Selected optimization traces">'
assert 'section+=' not in s
s = s.replace("body='<!doctype html>", 'section+=' + repr(html) + "\nbody='<!doctype html>")
s = s.replace("rb.replace('</html>',section+", 'rb.replace(\'</html>\',section.replace(\'href="PAIRED_COMPARISON.json"\',\'href="../\'+O.name+\'/PAIRED_COMPARISON.json"\').replace(\'src="PER_SEED_METRICS.png"\',\'src="../\'+O.name+\'/PER_SEED_METRICS.png"\').replace(\'src="OPTIMIZER_CURVES.png"\',\'src="../\'+O.name+\'/OPTIMIZER_CURVES.png"\')+')
f.write_text(s, encoding='utf-8')
