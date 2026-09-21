"""Build the dependency-free Pages report from the published score tables."""
import csv
import html
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / 'docs'

DATASET_METADATA = {
    'AG News': {
        'kind': 'text',
        'domain': 'Text',
        'classes': 4,
        'testRows': 1000,
        'description': 'Topic classification across 4 news categories: World, Sports, Business, and Sci/Tech.',
        'classesDetail': '4 classes: World (25%), Sports (25%), Business (25%), Sci/Tech (25%)',
        'metrics': 'Balanced Accuracy (unweighted average recall across classes)',
    },
    'Banking77': {
        'kind': 'text',
        'domain': 'Text',
        'classes': 77,
        'testRows': 1500,
        'description': 'Fine-grained customer service intent classification across 77 online banking categories.',
        'classesDetail': '77 fine-grained intent classes in banking customer support',
        'metrics': 'Balanced Accuracy (majority baseline = 1.3%)',
    },
    'SMS Spam': {
        'kind': 'text',
        'domain': 'Text',
        'classes': 2,
        'testRows': 1000,
        'description': 'Mobile phone SMS spam detection on imbalanced communication messages.',
        'classesDetail': '2 classes: Ham (legitimate) vs. Spam',
        'metrics': 'Balanced Accuracy (majority baseline = 50.0%)',
    },
    'IMDb': {
        'kind': 'text',
        'domain': 'Text',
        'classes': 2,
        'testRows': 1000,
        'description': 'Movie review sentiment polarity classification from long-form user reviews.',
        'classesDetail': '2 classes: Positive vs. Negative',
        'metrics': 'Balanced Accuracy (majority baseline = 50.0%)',
    },
    'Bank Marketing': {
        'kind': 'tabular',
        'domain': 'Tabular',
        'classes': 2,
        'testRows': 1000,
        'description': 'Direct marketing phone campaign prediction of client term deposit subscriptions.',
        'classesDetail': '2 classes: Subscribed (yes) vs. Not subscribed (no)',
        'metrics': 'Balanced Accuracy with demographic, economic, and contact features',
    },
    'Online Shoppers': {
        'kind': 'tabular',
        'domain': 'Tabular',
        'classes': 2,
        'testRows': 1000,
        'description': 'E-commerce session revenue purchasing intention from real-time web browsing analytics.',
        'classesDetail': '2 classes: Purchase (True) vs. No purchase (False)',
        'metrics': 'Balanced Accuracy across numerical web session attributes',
    },
    'Breast Cancer': {
        'kind': 'tabular',
        'domain': 'Tabular',
        'classes': 2,
        'testRows': 114,
        'description': 'Wisconsin diagnostic clinical features for malignant versus benign tumor classification.',
        'classesDetail': '2 classes: Malignant vs. Benign (small holdout n=114)',
        'metrics': 'Balanced Accuracy across 30 nuclear feature dimensions',
    },
    'Iris': {
        'kind': 'tabular',
        'domain': 'Tabular',
        'classes': 3,
        'testRows': 30,
        'description': 'Morphometric iris flower species classification from sepal/petal measurements.',
        'classesDetail': '3 classes: Setosa, Versicolor, Virginica (n=30 holdout)',
        'metrics': 'Balanced Accuracy across 4 morphometric measurements',
    },
}

def load_data():
    data = {}
    for panel in ('raw', 'adjusted'):
        path = ROOT / 'published_results' / f'{panel}_balanced_accuracy.csv'
        with path.open(encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            models = reader.fieldnames[1:]
            rows = []
            for row in reader:
                ds_name = row['dataset']
                meta = DATASET_METADATA.get(ds_name, {
                    'kind': 'text', 'domain': 'Text', 'classes': 2, 'testRows': 1000,
                    'description': '', 'classesDetail': '', 'metrics': 'Balanced Accuracy'
                })
                scores = {}
                for model in models:
                    match = re.fullmatch(r'(\d+\.\d) ± (\d+\.\d) \(n=(\d+)\)', row[model])
                    assert match, row[model]
                    mean, sd, n = match.groups()
                    scores[model] = dict(mean=float(mean), sd=float(sd), n=int(n), display=row[model])
                
                classical = models[:11]
                best_val = max(scores[m]['mean'] for m in classical)
                best_models = [m for m in classical if scores[m]['mean'] == best_val]
                
                jev_zero = scores['Jev zero-shot']['mean']
                jev_few = scores['Jev few-shot']['mean']
                delta = jev_zero - best_val
                
                rows.append({
                    'dataset': ds_name,
                    'kind': meta['kind'],
                    'domain': meta['domain'],
                    'classes': meta['classes'],
                    'testRows': meta['testRows'],
                    'description': meta['description'],
                    'classesDetail': meta['classesDetail'],
                    'metrics': meta['metrics'],
                    'scores': scores,
                    'best': best_val,
                    'bestModels': best_models,
                    'delta': delta,
                })
        data[panel] = dict(models=models, rows=rows)
    return data

def summary_table(panel):
    out = [
        '<caption class="sr-only">Main benchmark comparison: Jev vs best classical pipeline by dataset</caption>',
        '<thead><tr>',
        '<th scope="col" class="th-dataset">Dataset</th>',
        '<th scope="col" class="th-domain">Domain</th>',
        '<th scope="col" class="th-classes text-right">Classes</th>',
        '<th scope="col" class="th-test text-right">Test N</th>',
        '<th scope="col" class="th-score text-right">Jev Zero-shot</th>',
        '<th scope="col" class="th-score text-right">Jev Few-shot</th>',
        '<th scope="col" class="th-best text-right">Best Classical</th>',
        '<th scope="col" class="th-delta text-right">Δ (Zero vs Best)</th>',
        '<th scope="col" class="th-action text-center"><span class="sr-only">Details</span></th>',
        '</tr></thead>',
        '<tbody>'
    ]
    for row in panel['rows']:
        d_val = row['delta']
        delta_cls = 'delta-pos' if d_val > 0 else ('delta-neg' if d_val < 0 else 'delta-neutral')
        delta_str = f'+{d_val:.1f}%' if d_val > 0 else f'{d_val:.1f}%'
        best_model_name = ', '.join(row['bestModels'])
        
        out.append(f'''<tr class="summary-row" data-dataset="{html.escape(row['dataset'])}" tabindex="0" role="button" aria-expanded="false" title="Click to view full model scores and metadata for {html.escape(row['dataset'])}">
  <td class="font-medium text-foreground"><span class="ds-name">{html.escape(row['dataset'])}</span></td>
  <td><span class="badge-domain badge-{row['kind']}">{row['domain']}</span></td>
  <td class="text-right font-mono">{row['classes']}</td>
  <td class="text-right font-mono">{row['testRows']:,}</td>
  <td class="text-right font-mono font-semibold {'score-winner' if row['scores']['Jev zero-shot']['mean'] >= row['best'] else ''}">{row['scores']['Jev zero-shot']['mean']:.1f}% <span class="sd-sub">±{row['scores']['Jev zero-shot']['sd']:.1f}</span></td>
  <td class="text-right font-mono">{row['scores']['Jev few-shot']['mean']:.1f}% <span class="sd-sub">±{row['scores']['Jev few-shot']['sd']:.1f}</span></td>
  <td class="text-right font-mono font-semibold {'score-winner' if row['best'] > row['scores']['Jev zero-shot']['mean'] else ''}">{row['best']:.1f}% <span class="best-model-label">({html.escape(best_model_name)})</span></td>
  <td class="text-right font-mono"><span class="delta-chip {delta_cls}">{delta_str}</span></td>
  <td class="text-center"><span class="row-chevron" aria-hidden="true">▾</span></td>
</tr>
<tr class="detail-row hidden" id="detail-{re.sub(r'[^a-zA-Z0-9]', '-', row['dataset']).lower()}">
  <td colspan="9" class="detail-cell">
    <div class="detail-content">
      <div class="detail-header">
        <div>
          <h4 class="detail-title">{html.escape(row['dataset'])} <span class="detail-badge">{row['domain']} · {row['classes']} classes · {row['testRows']:,} test rows</span></h4>
          <p class="detail-desc">{html.escape(row['description'])}</p>
        </div>
        <div class="detail-meta-group">
          <span class="meta-tag">Holdout Seed: <code>20260920</code></span>
          <span class="meta-tag">Training Seeds: <code>2027, 2028, 2029</code></span>
        </div>
      </div>
      <div class="detail-scores-grid">''')
        
        # Grid of all 14 models
        for m in panel['models']:
            s = row['scores'][m]
            is_best = (s['mean'] == max(sc['mean'] for sc in row['scores'].values()))
            is_jev = m.startswith('Jev')
            card_cls = 'best-card' if is_best else ('jev-card' if is_jev else '')
            out.append(f'''<div class="score-card {card_cls}">
          <span class="score-model-name">{html.escape(m)}</span>
          <span class="score-val font-mono">{s['mean']:.1f}%</span>
          <span class="score-sd font-mono">±{s['sd']:.1f} (n={s['n']})</span>
        </div>''')
            
        out.append('''      </div>
    </div>
  </td>
</tr>''')
    out.append('</tbody>')
    return ''.join(out)

def full_table(panel):
    models = ['Jev zero-shot', 'Jev few-shot'] + [m for m in panel['models'] if not m.startswith('Jev ')]
    out = [
        '<caption class="sr-only">Raw balanced accuracy: mean ± sample standard deviation, three seeds.</caption>',
        '<thead><tr>',
        '<th scope="col" class="sticky-col">Dataset</th>'
    ]
    out.extend(f'<th scope="col" class="text-right">{html.escape(m)}</th>' for m in models)
    out.append('</tr></thead><tbody>')
    for row in panel['rows']:
        best = max(s['mean'] for s in row['scores'].values())
        out.append(f'<tr><th scope="row" class="sticky-col font-medium">{html.escape(row["dataset"])}</th>')
        for model in models:
            score = row['scores'][model]
            cls = 'font-semibold text-primary score-winner' if score['mean'] == best else ''
            value = f"{score['mean']:.1f} <span class='sd-sub'>±{score['sd']:.1f}</span>"
            out.append(f'<td class="text-right font-mono {cls}">{value}</td>')
        out.append('</tr>')
    return ''.join(out) + '</tbody>'

def chart_rows(rows):
    result = []
    for row in rows:
        bars = []
        for key, value, label in [
            ('zero', row['scores']['Jev zero-shot']['mean'], 'Jev zero-shot prompts'),
            ('few', row['scores']['Jev few-shot']['mean'], 'Jev few-shot prompts'),
            ('classic', row['best'], 'Best classical pipeline: ' + ', '.join(row['bestModels']))
        ]:
            bars.append(f'<div class="bar-line" style="--value:{value}%" aria-label="{html.escape(label)}: {value:.1f}%"><span class="bar {key}"></span><span class="value font-mono">{value:.1f}%</span></div>')
        result.append(f'''<div class="chart-row" data-kind="{row['kind']}">
  <div class="chart-label">
    <strong class="chart-ds-title">{html.escape(row["dataset"])}</strong>
    <span class="chart-ds-meta">{row["kind"].title()} · {row["testRows"]:,} test rows</span>
  </div>
  <div class="bars">{"".join(bars)}</div>
</div>''')
    return ''.join(result) + '<div class="axis font-mono" aria-hidden="true"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>'

def build():
    data = load_data()
    (DOCS / 'assets/data.js').write_text('window.BENCHMARK = ' + json.dumps(data, ensure_ascii=False, indent=2) + ';\n', encoding='utf-8')
    template = (DOCS / 'report.template.html').read_text(encoding='utf-8-sig')
    
    page = template.replace('__SUMMARY_TABLE__', summary_table(data['raw']))
    page = page.replace('__FULL_TABLE__', full_table(data['raw']))
    page = page.replace('__CHART__', chart_rows(data['raw']['rows']))
    
    for asset in ('assets/report.css', 'assets/report.js', 'assets/data.js'):
        if (DOCS / asset).exists():
            version = hashlib.sha256((DOCS / asset).read_bytes()).hexdigest()[:12]
            page = page.replace(f'"{asset}"', f'"{asset}?v={version}"')
            
    assert '__SUMMARY_TABLE__' not in page and '__FULL_TABLE__' not in page and '__CHART__' not in page, "Unreplaced template tags!"
    (DOCS / 'index.html').write_text(page, encoding='utf-8')
    
    for panel in data:
        name = f'{panel}_balanced_accuracy.csv'
        shutil.copyfile(ROOT / 'published_results' / name, DOCS / 'data' / name)
    print('Successfully built docs/index.html and updated data assets from both published result panels.')

if __name__ == '__main__':
    build()
