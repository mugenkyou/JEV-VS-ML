"""Build the dependency-free research benchmark website from published score tables."""
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
        'description': 'Topic classification across 4 balanced news categories: World, Sports, Business, and Sci/Tech.',
        'classesDetail': '4 classes: World, Sports, Business, Sci/Tech (250 examples each in test holdout)',
        'metrics': 'Balanced Accuracy (unweighted average recall across classes)',
    },
    'Banking77': {
        'kind': 'text',
        'domain': 'Text',
        'classes': 77,
        'testRows': 1500,
        'description': 'Fine-grained customer service intent classification across 77 online banking intent categories.',
        'classesDetail': '77 fine-grained customer intent classes (majority baseline = 1.3%)',
        'metrics': 'Balanced Accuracy (majority baseline = 1.3%)',
    },
    'SMS Spam': {
        'kind': 'text',
        'domain': 'Text',
        'classes': 2,
        'testRows': 1000,
        'description': 'Mobile phone SMS spam detection on imbalanced English mobile communications.',
        'classesDetail': '2 classes: Ham (legitimate) vs. Spam (imbalanced)',
        'metrics': 'Balanced Accuracy (majority baseline = 50.0%)',
    },
    'IMDb': {
        'kind': 'text',
        'domain': 'Text',
        'classes': 2,
        'testRows': 1000,
        'description': 'Binary movie review sentiment polarity classification from long-form user reviews.',
        'classesDetail': '2 classes: Positive vs. Negative review sentiment',
        'metrics': 'Balanced Accuracy (majority baseline = 50.0%)',
    },
    'Bank Marketing': {
        'kind': 'tabular',
        'domain': 'Tabular',
        'classes': 2,
        'testRows': 1000,
        'description': 'Direct marketing phone campaign prediction of client term deposit subscriptions.',
        'classesDetail': '2 classes: Subscribed (yes) vs. Not subscribed (no)',
        'metrics': 'Balanced Accuracy with demographic, economic, and campaign contact features',
    },
    'Online Shoppers': {
        'kind': 'tabular',
        'domain': 'Tabular',
        'classes': 2,
        'testRows': 1000,
        'description': 'E-commerce session revenue purchasing intention from real-time web browsing analytics.',
        'classesDetail': '2 classes: Purchase revenue generated (True) vs. No purchase (False)',
        'metrics': 'Balanced Accuracy across numerical session attributes and administrative features',
    },
    'Breast Cancer': {
        'kind': 'tabular',
        'domain': 'Tabular',
        'classes': 2,
        'testRows': 114,
        'description': 'Wisconsin diagnostic clinical features for malignant versus benign tumor classification.',
        'classesDetail': '2 classes: Malignant vs. Benign (compact holdout n=114)',
        'metrics': 'Balanced Accuracy across 30 continuous nuclear feature dimensions',
    },
    'Iris': {
        'kind': 'tabular',
        'domain': 'Tabular',
        'classes': 3,
        'testRows': 30,
        'description': 'Morphometric iris flower species classification from sepal and petal measurements.',
        'classesDetail': '3 classes: Setosa, Versicolor, Virginica (compact holdout n=30)',
        'metrics': 'Balanced Accuracy across 4 morphometric dimensions',
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

def build_dumbbell_chart_svg(rows):
    """Generate static SVG dumbbell comparison chart."""
    h_row = 44
    padding_top = 20
    padding_bottom = 30
    total_height = padding_top + len(rows) * h_row + padding_bottom
    
    lines = [f'<svg class="dumbbell-svg" viewBox="0 0 800 {total_height}" width="100%" height="{total_height}" aria-label="Jev vs Classical Balanced Accuracy Dumbbell Plot">']
    
    # Grid lines at 0%, 25%, 50%, 75%, 100%
    x_offset = 150
    width_chart = 600
    for tick in (0, 25, 50, 75, 100):
        gx = x_offset + (tick / 100.0) * width_chart
        lines.append(f'<line x1="{gx}" y1="{padding_top}" x2="{gx}" y2="{total_height - padding_bottom}" stroke="var(--border)" stroke-width="1" stroke-dasharray="3 3"/>')
        lines.append(f'<text x="{gx}" y="{total_height - 10}" text-anchor="middle" font-size="10" font-family="ui-monospace, monospace" fill="var(--muted-foreground)">{tick}%</text>')

    for i, row in enumerate(rows):
        y = padding_top + i * h_row + 22
        jev_score = row['scores']['Jev zero-shot']['mean']
        classic_score = row['best']
        
        x_jev = x_offset + (jev_score / 100.0) * width_chart
        x_classic = x_offset + (classic_score / 100.0) * width_chart
        
        x_min = min(x_jev, x_classic)
        x_max = max(x_jev, x_classic)
        
        # Track line
        track_color = 'var(--accent)' if jev_score >= classic_score else 'var(--muted-foreground)'
        lines.append(f'<g class="dumbbell-row" data-dataset="{html.escape(row["dataset"])}" tabindex="0" role="button">')
        lines.append(f'<text x="140" y="{y + 4}" text-anchor="end" font-size="12" font-weight="600" fill="var(--foreground)">{html.escape(row["dataset"])}</text>')
        lines.append(f'<line x1="{x_min}" y1="{y}" x2="{x_max}" y2="{y}" stroke="{track_color}" stroke-width="2" stroke-opacity="0.6"/>')
        
        # Classical dot
        lines.append(f'<circle cx="{x_classic}" cy="{y}" r="5.5" fill="var(--bar-classic)" stroke="var(--card)" stroke-width="2"/>')
        
        # Jev dot
        lines.append(f'<circle cx="{x_jev}" cy="{y}" r="6.5" fill="var(--bar-jev)" stroke="var(--card)" stroke-width="2"/>')
        
        # Score labels
        if abs(x_jev - x_classic) > 40:
            lines.append(f'<text x="{x_classic}" y="{y - 9}" text-anchor="middle" font-size="9" font-family="ui-monospace, monospace" fill="var(--muted-foreground)">{classic_score:.1f}%</text>')
            lines.append(f'<text x="{x_jev}" y="{y - 9}" text-anchor="middle" font-size="9" font-family="ui-monospace, monospace" font-weight="700" fill="var(--bar-jev)">{jev_score:.1f}%</text>')
        else:
            lines.append(f'<text x="{max(x_jev, x_classic) + 12}" y="{y + 3}" text-anchor="start" font-size="9" font-family="ui-monospace, monospace" fill="var(--muted-foreground)">Jev {jev_score:.1f}% / ML {classic_score:.1f}%</text>')
            
        lines.append('</g>')
        
    lines.append('</svg>')
    return ''.join(lines)

def build_diverging_chart_svg(rows):
    """Generate diverging bar chart showing difference (Δ) vs best classical."""
    h_row = 38
    padding_top = 20
    padding_bottom = 25
    total_height = padding_top + len(rows) * h_row + padding_bottom
    
    lines = [f'<svg class="diverging-svg" viewBox="0 0 800 {total_height}" width="100%" height="{total_height}" aria-label="Diverging Delta Chart">']
    
    # Scale from -40% to +15%
    min_val = -40.0
    max_val = 15.0
    range_val = max_val - min_val
    chart_x = 160
    chart_w = 580
    zero_x = chart_x + ((0.0 - min_val) / range_val) * chart_w
    
    # Grid ticks at -40, -30, -20, -10, 0, +10
    for tick in (-40, -30, -20, -10, 0, 10):
        tx = chart_x + ((tick - min_val) / range_val) * chart_w
        is_zero = (tick == 0)
        line_color = 'var(--foreground)' if is_zero else 'var(--border)'
        line_width = '1.5' if is_zero else '1'
        dash = '' if is_zero else 'stroke-dasharray="2 2"'
        lines.append(f'<line x1="{tx}" y1="{padding_top}" x2="{tx}" y2="{total_height - padding_bottom}" stroke="{line_color}" stroke-width="{line_width}" {dash}/>')
        tick_str = f'+{tick}pp' if tick > 0 else (f'{tick}pp' if tick < 0 else '0.0')
        lines.append(f'<text x="{tx}" y="{total_height - 8}" text-anchor="middle" font-size="9" font-family="ui-monospace, monospace" fill="var(--muted-foreground)">{tick_str}</text>')
        
    for i, row in enumerate(rows):
        y = padding_top + i * h_row + 12
        d_val = row['delta']
        bx = chart_x + ((min(0, d_val) - min_val) / range_val) * chart_w
        bw = (abs(d_val) / range_val) * chart_w
        fill_color = 'var(--delta-pos-fg)' if d_val > 0 else 'var(--bar-classic)'
        
        lines.append(f'<g class="diverging-row" data-dataset="{html.escape(row["dataset"])}">')
        lines.append(f'<text x="150" y="{y + 11}" text-anchor="end" font-size="11" font-weight="500" fill="var(--foreground)">{html.escape(row["dataset"])}</text>')
        lines.append(f'<rect x="{bx}" y="{y}" width="{max(2, bw)}" height="14" fill="{fill_color}" rx="2" opacity="0.85"/>')
        
        # Label position
        text_x = zero_x + (bw + 6 if d_val >= 0 else -(bw + 6))
        anchor = 'start' if d_val >= 0 else 'end'
        d_str = f'+{d_val:.1f} pp' if d_val > 0 else f'{d_val:.1f} pp'
        lines.append(f'<text x="{text_x}" y="{y + 11}" text-anchor="{anchor}" font-size="9" font-family="ui-monospace, monospace" font-weight="600" fill="{fill_color}">{d_str}</text>')
        lines.append('</g>')
        
    lines.append('</svg>')
    return ''.join(lines)

def build_dataset_grid(rows):
    """Generate Swiss editorial small-multiples dataset grid."""
    cards = []
    for row in rows:
        d_val = row['delta']
        d_str = f'+{d_val:.1f}%' if d_val > 0 else f'{d_val:.1f}%'
        d_cls = 'delta-pos' if d_val > 0 else ('delta-neg' if d_val < 0 else 'delta-neutral')
        jev_val = row['scores']['Jev zero-shot']['mean']
        best_val = row['best']
        best_model = ', '.join(row['bestModels'])
        
        cards.append(f'''<div class="sm-card" data-dataset="{html.escape(row['dataset'])}" tabindex="0" role="button">
  <div class="sm-card-head">
    <div>
      <span class="sm-card-domain font-mono">{row['domain']} · {row['classes']} cl</span>
      <h4 class="sm-card-title">{html.escape(row['dataset'])}</h4>
    </div>
    <span class="delta-chip {d_cls} font-mono">{d_str}</span>
  </div>
  
  <div class="sm-spark">
    <div class="sm-spark-track">
      <div class="sm-spark-bar" style="left: {min(jev_val, best_val):.1f}%; width: {abs(jev_val - best_val):.1f}%;"></div>
      <div class="sm-spark-dot dot-classic" style="left: {best_val:.1f}%;" title="Best Classical: {best_val:.1f}%"></div>
      <div class="sm-spark-dot dot-jev" style="left: {jev_val:.1f}%;" title="Jev Zero-Shot: {jev_val:.1f}%"></div>
    </div>
    <div class="sm-spark-axis font-mono">
      <span>0%</span>
      <span>50%</span>
      <span>100%</span>
    </div>
  </div>

  <div class="sm-card-stats font-mono">
    <div class="sm-stat">
      <span class="sm-stat-label">Jev Zero</span>
      <span class="sm-stat-val text-accent">{jev_val:.1f}%</span>
    </div>
    <div class="sm-stat">
      <span class="sm-stat-label">Best ML ({html.escape(best_model[:12])})</span>
      <span class="sm-stat-val text-muted">{best_val:.1f}%</span>
    </div>
  </div>
</div>''')
    return ''.join(cards)

def summary_table(panel):
    out = [
        '<caption class="sr-only">Main benchmark comparison: Jev vs best classical pipeline by dataset</caption>',
        '<thead><tr>',
        '<th scope="col" class="th-dataset sortable" data-sort="dataset">Dataset ↕</th>',
        '<th scope="col" class="th-domain sortable" data-sort="domain">Domain ↕</th>',
        '<th scope="col" class="th-classes text-right sortable" data-sort="classes">Classes ↕</th>',
        '<th scope="col" class="th-test text-right sortable" data-sort="testRows">Test N ↕</th>',
        '<th scope="col" class="th-score text-right sortable" data-sort="jevZero">Jev Zero-shot ↕</th>',
        '<th scope="col" class="th-score text-right sortable" data-sort="jevFew">Jev Few-shot ↕</th>',
        '<th scope="col" class="th-best text-right sortable" data-sort="best">Best Classical ↕</th>',
        '<th scope="col" class="th-delta text-right sortable" data-sort="delta">Δ vs Best ↕</th>',
        '<th scope="col" class="th-action text-center"><span class="sr-only">Inspect</span></th>',
        '</tr></thead>',
        '<tbody>'
    ]
    for row in panel['rows']:
        d_val = row['delta']
        delta_cls = 'delta-pos' if d_val > 0 else ('delta-neg' if d_val < 0 else 'delta-neutral')
        delta_str = f'+{d_val:.1f}%' if d_val > 0 else f'{d_val:.1f}%'
        best_model_name = ', '.join(row['bestModels'])
        
        out.append(f'''<tr class="summary-row" data-dataset="{html.escape(row['dataset'])}" tabindex="0" role="button" aria-expanded="false" title="Click to inspect model scores for {html.escape(row['dataset'])}">
  <td class="font-medium text-foreground"><span class="ds-name">{html.escape(row['dataset'])}</span></td>
  <td><span class="badge-domain badge-{row['kind']}">{row['domain']}</span></td>
  <td class="text-right font-mono">{row['classes']}</td>
  <td class="text-right font-mono">{row['testRows']:,}</td>
  <td class="text-right font-mono font-semibold {'score-winner' if row['scores']['Jev zero-shot']['mean'] >= row['best'] else ''}">{row['scores']['Jev zero-shot']['mean']:.1f}% <span class="sd-sub">±{row['scores']['Jev zero-shot']['sd']:.1f}</span></td>
  <td class="text-right font-mono">{row['scores']['Jev few-shot']['mean']:.1f}% <span class="sd-sub">±{row['scores']['Jev few-shot']['sd']:.1f}</span></td>
  <td class="text-right font-mono font-semibold {'score-winner' if row['best'] > row['scores']['Jev zero-shot']['mean'] else ''}">{row['best']:.1f}% <span class="best-model-label">({html.escape(best_model_name)})</span></td>
  <td class="text-right font-mono"><span class="delta-chip {delta_cls}">{delta_str}</span></td>
  <td class="text-center"><span class="inspect-tag font-mono">inspect →</span></td>
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

def build():
    data = load_data()
    (DOCS / 'assets/data.js').write_text('window.BENCHMARK = ' + json.dumps(data, ensure_ascii=False, indent=2) + ';\n', encoding='utf-8')
    template = (DOCS / 'report.template.html').read_text(encoding='utf-8-sig')
    
    page = template.replace('__SUMMARY_TABLE__', summary_table(data['raw']))
    page = page.replace('__FULL_TABLE__', full_table(data['raw']))
    page = page.replace('__DUMBBELL_SVG__', build_dumbbell_chart_svg(data['raw']['rows']))
    page = page.replace('__DIVERGING_SVG__', build_diverging_chart_svg(data['raw']['rows']))
    page = page.replace('__DATASET_GRID__', build_dataset_grid(data['raw']['rows']))
    
    for asset in ('assets/report.css', 'assets/report.js', 'assets/data.js'):
        if (DOCS / asset).exists():
            version = hashlib.sha256((DOCS / asset).read_bytes()).hexdigest()[:12]
            page = page.replace(f'"{asset}"', f'"{asset}?v={version}"')
            
    assert '__SUMMARY_TABLE__' not in page and '__DUMBBELL_SVG__' not in page and '__DIVERGING_SVG__' not in page, "Unreplaced template tags!"
    (DOCS / 'index.html').write_text(page, encoding='utf-8')
    
    for panel in data:
        name = f'{panel}_balanced_accuracy.csv'
        shutil.copyfile(ROOT / 'published_results' / name, DOCS / 'data' / name)
    print('Successfully built docs/index.html with interactive SVG charts and data assets.')

if __name__ == '__main__':
    build()
