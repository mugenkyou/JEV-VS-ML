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

def annotate_dataset(delta, is_max_pos, is_max_neg, is_parity):
    """Generate scientific editorial badge based on data."""
    if is_max_pos:
        return 'Largest Jev Lead'
    if is_max_neg:
        return 'Largest ML Lead'
    if is_parity:
        return 'Near Parity'
    return ''

def build_dumbbell_chart_svg(rows, sort_by='delta'):
    """Generate full-width editorial performance range comparison SVG."""
    # Sort rows by delta descending by default (largest Jev lead -> near parity -> largest classical lead)
    sorted_rows = sorted(rows, key=lambda r: r['delta'], reverse=True)
    if not sorted_rows:
        return '<p class="text-muted text-center" style="padding: 2rem;">No dataset records match current filter.</p>'

    h_row = 56
    padding_top = 28
    padding_bottom = 36
    total_height = padding_top + len(sorted_rows) * h_row + padding_bottom

    view_width = 960
    x_label = 200
    x_chart_start = 220
    x_chart_end = 760
    x_chart_width = x_chart_end - x_chart_start
    x_delta = 820

    lines = [
        f'<svg class="range-hero-svg" viewBox="0 0 {view_width} {total_height}" width="100%" height="{total_height}" aria-label="Jev vs Classical Balanced Accuracy Range Comparison Plot">'
    ]

    # Clean Axis Grid Ticks (40%, 50%, 60%, 70%, 80%, 90%, 100%)
    # Range of balanced accuracy on test sets is ~40% to 100%
    min_scale = 40.0
    max_scale = 100.0
    scale_range = max_scale - min_scale

    ticks = [40, 50, 60, 70, 80, 90, 100]
    for tick in ticks:
        tx = x_chart_start + ((tick - min_scale) / scale_range) * x_chart_width
        lines.append(f'<line x1="{tx:.1f}" y1="{padding_top - 10}" x2="{tx:.1f}" y2="{total_height - padding_bottom}" stroke="var(--border)" stroke-width="1" stroke-dasharray="2 3" opacity="0.6"/>')
        lines.append(f'<text x="{tx:.1f}" y="{total_height - 14}" text-anchor="middle" font-size="10" font-family="ui-monospace, monospace" fill="var(--muted-foreground)">{tick}%</text>')

    # Axis Baseline Header
    lines.append(f'<text x="{x_chart_start}" y="16" font-size="10" font-family="ui-monospace, monospace" fill="var(--muted-foreground)" font-weight="600">BALANCED ACCURACY SCALE</text>')
    lines.append(f'<text x="{x_delta}" y="16" font-size="10" font-family="ui-monospace, monospace" fill="var(--muted-foreground)" font-weight="600">SIGNED Δ (pp)</text>')

    # Editorial Annotations (Max Jev Lead, Max ML Lead, Near Parity)
    max_delta = max(r['delta'] for r in sorted_rows)
    min_delta = min(r['delta'] for r in sorted_rows)

    for i, row in enumerate(sorted_rows):
        y = padding_top + i * h_row + 24
        jev_score = row['scores']['Jev zero-shot']['mean']
        classic_score = row['best']
        delta_val = row['delta']

        # Position mapping
        x_jev = x_chart_start + (max(0.0, min(100.0, jev_score) - min_scale) / scale_range) * x_chart_width
        x_classic = x_chart_start + (max(0.0, min(100.0, classic_score) - min_scale) / scale_range) * x_chart_width

        x_min = min(x_jev, x_classic)
        x_max = max(x_jev, x_classic)

        # Determine annotation
        is_max_pos = (delta_val == max_delta and delta_val > 2.0)
        is_max_neg = (delta_val == min_delta and delta_val < -5.0)
        is_parity = (abs(delta_val) <= 1.5)
        annotation = annotate_dataset(delta_val, is_max_pos, is_max_neg, is_parity)

        # Delta formatting
        delta_str = f'+{delta_val:.1f} pp' if delta_val > 0 else f'{delta_val:.1f} pp'
        delta_color = 'var(--delta-pos-fg)' if delta_val > 0 else ('var(--delta-neg-fg)' if delta_val < -2.0 else 'var(--muted-foreground)')
        delta_bg = 'var(--delta-pos-bg)' if delta_val > 0 else ('var(--delta-neg-bg)' if delta_val < -2.0 else 'var(--secondary)')

        lines.append(f'<g class="range-row" data-dataset="{html.escape(row["dataset"])}" tabindex="0" role="button" onclick="window.openDrawer(\'{html.escape(row["dataset"])}\')">')

        # Hover background strip
        lines.append(f'<rect x="0" y="{y - 20}" width="{view_width}" height="{h_row}" fill="transparent" class="row-hover-bg" rx="4"/>')

        # Dataset label & metadata
        lines.append(f'<text x="{x_label}" y="{y - 2}" text-anchor="end" font-size="12.5" font-weight="600" fill="var(--foreground)">{html.escape(row["dataset"])}</text>')
        lines.append(f'<text x="{x_label}" y="{y + 12}" text-anchor="end" font-size="9" font-family="ui-monospace, monospace" fill="var(--muted-foreground)">{html.escape(row["domain"].upper())} · {row["classes"]} CL</text>')

        # Base subtle track
        lines.append(f'<line x1="{x_chart_start}" y1="{y}" x2="{x_chart_end}" y2="{y}" stroke="var(--border)" stroke-width="1.5" opacity="0.4"/>')

        # Connector range line
        track_stroke = 'var(--bar-jev)' if delta_val >= 0 else 'var(--bar-classic)'
        lines.append(f'<line x1="{x_min}" y1="{y}" x2="{x_max}" y2="{y}" stroke="{track_stroke}" stroke-width="3.5" stroke-linecap="round"/>')

        # Classical dot & label
        lines.append(f'<circle cx="{x_classic}" cy="{y}" r="6" fill="var(--bar-classic)" stroke="var(--card)" stroke-width="2"/>')

        # Jev dot & label
        lines.append(f'<circle cx="{x_jev}" cy="{y}" r="7" fill="var(--bar-jev)" stroke="var(--card)" stroke-width="2"/>')

        # Direct numeric labels attached to dots
        if abs(x_jev - x_classic) >= 36:
            lines.append(f'<text x="{x_classic}" y="{y - 10}" text-anchor="middle" font-size="9.5" font-family="ui-monospace, monospace" fill="var(--muted-foreground)">{classic_score:.1f}%</text>')
            lines.append(f'<text x="{x_jev}" y="{y - 10}" text-anchor="middle" font-size="9.5" font-family="ui-monospace, monospace" font-weight="700" fill="var(--bar-jev)">{jev_score:.1f}%</text>')
        else:
            # Overlapping or near parity points
            lead_x = max(x_jev, x_classic)
            lines.append(f'<text x="{lead_x + 14}" y="{y - 8}" text-anchor="start" font-size="9" font-family="ui-monospace, monospace" fill="var(--bar-jev)" font-weight="600">Jev {jev_score:.1f}%</text>')
            lines.append(f'<text x="{lead_x + 14}" y="{y + 8}" text-anchor="start" font-size="9" font-family="ui-monospace, monospace" fill="var(--muted-foreground)">ML {classic_score:.1f}%</text>')

        # Prominent Delta Badge
        lines.append(f'<rect x="{x_delta}" y="{y - 11}" width="68" height="22" rx="4" fill="{delta_bg}" stroke="{delta_color}" stroke-opacity="0.3"/>')
        lines.append(f'<text x="{x_delta + 34}" y="{y + 4}" text-anchor="middle" font-size="10.5" font-family="ui-monospace, monospace" font-weight="700" fill="{delta_color}">{delta_str}</text>')

        # Editorial Annotation tag
        if annotation:
            ann_cls = 'tag-pos' if delta_val > 0 else ('tag-neg' if delta_val < -2.0 else 'tag-parity')
            lines.append(f'<text x="{x_delta + 78}" y="{y + 4}" text-anchor="start" font-size="9" font-family="ui-monospace, monospace" fill="var(--muted-foreground)" class="editorial-tag">{annotation}</text>')

        lines.append('</g>')

    lines.append('</svg>')
    return ''.join(lines)

def build_diverging_chart_svg(rows):
    """Generate zero-centered signed difference dot / lollipop plot SVG."""
    # Sort descending by signed delta
    sorted_rows = sorted(rows, key=lambda r: r['delta'], reverse=True)
    if not sorted_rows:
        return '<p class="text-muted text-center" style="padding: 2rem;">No dataset records match current filter.</p>'

    h_row = 42
    padding_top = 30
    padding_bottom = 34
    total_height = padding_top + len(sorted_rows) * h_row + padding_bottom

    view_width = 960
    x_label = 200
    chart_x = 220
    chart_w = 680

    # Scale from -40 pp to +15 pp
    min_val = -40.0
    max_val = 15.0
    range_val = max_val - min_val
    zero_x = chart_x + ((0.0 - min_val) / range_val) * chart_w

    lines = [
        f'<svg class="diff-dot-svg" viewBox="0 0 {view_width} {total_height}" width="100%" height="{total_height}" aria-label="Zero-Centered Signed Difference Plot">'
    ]

    # Side Region Shading
    lines.append(f'<rect x="{chart_x}" y="{padding_top - 12}" width="{zero_x - chart_x}" height="{total_height - padding_top - padding_bottom + 12}" fill="var(--secondary)" opacity="0.4"/>')

    # Editorial Region Headers
    lines.append(f'<text x="{chart_x + 10}" y="16" font-size="9.5" font-family="ui-monospace, monospace" fill="var(--muted-foreground)" font-weight="600">← CLASSICAL ADVANTAGE</text>')
    lines.append(f'<text x="{chart_x + chart_w - 10}" y="16" text-anchor="end" font-size="9.5" font-family="ui-monospace, monospace" fill="var(--accent)" font-weight="600">JEV ADVANTAGE →</text>')

    # Grid ticks (-40, -30, -20, -10, 0, +10)
    for tick in (-40, -30, -20, -10, 0, 10):
        tx = chart_x + ((tick - min_val) / range_val) * chart_w
        is_zero = (tick == 0)
        line_color = 'var(--foreground)' if is_zero else 'var(--border)'
        line_width = '1.8' if is_zero else '1'
        dash = '' if is_zero else 'stroke-dasharray="2 3"'
        lines.append(f'<line x1="{tx:.1f}" y1="{padding_top - 8}" x2="{tx:.1f}" y2="{total_height - padding_bottom}" stroke="{line_color}" stroke-width="{line_width}" {dash} opacity="0.8"/>')
        tick_str = f'+{tick} pp' if tick > 0 else (f'{tick} pp' if tick < 0 else '0.0 (Parity)')
        tick_weight = '700' if is_zero else '500'
        tick_color = 'var(--foreground)' if is_zero else 'var(--muted-foreground)'
        lines.append(f'<text x="{tx:.1f}" y="{total_height - 12}" text-anchor="middle" font-size="9" font-family="ui-monospace, monospace" font-weight="{tick_weight}" fill="{tick_color}">{tick_str}</text>')

    for i, row in enumerate(sorted_rows):
        y = padding_top + i * h_row + 18
        d_val = row['delta']
        dx = chart_x + ((d_val - min_val) / range_val) * chart_w

        is_pos = d_val > 0
        fill_color = 'var(--bar-jev)' if is_pos else 'var(--bar-classic)'

        lines.append(f'<g class="diff-row" data-dataset="{html.escape(row["dataset"])}" tabindex="0" role="button" onclick="window.openDrawer(\'{html.escape(row["dataset"])}\')">')
        lines.append(f'<rect x="0" y="{y - 16}" width="{view_width}" height="{h_row}" fill="transparent" class="row-hover-bg" rx="4"/>')

        # Dataset name & modality
        lines.append(f'<text x="{x_label}" y="{y - 2}" text-anchor="end" font-size="12" font-weight="600" fill="var(--foreground)">{html.escape(row["dataset"])}</text>')
        lines.append(f'<text x="{x_label}" y="{y + 11}" text-anchor="end" font-size="8.5" font-family="ui-monospace, monospace" fill="var(--muted-foreground)">{html.escape(row["domain"].upper())}</text>')

        # Connecting stem from 0 to dot
        lines.append(f'<line x1="{zero_x:.1f}" y1="{y}" x2="{dx:.1f}" y2="{y}" stroke="{fill_color}" stroke-width="2.5" stroke-linecap="round"/>')

        # Signed Dot at value point
        lines.append(f'<circle cx="{dx:.1f}" cy="{y}" r="6.5" fill="{fill_color}" stroke="var(--card)" stroke-width="2"/>')

        # Direct signed text label
        d_str = f'+{d_val:.1f} pp' if d_val > 0 else f'{d_val:.1f} pp'
        if d_val >= 0:
            lines.append(f'<text x="{dx + 12:.1f}" y="{y + 4}" text-anchor="start" font-size="9.5" font-family="ui-monospace, monospace" font-weight="700" fill="{fill_color}">{d_str}</text>')
        else:
            lines.append(f'<text x="{dx - 12:.1f}" y="{y + 4}" text-anchor="end" font-size="9.5" font-family="ui-monospace, monospace" font-weight="700" fill="{fill_color}">{d_str}</text>')

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
