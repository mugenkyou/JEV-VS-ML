(() => {
  'use strict';

  // --- Theme Management ---
  const initTheme = () => {
    const saved = localStorage.getItem('theme');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const theme = saved || (prefersDark ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', theme);
    updateThemeToggleUI(theme);
  };

  const updateThemeToggleUI = (theme) => {
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;
    btn.innerHTML = theme === 'dark' 
      ? '<span aria-hidden="true">☀️</span><span class="sr-only">Switch to light mode</span>' 
      : '<span aria-hidden="true">🌙</span><span class="sr-only">Switch to dark mode</span>';
  };

  const toggleTheme = () => {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    updateThemeToggleUI(next);
  };

  // --- Benchmark Data ---
  const data = window.BENCHMARK;
  if (!data) return;

  const escape = value => String(value).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));

  // State
  let currentSort = { column: 'dataset', direction: 'asc' };

  const getActivePanel = () => {
    const checked = document.querySelector('input[name="panel"]:checked');
    return checked ? checked.value : 'raw';
  };

  const getActiveDomain = () => {
    const checked = document.querySelector('input[name="domain-filter"]:checked');
    return checked ? checked.value : 'all';
  };

  // --- Drawer Interaction (WOW #3) ---
  const openDrawer = (datasetName) => {
    const panel = getActivePanel();
    const panelData = data[panel];
    if (!panelData) return;

    const row = panelData.rows.find(r => r.dataset === datasetName);
    if (!row) return;

    const drawer = document.getElementById('dataset-drawer');
    const backdrop = document.getElementById('drawer-backdrop');
    if (!drawer || !backdrop) return;

    // Populate drawer elements
    document.getElementById('drawer-title').textContent = row.dataset;
    document.getElementById('drawer-meta').textContent = `${row.domain} Classification · ${row.classes} Classes · ${row.testRows.toLocaleString('en-US')} Test Rows`;
    document.getElementById('drawer-desc').textContent = row.description;

    const jevZero = row.scores['Jev zero-shot'].mean;
    const jevFew = row.scores['Jev few-shot'].mean;
    const classicBest = row.best;
    const bestModelName = row.bestModels.join(', ');
    const dVal = row.delta;

    document.getElementById('drawer-val-jev-zero').textContent = `${jevZero.toFixed(1)}%`;
    document.getElementById('drawer-val-jev-few').textContent = `${jevFew.toFixed(1)}%`;
    document.getElementById('drawer-val-classic').textContent = `${classicBest.toFixed(1)}%`;
    document.getElementById('drawer-lbl-classic').textContent = `Best ML (${bestModelName})`;
    
    const deltaEl = document.getElementById('drawer-val-delta');
    deltaEl.textContent = dVal > 0 ? `+${dVal.toFixed(1)} pp` : `${dVal.toFixed(1)} pp`;
    deltaEl.className = `drawer-m-val font-mono ${dVal > 0 ? 'text-accent' : 'text-muted'}`;

    // Render dot plot dispersion for top models
    const varBox = document.getElementById('drawer-variance-content');
    if (varBox) {
      const displayModels = ['Jev zero-shot', 'Jev few-shot', ...row.bestModels.slice(0, 2)];
      varBox.innerHTML = displayModels.map(m => {
        const s = row.scores[m];
        const minX = Math.max(0, s.mean - s.sd);
        const maxX = Math.min(100, s.mean + s.sd);
        const isJev = m.startsWith('Jev');
        return `
          <div style="margin-bottom: 0.75rem;">
            <div style="display: flex; justify-content: space-between; font-size: 0.6875rem; margin-bottom: 0.25rem;">
              <span class="font-medium">${escape(m)}</span>
              <span class="font-mono text-muted">${s.mean.toFixed(1)} ± ${s.sd.toFixed(1)}%</span>
            </div>
            <div style="position: relative; height: 8px; background: var(--secondary); border-radius: 4px; border: 1px solid var(--border);">
              <div style="position: absolute; left: ${minX}%; width: ${Math.max(2, maxX - minX)}%; height: 100%; background: ${isJev ? 'var(--bar-jev)' : 'var(--bar-classic)'}; opacity: 0.3; border-radius: 2px;"></div>
              <div style="position: absolute; left: ${s.mean}%; top: -2px; width: 10px; height: 10px; border-radius: 50%; background: ${isJev ? 'var(--bar-jev)' : 'var(--bar-classic)'}; transform: translateX(-50%); border: 2px solid var(--card);"></div>
            </div>
          </div>
        `;
      }).join('');
    }

    // Render 14 models grid
    const gridEl = document.getElementById('drawer-models-list');
    if (gridEl) {
      gridEl.innerHTML = panelData.models.map(m => {
        const s = row.scores[m];
        const maxScore = Math.max(...Object.values(row.scores).map(sc => sc.mean));
        const isBest = s.mean === maxScore;
        return `
          <div class="drawer-model-card ${isBest ? 'best' : ''}">
            <span class="font-mono" style="font-size: 0.6875rem; color: var(--muted-foreground);">${escape(m)}</span>
            <span class="font-mono" style="font-size: 0.75rem; font-weight: ${isBest ? '700' : '500'};">${s.mean.toFixed(1)}%</span>
          </div>
        `;
      }).join('');
    }

    drawer.classList.add('open');
    backdrop.classList.add('open');
    document.body.style.overflow = 'hidden';
  };

  const closeDrawer = () => {
    const drawer = document.getElementById('dataset-drawer');
    const backdrop = document.getElementById('drawer-backdrop');
    if (drawer) drawer.classList.remove('open');
    if (backdrop) backdrop.classList.remove('open');
    document.body.style.overflow = '';
  };

  // --- SVG Chart Builders ---
  const renderDumbbellChart = (rows) => {
    const container = document.getElementById('dumbbell-chart-wrap');
    if (!container) return;

    const hRow = 44;
    const paddingTop = 20;
    const paddingBottom = 30;
    const totalHeight = paddingTop + rows.length * hRow + paddingBottom;
    const xOffset = 150;
    const widthChart = 600;

    let lines = [`<svg class="dumbbell-svg" viewBox="0 0 800 ${totalHeight}" width="100%" height="${totalHeight}">`];

    for (const tick of [0, 25, 50, 75, 100]) {
      const gx = xOffset + (tick / 100.0) * widthChart;
      lines.push(`<line x1="${gx}" y1="${paddingTop}" x2="${gx}" y2="${totalHeight - paddingBottom}" stroke="var(--border)" stroke-width="1" stroke-dasharray="3 3"/>`);
      lines.push(`<text x="${gx}" y="${totalHeight - 10}" text-anchor="middle" font-size="10" font-family="ui-monospace, monospace" fill="var(--muted-foreground)">${tick}%</text>`);
    }

    rows.forEach((row, i) => {
      const y = paddingTop + i * hRow + 22;
      const jevScore = row.scores['Jev zero-shot'].mean;
      const classicScore = row.best;
      const xJev = xOffset + (jevScore / 100.0) * widthChart;
      const xClassic = xOffset + (classicScore / 100.0) * widthChart;
      const xMin = Math.min(xJev, xClassic);
      const xMax = Math.max(xJev, xClassic);
      const trackColor = jevScore >= classicScore ? 'var(--accent)' : 'var(--muted-foreground)';

      lines.push(`
        <g class="dumbbell-row" data-dataset="${escape(row.dataset)}" tabindex="0" role="button" onclick="window.openDrawer('${escape(row.dataset)}')">
          <text x="140" y="${y + 4}" text-anchor="end" font-size="12" font-weight="600" fill="var(--foreground)">${escape(row.dataset)}</text>
          <line x1="${xMin}" y1="${y}" x2="${xMax}" y2="${y}" stroke="${trackColor}" stroke-width="2" stroke-opacity="0.6"/>
          <circle cx="${xClassic}" cy="${y}" r="5.5" fill="var(--bar-classic)" stroke="var(--card)" stroke-width="2"/>
          <circle cx="${xJev}" cy="${y}" r="6.5" fill="var(--bar-jev)" stroke="var(--card)" stroke-width="2"/>
          ${Math.abs(xJev - xClassic) > 40
            ? `<text x="${xClassic}" y="${y - 9}" text-anchor="middle" font-size="9" font-family="ui-monospace, monospace" fill="var(--muted-foreground)">${classicScore.toFixed(1)}%</text>
               <text x="${xJev}" y="${y - 9}" text-anchor="middle" font-size="9" font-family="ui-monospace, monospace" font-weight="700" fill="var(--bar-jev)">${jevScore.toFixed(1)}%</text>`
            : `<text x="${Math.max(xJev, xClassic) + 12}" y="${y + 3}" text-anchor="start" font-size="9" font-family="ui-monospace, monospace" fill="var(--muted-foreground)">Jev ${jevScore.toFixed(1)}% / ML ${classicScore.toFixed(1)}%</text>`
          }
        </g>
      `);
    });

    lines.push('</svg>');
    container.innerHTML = lines.join('');
  };

  const renderDivergingChart = (rows) => {
    const container = document.getElementById('diverging-chart-wrap');
    if (!container) return;

    const hRow = 38;
    const paddingTop = 20;
    const paddingBottom = 25;
    const totalHeight = paddingTop + rows.length * hRow + paddingBottom;
    const minVal = -40.0;
    const maxVal = 15.0;
    const rangeVal = maxVal - minVal;
    const chartX = 160;
    const chartW = 580;
    const zeroX = chartX + ((0.0 - minVal) / rangeVal) * chartW;

    let lines = [`<svg class="diverging-svg" viewBox="0 0 800 ${totalHeight}" width="100%" height="${totalHeight}">`];

    for (const tick of [-40, -30, -20, -10, 0, 10]) {
      const tx = chartX + ((tick - minVal) / rangeVal) * chartW;
      const isZero = tick === 0;
      const lineColor = isZero ? 'var(--foreground)' : 'var(--border)';
      const lineWidth = isZero ? '1.5' : '1';
      const dash = isZero ? '' : 'stroke-dasharray="2 2"';
      lines.push(`<line x1="${tx}" y1="${paddingTop}" x2="${tx}" y2="${totalHeight - paddingBottom}" stroke="${lineColor}" stroke-width="${lineWidth}" ${dash}/>`);
      const tickStr = tick > 0 ? `+${tick}pp` : (tick < 0 ? `${tick}pp` : '0.0');
      lines.push(`<text x="${tx}" y="${totalHeight - 8}" text-anchor="middle" font-size="9" font-family="ui-monospace, monospace" fill="var(--muted-foreground)">${tickStr}</text>`);
    }

    rows.forEach((row, i) => {
      const y = paddingTop + i * hRow + 12;
      const dVal = row.delta;
      const bx = chartX + ((Math.min(0, dVal) - minVal) / rangeVal) * chartW;
      const bw = (Math.abs(dVal) / rangeVal) * chartW;
      const fillColor = dVal > 0 ? 'var(--delta-pos-fg)' : 'var(--bar-classic)';
      const textX = zeroX + (dVal >= 0 ? bw + 6 : -(bw + 6));
      const anchor = dVal >= 0 ? 'start' : 'end';
      const dStr = dVal > 0 ? `+${dVal.toFixed(1)} pp` : `${dVal.toFixed(1)} pp`;

      lines.push(`
        <g class="diverging-row" data-dataset="${escape(row.dataset)}" onclick="window.openDrawer('${escape(row.dataset)}')">
          <text x="150" y="${y + 11}" text-anchor="end" font-size="11" font-weight="500" fill="var(--foreground)">${escape(row.dataset)}</text>
          <rect x="${bx}" y="${y}" width="${Math.max(2, bw)}" height="14" fill="${fillColor}" rx="2" opacity="0.85"/>
          <text x="${textX}" y="${y + 11}" text-anchor="${anchor}" font-size="9" font-family="ui-monospace, monospace" font-weight="600" fill="${fillColor}">${dStr}</text>
        </g>
      `);
    });

    lines.push('</svg>');
    container.innerHTML = lines.join('');
  };

  const renderSwissGrid = (rows) => {
    const container = document.getElementById('swiss-dataset-grid');
    if (!container) return;

    container.innerHTML = rows.map(row => {
      const dVal = row.delta;
      const dStr = dVal > 0 ? `+${dVal.toFixed(1)}%` : `${dVal.toFixed(1)}%`;
      const dCls = dVal > 0 ? 'delta-pos' : (dVal < 0 ? 'delta-neg' : 'delta-neutral');
      const jevVal = row.scores['Jev zero-shot'].mean;
      const bestVal = row.best;
      const bestModel = row.bestModels.join(', ');

      return `
        <div class="sm-card" data-dataset="${escape(row.dataset)}" tabindex="0" role="button" onclick="window.openDrawer('${escape(row.dataset)}')">
          <div class="sm-card-head">
            <div>
              <span class="sm-card-domain font-mono">${row.domain} · ${row.classes} cl</span>
              <h4 class="sm-card-title">${escape(row.dataset)}</h4>
            </div>
            <span class="delta-chip ${dCls} font-mono">${dStr}</span>
          </div>
          
          <div class="sm-spark">
            <div class="sm-spark-track">
              <div class="sm-spark-bar" style="left: ${Math.min(jevVal, bestVal).toFixed(1)}%; width: ${Math.abs(jevVal - bestVal).toFixed(1)}%;"></div>
              <div class="sm-spark-dot dot-classic" style="left: ${bestVal.toFixed(1)}%;" title="Best Classical: ${bestVal.toFixed(1)}%"></div>
              <div class="sm-spark-dot dot-jev" style="left: ${jevVal.toFixed(1)}%;" title="Jev Zero-Shot: ${jevVal.toFixed(1)}%"></div>
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
              <span class="sm-stat-val text-accent">${jevVal.toFixed(1)}%</span>
            </div>
            <div class="sm-stat">
              <span class="sm-stat-label">Best ML (${escape(bestModel.slice(0, 12))})</span>
              <span class="sm-stat-val text-muted">${bestVal.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      `;
    }).join('');
  };

  const renderSummaryTable = (rows) => {
    const tbody = document.querySelector('#summary-table tbody');
    if (!tbody) return;

    tbody.innerHTML = rows.map(row => {
      const dVal = row.delta;
      const deltaCls = dVal > 0 ? 'delta-pos' : (dVal < 0 ? 'delta-neg' : 'delta-neutral');
      const deltaStr = dVal > 0 ? `+${dVal.toFixed(1)}%` : `${dVal.toFixed(1)}%`;
      const bestModelName = row.bestModels.join(', ');
      const jevZero = row.scores['Jev zero-shot'].mean;
      const jevFew = row.scores['Jev few-shot'].mean;
      const isJevWinner = jevZero >= row.best;
      const isClassicWinner = row.best > jevZero;

      return `
        <tr class="summary-row" data-dataset="${escape(row.dataset)}" tabindex="0" role="button" onclick="window.openDrawer('${escape(row.dataset)}')">
          <td class="font-medium text-foreground"><span class="ds-name">${escape(row.dataset)}</span></td>
          <td><span class="badge-domain badge-${row.kind}">${row.domain}</span></td>
          <td class="text-right font-mono">${row.classes}</td>
          <td class="text-right font-mono">${row.testRows.toLocaleString('en-US')}</td>
          <td class="text-right font-mono font-semibold ${isJevWinner ? 'score-winner' : ''}">${jevZero.toFixed(1)}% <span class="sd-sub">±${row.scores['Jev zero-shot'].sd.toFixed(1)}</span></td>
          <td class="text-right font-mono">${jevFew.toFixed(1)}% <span class="sd-sub">±${row.scores['Jev few-shot'].sd.toFixed(1)}</span></td>
          <td class="text-right font-mono font-semibold ${isClassicWinner ? 'score-winner' : ''}">${row.best.toFixed(1)}% <span class="best-model-label">(${escape(bestModelName)})</span></td>
          <td class="text-right font-mono"><span class="delta-chip ${deltaCls}">${deltaStr}</span></td>
          <td class="text-center"><span class="inspect-tag font-mono">inspect →</span></td>
        </tr>
      `;
    }).join('');
  };

  const renderFullMatrix = (panelData, rows) => {
    const table = document.getElementById('full-table');
    if (!table) return;

    const models = ['Jev zero-shot', 'Jev few-shot', ...panelData.models.filter(m => !m.startsWith('Jev '))];
    table.innerHTML = `
      <caption class="sr-only">Balanced accuracy: mean ± sample standard deviation, three seeds.</caption>
      <thead>
        <tr>
          <th scope="col" class="sticky-col">Dataset</th>
          ${models.map(m => `<th scope="col" class="text-right">${escape(m)}</th>`).join('')}
        </tr>
      </thead>
      <tbody>
        ${rows.map(row => {
          const best = Math.max(...Object.values(row.scores).map(s => s.mean));
          return `
            <tr onclick="window.openDrawer('${escape(row.dataset)}')">
              <th scope="row" class="sticky-col font-medium">${escape(row.dataset)}</th>
              ${models.map(m => {
                const s = row.scores[m];
                const isBest = s.mean === best;
                return `<td class="text-right font-mono ${isBest ? 'score-winner' : ''}">${s.mean.toFixed(1)} <span class="sd-sub">±${s.sd.toFixed(1)}</span></td>`;
              }).join('')}
            </tr>
          `;
        }).join('')}
      </tbody>
    `;
  };

  // --- Main Render Dispatcher ---
  const render = () => {
    const panel = getActivePanel();
    const domain = getActiveDomain();
    const panelData = data[panel];
    if (!panelData) return;

    let rows = panelData.rows.filter(r => domain === 'all' || r.kind === domain);

    // Apply Sorting
    rows.sort((a, b) => {
      let va, vb;
      switch (currentSort.column) {
        case 'domain': va = a.domain; vb = b.domain; break;
        case 'classes': va = a.classes; vb = b.classes; break;
        case 'testRows': va = a.testRows; vb = b.testRows; break;
        case 'jevZero': va = a.scores['Jev zero-shot'].mean; vb = b.scores['Jev zero-shot'].mean; break;
        case 'jevFew': va = a.scores['Jev few-shot'].mean; vb = b.scores['Jev few-shot'].mean; break;
        case 'best': va = a.best; vb = b.best; break;
        case 'delta': va = a.delta; vb = b.delta; break;
        default: va = a.dataset; vb = b.dataset; break;
      }
      if (typeof va === 'string') {
        return currentSort.direction === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
      }
      return currentSort.direction === 'asc' ? va - vb : vb - va;
    });

    // Update CSV link
    const csvBtn = document.getElementById('csv-download');
    if (csvBtn) {
      csvBtn.href = `data/${panel}_balanced_accuracy.csv`;
      csvBtn.setAttribute('download', `${panel}_balanced_accuracy.csv`);
    }

    // Update Context Note
    const noteEl = document.getElementById('panel-note');
    if (noteEl) {
      noteEl.textContent = panel === 'adjusted'
        ? 'Adjusted: Binary decision thresholds calibrated on dedicated labeled policy split (up to 500 rows). Jev “zero-shot” describes prompt conditioning only. Multiclass tasks remain unadjusted.'
        : 'Raw: Default model decision thresholds without post-hoc adjustment. Jev zero-shot operates strictly without labeled examples or policy threshold fitting.';
    }

    renderDumbbellChart(rows);
    renderDivergingChart(rows);
    renderSwissGrid(rows);
    renderSummaryTable(rows);
    renderFullMatrix(panelData, rows);
  };

  // Expose global drawer helper
  window.openDrawer = openDrawer;
  window.closeDrawer = closeDrawer;

  // --- Interactive Pipeline Stepper (WOW #4) ---
  const initPipeline = () => {
    const pipelineData = {
      '01-split': {
        title: '01 · Data Split & Holdout Isolation',
        desc: 'A dedicated test holdout is isolated before any model exploration (Seed 20260920). It remains completely untouched during feature extraction, hyperparameter tuning, and threshold selection.',
        boxes: [
          { label: 'Train Partition', val: 'Up to 8,000 rows', note: 'Model parameter learning' },
          { label: 'Validation Split', val: 'Up to 1,000 rows', note: 'Hyperparameter candidate tuning' },
          { label: 'Policy Split', val: 'Up to 500 rows', note: 'Threshold search & calibration' },
          { label: 'Test Holdout', val: 'Shared & Frozen', note: 'Final evaluation across 3 seeds' },
        ]
      },
      '02-features': {
        title: '02 · Feature Transformation Pipeline',
        desc: 'Feature extractors are fitted strictly on the training partition and applied as transform-only operations to validation, policy, and test splits.',
        boxes: [
          { label: 'Text TF-IDF', val: '5,000 max features', note: 'Sublinear TF scaling & n-grams' },
          { label: 'Dimensionality', val: 'TruncatedSVD (300d)', note: 'Dense projection for k-NN' },
          { label: 'Tabular Preprocessing', val: 'Imputation + Scaling', note: 'StandardScaler & target encoding' },
          { label: 'Jev Prompts', val: 'Structured JSON', note: 'Zero-shot and 1-shot in-context' },
        ]
      },
      '03-gpu': {
        title: '03 · Multi-GPU Subprocess Isolation',
        desc: 'GPU models run in fresh subprocesses with CUDA_VISIBLE_DEVICES masking to eliminate device contention, state leakage, and memory thrashing.',
        boxes: [
          { label: 'Device Masking', val: 'CUDA_VISIBLE_DEVICES=k', note: 'Hardware isolation per worker' },
          { label: 'Preflight Barrier', val: 'Device probe & UUID test', note: 'Guarantees GPU readiness' },
          { label: 'Worker Reaping', val: 'Explicit process termination', note: 'Clean VRAM recycling' },
          { label: 'Fallback Logic', val: 'Fast-fail exception barrier', note: 'Deterministic error propagation' },
        ]
      },
      '04-calibration': {
        title: '04 · Decision Threshold Calibration',
        desc: 'Policy-calibrated evaluation searches a 101-quantile threshold grid on the dedicated labeled policy split to optimize balanced accuracy.',
        boxes: [
          { label: 'Grid Search', val: '101 uniform quantiles', note: 'Evaluated on policy partition' },
          { label: 'Objective', val: 'Balanced Accuracy', note: 'Equal weight per class' },
          { label: 'Scope', val: 'Binary Datasets', note: 'IMDb, SMS, Bank, Online, Cancer' },
          { label: 'Multiclass Rule', val: 'Argmax default', note: 'AG News, Banking77, Iris unchanged' },
        ]
      },
    };

    document.querySelectorAll('.pipeline-step-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.pipeline-step-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const stepKey = btn.getAttribute('data-step');
        const info = pipelineData[stepKey];
        if (!info) return;

        document.getElementById('pipeline-pane-title').textContent = info.title;
        document.getElementById('pipeline-pane-desc').textContent = info.desc;

        const boxesEl = document.getElementById('pipeline-flow-boxes');
        if (boxesEl) {
          boxesEl.innerHTML = info.boxes.map(b => `
            <div class="flow-box">
              <span class="font-mono" style="font-size: 0.625rem; color: var(--muted-foreground); text-transform: uppercase;">${escape(b.label)}</span>
              <strong style="display: block; font-size: 0.875rem; margin: 0.25rem 0;">${escape(b.val)}</strong>
              <span style="font-size: 0.6875rem; color: var(--muted-foreground);">${escape(b.note)}</span>
            </div>
          `).join('');
        }
      });
    });
  };

  // --- Terminal Tabs ---
  const initTerminal = () => {
    const terminalLogs = {
      'validate': `<span class="term-comment"># Run complete offline validation test suite (zero GPU requirement)</span>
<span class="term-cmd">$ python validate_all.py</span>

<span class="term-pass">✓ validate_benchmark_package.py</span> · Modular benchmark.* package contracts
<span class="term-pass">✓ validate_v3_artifact.py</span> · Source hashes, complete final panels, CSVs & manifest
<span class="term-pass">✓ validate_v3_backends.py</span> · Backend estimator routing matrix & constructors
<span class="term-pass">✓ validate_gpu_process.py</span> · Multi-GPU subprocess device masking & isolation
<span class="term-pass">✓ validate_sampling.py</span> · Proportional capping & stratified split checks

<span class="term-pass font-semibold">========================================================
Summary: 5/5 checks passed in 15.47s
All offline checks PASSED. Ready for evaluation.</span>`,
      'setup': `<span class="term-comment"># Clone repository and create reproducible Python 3.10+ environment</span>
<span class="term-cmd">$ git clone https://github.com/mugenkyou/JEV-VS-ML.git</span>
<span class="term-cmd">$ cd Jev-vs-ML</span>
<span class="term-cmd">$ python -m venv .venv &amp;&amp; source .venv/bin/activate</span>
<span class="term-cmd">$ pip install -r requirements-v2.txt</span>

<span class="term-pass">✓ Dependencies successfully locked &amp; installed</span>`,
      'run': `<span class="term-comment"># Run full benchmark in dual T4 / CUDA GPU environment</span>
<span class="term-cmd">$ python kaggle_benchmark.py</span>

[INFO] Benchmark Protocol 3.0.1 initialized
[INFO] CUDA devices detected: 2 (T4)
[INFO] Running 8 datasets × 11 model families × 3 training seeds...
[INFO] Subprocess worker pools spawned with CUDA_VISIBLE_DEVICES isolation
[INFO] Jev API cache active · requests verified

<span class="term-pass">✓ Benchmark execution complete. Output logged to results/</span>`,
      'export': `<span class="term-comment"># Export canonical CSV score panels & build static site</span>
<span class="term-cmd">$ python export_published_results.py</span>
<span class="term-cmd">$ python build_site.py</span>

<span class="term-pass">✓ Generated published_results/raw_balanced_accuracy.csv</span>
<span class="term-pass">✓ Generated published_results/adjusted_balanced_accuracy.csv</span>
<span class="term-pass">✓ Built docs/index.html and dependency-free visual assets</span>`
    };

    document.querySelectorAll('.term-tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.term-tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const tab = btn.getAttribute('data-term');
        const bodyEl = document.getElementById('terminal-content');
        if (bodyEl && terminalLogs[tab]) {
          bodyEl.innerHTML = terminalLogs[tab];
        }
      });
    });
  };

  // --- Copy Buttons ---
  const initCopyButtons = () => {
    document.querySelectorAll('.copy-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const text = btn.getAttribute('data-copy');
        if (!text) return;
        try {
          await navigator.clipboard.writeText(text);
          const old = btn.textContent;
          btn.textContent = 'Copied!';
          setTimeout(() => { btn.textContent = old; }, 1500);
        } catch (e) {
          console.error(e);
        }
      });
    });
  };

  // --- Init ---
  document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initPipeline();
    initTerminal();
    initCopyButtons();

    const themeToggleBtn = document.getElementById('theme-toggle');
    if (themeToggleBtn) themeToggleBtn.addEventListener('click', toggleTheme);

    const drawerBackdrop = document.getElementById('drawer-backdrop');
    if (drawerBackdrop) drawerBackdrop.addEventListener('click', closeDrawer);

    const drawerCloseBtn = document.getElementById('drawer-close-btn');
    if (drawerCloseBtn) drawerCloseBtn.addEventListener('click', closeDrawer);

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeDrawer();
    });

    document.querySelectorAll('input[name="panel"]').forEach(el => el.addEventListener('change', render));
    document.querySelectorAll('input[name="domain-filter"]').forEach(el => el.addEventListener('change', render));

    // Table sorting
    document.querySelectorAll('th.sortable').forEach(th => {
      th.addEventListener('click', () => {
        const col = th.getAttribute('data-sort');
        if (currentSort.column === col) {
          currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
        } else {
          currentSort.column = col;
          currentSort.direction = 'desc';
        }
        render();
      });
    });

    render();
  });
})();
