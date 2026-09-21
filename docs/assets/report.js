(() => {
  'use strict';

  // --- Theme Management ---
  const sunIcon = `<svg class="theme-icon-svg" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>`;
  const moonIcon = `<svg class="theme-icon-svg" viewBox="0 0 24 24"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>`;

  const initTheme = () => {
    try {
      const saved = localStorage.getItem('theme');
      const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
      const theme = saved || (prefersDark ? 'dark' : 'light');
      document.documentElement.setAttribute('data-theme', theme);
      updateThemeToggleUI(theme);
    } catch (e) {
      console.warn('Theme init fallback:', e);
    }
  };

  const updateThemeToggleUI = (theme) => {
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;
    btn.innerHTML = theme === 'dark' ? sunIcon : moonIcon;
    btn.setAttribute('aria-label', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
  };

  const toggleTheme = () => {
    try {
      const current = document.documentElement.getAttribute('data-theme') || 'light';
      const next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
      updateThemeToggleUI(next);
    } catch (e) {
      console.warn('Theme toggle error:', e);
    }
  };

  // --- Benchmark Data Validation ---
  const data = window.BENCHMARK;
  if (!data || !data.raw || !data.adjusted) {
    console.error('Benchmark data artifact is missing or invalid.');
    return;
  }

  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));

  // State
  let currentSort = { column: 'dataset', direction: 'asc' };
  let currentOpenDataset = null;

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
    try {
      const panel = getActivePanel();
      const panelData = data[panel];
      if (!panelData || !panelData.rows) return;

      const row = panelData.rows.find(r => r.dataset === datasetName);
      if (!row) return;

      currentOpenDataset = datasetName;

      const drawer = document.getElementById('dataset-drawer');
      const backdrop = document.getElementById('drawer-backdrop');
      if (!drawer || !backdrop) return;

      // Populate metadata
      const titleEl = document.getElementById('drawer-title');
      const metaEl = document.getElementById('drawer-meta');
      const descEl = document.getElementById('drawer-desc');
      if (titleEl) titleEl.textContent = row.dataset;
      if (metaEl) metaEl.textContent = `${row.domain} Classification · ${row.classes} Classes · ${row.testRows.toLocaleString('en-US')} Test Rows`;
      if (descEl) descEl.textContent = row.description || 'Standard empirical classification task holdout.';

      const jevZero = row.scores['Jev zero-shot'] ? row.scores['Jev zero-shot'].mean : 0;
      const jevFew = row.scores['Jev few-shot'] ? row.scores['Jev few-shot'].mean : 0;
      const classicBest = row.best || 0;
      const bestModelName = (row.bestModels && row.bestModels.length) ? row.bestModels.join(', ') : 'Classical Pipeline';
      const dVal = row.delta || 0;

      const valJevZero = document.getElementById('drawer-val-jev-zero');
      const valJevFew = document.getElementById('drawer-val-jev-few');
      const valClassic = document.getElementById('drawer-val-classic');
      const lblClassic = document.getElementById('drawer-lbl-classic');
      const deltaEl = document.getElementById('drawer-val-delta');

      if (valJevZero) valJevZero.textContent = `${jevZero.toFixed(1)}%`;
      if (valJevFew) valJevFew.textContent = `${jevFew.toFixed(1)}%`;
      if (valClassic) valClassic.textContent = `${classicBest.toFixed(1)}%`;
      if (lblClassic) lblClassic.textContent = `Best ML (${bestModelName})`;
      
      if (deltaEl) {
        deltaEl.textContent = dVal > 0 ? `+${dVal.toFixed(1)} pp` : `${dVal.toFixed(1)} pp`;
        deltaEl.className = `drawer-m-val font-mono ${dVal > 0 ? 'text-accent' : 'text-muted'}`;
      }

      // Dispersion Plot
      const varBox = document.getElementById('drawer-variance-content');
      if (varBox) {
        const displayModels = ['Jev zero-shot', 'Jev few-shot', ...(row.bestModels || []).slice(0, 2)];
        varBox.innerHTML = displayModels.map(m => {
          const s = row.scores[m] || { mean: 0, sd: 0 };
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
                <div style="position: absolute; left: ${minX}%; width: ${Math.max(2, maxX - minX)}%; height: 100%; background: ${isJev ? 'var(--bar-jev)' : 'var(--bar-classic)'}; opacity: 0.35; border-radius: 2px;"></div>
                <div style="position: absolute; left: ${s.mean}%; top: -2px; width: 10px; height: 10px; border-radius: 50%; background: ${isJev ? 'var(--bar-jev)' : 'var(--bar-classic)'}; transform: translateX(-50%); border: 2px solid var(--card);"></div>
              </div>
            </div>
          `;
        }).join('');
      }

      // Complete 14-Models Score Breakdown
      const gridEl = document.getElementById('drawer-models-list');
      if (gridEl && panelData.models) {
        gridEl.innerHTML = panelData.models.map(m => {
          const s = row.scores[m] || { mean: 0 };
          const allMeans = Object.values(row.scores).map(sc => sc.mean);
          const maxScore = Math.max(...allMeans);
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
    } catch (err) {
      console.error('Error opening drawer:', err);
    }
  };

  const closeDrawer = () => {
    currentOpenDataset = null;
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

    try {
      if (!rows || !rows.length) {
        container.innerHTML = '<p class="text-muted text-center" style="padding: 2rem 0;">No dataset results match current filter.</p>';
        return;
      }

      const hRow = 44;
      const paddingTop = 20;
      const paddingBottom = 30;
      const totalHeight = paddingTop + rows.length * hRow + paddingBottom;
      const xOffset = 140;
      const widthChart = 620;

      let lines = [`<svg class="dumbbell-svg" viewBox="0 0 800 ${totalHeight}" width="100%" height="${totalHeight}">`];

      for (const tick of [0, 25, 50, 75, 100]) {
        const gx = xOffset + (tick / 100.0) * widthChart;
        lines.push(`<line x1="${gx}" y1="${paddingTop}" x2="${gx}" y2="${totalHeight - paddingBottom}" stroke="var(--border)" stroke-width="1" stroke-dasharray="3 3"/>`);
        lines.push(`<text x="${gx}" y="${totalHeight - 10}" text-anchor="middle" font-size="10" font-family="ui-monospace, monospace" fill="var(--muted-foreground)">${tick}%</text>`);
      }

      rows.forEach((row, i) => {
        const y = paddingTop + i * hRow + 22;
        const jevScore = row.scores['Jev zero-shot'] ? row.scores['Jev zero-shot'].mean : 0;
        const classicScore = row.best || 0;
        const xJev = xOffset + (jevScore / 100.0) * widthChart;
        const xClassic = xOffset + (classicScore / 100.0) * widthChart;
        const xMin = Math.min(xJev, xClassic);
        const xMax = Math.max(xJev, xClassic);
        const trackColor = jevScore >= classicScore ? 'var(--accent)' : 'var(--muted-foreground)';

        lines.push(`
          <g class="dumbbell-row" data-dataset="${escape(row.dataset)}" tabindex="0" role="button" onclick="window.openDrawer('${escape(row.dataset)}')">
            <text x="130" y="${y + 4}" text-anchor="end" font-size="12" font-weight="600" fill="var(--foreground)">${escape(row.dataset)}</text>
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
    } catch (err) {
      console.error('Error rendering dumbbell chart:', err);
      container.innerHTML = '<p class="text-muted text-center" style="padding: 1.5rem 0;">Unable to render comparison chart.</p>';
    }
  };

  const renderDivergingChart = (rows) => {
    const container = document.getElementById('diverging-chart-wrap');
    if (!container) return;

    try {
      if (!rows || !rows.length) {
        container.innerHTML = '<p class="text-muted text-center" style="padding: 2rem 0;">No dataset results match current filter.</p>';
        return;
      }

      const hRow = 38;
      const paddingTop = 20;
      const paddingBottom = 25;
      const totalHeight = paddingTop + rows.length * hRow + paddingBottom;
      const minVal = -40.0;
      const maxVal = 15.0;
      const rangeVal = maxVal - minVal;
      const chartX = 150;
      const chartW = 600;
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
        const dVal = row.delta || 0;
        const bx = chartX + ((Math.min(0, dVal) - minVal) / rangeVal) * chartW;
        const bw = (Math.abs(dVal) / rangeVal) * chartW;
        const fillColor = dVal > 0 ? 'var(--delta-pos-fg)' : 'var(--bar-classic)';
        const textX = zeroX + (dVal >= 0 ? bw + 6 : -(bw + 6));
        const anchor = dVal >= 0 ? 'start' : 'end';
        const dStr = dVal > 0 ? `+${dVal.toFixed(1)} pp` : `${dVal.toFixed(1)} pp`;

        lines.push(`
          <g class="diverging-row" data-dataset="${escape(row.dataset)}" onclick="window.openDrawer('${escape(row.dataset)}')">
            <text x="140" y="${y + 11}" text-anchor="end" font-size="11" font-weight="500" fill="var(--foreground)">${escape(row.dataset)}</text>
            <rect x="${bx}" y="${y}" width="${Math.max(2, bw)}" height="14" fill="{fillColor}" rx="2" opacity="0.85"/>
            <text x="${textX}" y="${y + 11}" text-anchor="${anchor}" font-size="9" font-family="ui-monospace, monospace" font-weight="600" fill="${fillColor}">${dStr}</text>
          </g>
        `);
      });

      lines.push('</svg>');
      container.innerHTML = lines.join('');
    } catch (err) {
      console.error('Error rendering diverging chart:', err);
      container.innerHTML = '<p class="text-muted text-center" style="padding: 1.5rem 0;">Unable to render difference chart.</p>';
    }
  };

  const renderSwissGrid = (rows) => {
    const container = document.getElementById('swiss-dataset-grid');
    if (!container) return;

    try {
      if (!rows || !rows.length) {
        container.innerHTML = '<p class="text-muted text-center" style="grid-column: 1 / -1; padding: 2rem 0;">No datasets match current filter.</p>';
        return;
      }

      container.innerHTML = rows.map(row => {
        const dVal = row.delta || 0;
        const dStr = dVal > 0 ? `+${dVal.toFixed(1)}%` : `${dVal.toFixed(1)}%`;
        const dCls = dVal > 0 ? 'delta-pos' : (dVal < 0 ? 'delta-neg' : 'delta-neutral');
        const jevVal = row.scores['Jev zero-shot'] ? row.scores['Jev zero-shot'].mean : 0;
        const bestVal = row.best || 0;
        const bestModel = (row.bestModels && row.bestModels.length) ? row.bestModels.join(', ') : 'Classical';

        return `
          <div class="sm-card" data-dataset="${escape(row.dataset)}" tabindex="0" role="button" onclick="window.openDrawer('${escape(row.dataset)}')">
            <div class="sm-card-head">
              <div>
                <span class="sm-card-domain font-mono">${escape(row.domain)} · ${row.classes} cl</span>
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
    } catch (err) {
      console.error('Error rendering Swiss grid:', err);
    }
  };

  const renderSummaryTable = (rows) => {
    const tbody = document.querySelector('#summary-table tbody');
    if (!tbody) return;

    try {
      if (!rows || !rows.length) {
        tbody.innerHTML = '<tr><td colspan="9" class="text-center text-muted" style="padding: 2rem;">No dataset records match current criteria.</td></tr>';
        return;
      }

      tbody.innerHTML = rows.map(row => {
        const dVal = row.delta || 0;
        const deltaCls = dVal > 0 ? 'delta-pos' : (dVal < 0 ? 'delta-neg' : 'delta-neutral');
        const deltaStr = dVal > 0 ? `+${dVal.toFixed(1)}%` : `${dVal.toFixed(1)}%`;
        const bestModelName = (row.bestModels && row.bestModels.length) ? row.bestModels.join(', ') : 'Classical';
        const jevZero = row.scores['Jev zero-shot'] ? row.scores['Jev zero-shot'].mean : 0;
        const jevFew = row.scores['Jev few-shot'] ? row.scores['Jev few-shot'].mean : 0;
        const isJevWinner = jevZero >= row.best;
        const isClassicWinner = row.best > jevZero;

        return `
          <tr class="summary-row" data-dataset="${escape(row.dataset)}" tabindex="0" role="button" onclick="window.openDrawer('${escape(row.dataset)}')">
            <td class="font-medium text-foreground"><span class="ds-name">${escape(row.dataset)}</span></td>
            <td><span class="badge-domain badge-${row.kind}">${escape(row.domain)}</span></td>
            <td class="text-right font-mono">${row.classes}</td>
            <td class="text-right font-mono">${row.testRows.toLocaleString('en-US')}</td>
            <td class="text-right font-mono font-semibold ${isJevWinner ? 'score-winner' : ''}">${jevZero.toFixed(1)}% <span class="sd-sub">±${row.scores['Jev zero-shot'] ? row.scores['Jev zero-shot'].sd.toFixed(1) : '0.0'}</span></td>
            <td class="text-right font-mono">${jevFew.toFixed(1)}% <span class="sd-sub">±${row.scores['Jev few-shot'] ? row.scores['Jev few-shot'].sd.toFixed(1) : '0.0'}</span></td>
            <td class="text-right font-mono font-semibold ${isClassicWinner ? 'score-winner' : ''}">${row.best.toFixed(1)}% <span class="best-model-label">(${escape(bestModelName)})</span></td>
            <td class="text-right font-mono"><span class="delta-chip ${deltaCls}">${deltaStr}</span></td>
            <td class="text-center"><span class="inspect-tag font-mono">inspect →</span></td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      console.error('Error rendering summary table:', err);
    }
  };

  const updateSortIndicators = () => {
    document.querySelectorAll('th.sortable').forEach(th => {
      const col = th.getAttribute('data-sort');
      const baseText = th.textContent.replace(/[ ↕▲▼]/g, '');
      if (col === currentSort.column) {
        th.textContent = `${baseText} ${currentSort.direction === 'asc' ? '▲' : '▼'}`;
        th.style.color = 'var(--foreground)';
      } else {
        th.textContent = `${baseText} ↕`;
        th.style.color = '';
      }
    });
  };

  // --- Main Render Dispatcher ---
  const render = () => {
    try {
      const panel = getActivePanel();
      const domain = getActiveDomain();
      const panelData = data[panel];
      if (!panelData || !panelData.rows) return;

      let rows = panelData.rows.filter(r => domain === 'all' || r.kind === domain);

      // Robust Numeric & String Sorting
      rows.sort((a, b) => {
        let va, vb;
        switch (currentSort.column) {
          case 'domain': va = a.domain; vb = b.domain; break;
          case 'classes': va = parseInt(a.classes, 10); vb = parseInt(b.classes, 10); break;
          case 'testRows': va = parseInt(a.testRows, 10); vb = parseInt(b.testRows, 10); break;
          case 'jevZero': va = parseFloat(a.scores['Jev zero-shot'].mean); vb = parseFloat(b.scores['Jev zero-shot'].mean); break;
          case 'jevFew': va = parseFloat(a.scores['Jev few-shot'].mean); vb = parseFloat(b.scores['Jev few-shot'].mean); break;
          case 'best': va = parseFloat(a.best); vb = parseFloat(b.best); break;
          case 'delta': va = parseFloat(a.delta); vb = parseFloat(b.delta); break;
          default: va = a.dataset; vb = b.dataset; break;
        }
        if (typeof va === 'string') {
          return currentSort.direction === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
        }
        return currentSort.direction === 'asc' ? va - vb : vb - va;
      });

      // Update CSV download link
      const csvBtn = document.getElementById('csv-download');
      if (csvBtn) {
        csvBtn.href = `data/${panel}_balanced_accuracy.csv`;
        csvBtn.setAttribute('download', `${panel}_balanced_accuracy.csv`);
      }

      renderDumbbellChart(rows);
      renderDivergingChart(rows);
      renderSwissGrid(rows);
      renderSummaryTable(rows);
      updateSortIndicators();

      // If drawer is currently open, synchronize its content with the active panel
      if (currentOpenDataset) {
        openDrawer(currentOpenDataset);
      }
    } catch (err) {
      console.error('Render error:', err);
    }
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

        const paneTitle = document.getElementById('pipeline-pane-title');
        const paneDesc = document.getElementById('pipeline-pane-desc');
        if (paneTitle) paneTitle.textContent = info.title;
        if (paneDesc) paneDesc.textContent = info.desc;

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
          console.error('Clipboard copy failed:', e);
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
