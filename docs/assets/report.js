(() => {
  'use strict';

  // --- Theme Management ---
  const initTheme = () => {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const theme = savedTheme || (prefersDark ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', theme);
    updateThemeToggleUI(theme);
  };

  const updateThemeToggleUI = (theme) => {
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;
    btn.innerHTML = theme === 'dark' 
      ? '<span aria-hidden="true">☀️</span><span class="sr-only">Switch to light mode</span>' 
      : '<span aria-hidden="true">🌙</span><span class="sr-only">Switch to dark mode</span>';
    btn.setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
  };

  const toggleTheme = () => {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeToggleUI(newTheme);
  };

  // --- Benchmark Data & Rendering ---
  const data = window.BENCHMARK;
  if (!data) return;

  const escape = value => String(value).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));

  const summaryTbody = document.querySelector('#summary-table tbody');
  const fullTableEl = document.getElementById('full-table');
  const chartEl = document.getElementById('comparison-chart');
  const panelNoteEl = document.getElementById('panel-note');
  const resultTakeawayEl = document.getElementById('result-takeaway');
  const csvDownloadEl = document.getElementById('csv-download');

  const getActivePanel = () => {
    const checked = document.querySelector('input[name="panel"]:checked');
    return checked ? checked.value : 'raw';
  };

  const getActiveFilter = () => {
    const checked = document.querySelector('input[name="domain-filter"]:checked');
    return checked ? checked.value : 'all';
  };

  const getActiveView = () => {
    const checked = document.querySelector('input[name="view-mode"]:checked');
    return checked ? checked.value : 'summary';
  };

  const render = () => {
    const panel = getActivePanel();
    const kind = getActiveFilter();
    const viewMode = getActiveView();
    const panelData = data[panel];
    if (!panelData) return;

    const isAdjusted = panel === 'adjusted';
    const rows = panelData.rows.filter(row => kind === 'all' || row.kind === kind);

    // Update context note
    if (panelNoteEl) {
      panelNoteEl.textContent = isAdjusted
        ? 'Adjusted: Binary decision thresholds calibrated on dedicated labeled policy data (up to 500 rows). Jev “zero-shot” describes prompt formulation only. Multiclass tasks (AG News, Banking77, Iris) remain unadjusted.'
        : 'Raw: Default model decision rules without post-hoc threshold adjustment. Jev zero-shot operates strictly without labeled examples or policy threshold fitting.';
    }

    // Update CSV download link
    if (csvDownloadEl) {
      csvDownloadEl.href = `data/${panel}_balanced_accuracy.csv`;
      csvDownloadEl.setAttribute('download', `${panel}_balanced_accuracy.csv`);
    }

    // 1. Render Summary Table
    if (summaryTbody) {
      summaryTbody.innerHTML = rows.map(row => {
        const dVal = row.delta;
        const deltaCls = dVal > 0 ? 'delta-pos' : (dVal < 0 ? 'delta-neg' : 'delta-neutral');
        const deltaStr = dVal > 0 ? `+${dVal.toFixed(1)}%` : `${dVal.toFixed(1)}%`;
        const bestModelName = row.bestModels.join(', ');
        const idSafe = row.dataset.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase();
        const jevZeroScore = row.scores['Jev zero-shot'].mean;
        const jevFewScore = row.scores['Jev few-shot'].mean;
        const isJevWinner = jevZeroScore >= row.best;
        const isClassicWinner = row.best > jevZeroScore;

        const scoreCardsHtml = panelData.models.map(m => {
          const s = row.scores[m];
          const allMeans = Object.values(row.scores).map(sc => sc.mean);
          const maxMean = Math.max(...allMeans);
          const isBest = s.mean === maxMean;
          const isJev = m.startsWith('Jev');
          const cardCls = isBest ? 'best-card' : (isJev ? 'jev-card' : '');
          return `<div class="score-card ${cardCls}">
            <span class="score-model-name">${escape(m)}</span>
            <span class="score-val font-mono">${s.mean.toFixed(1)}%</span>
            <span class="score-sd font-mono">±${s.sd.toFixed(1)} (n=${s.n})</span>
          </div>`;
        }).join('');

        return `
          <tr class="summary-row" data-dataset="${escape(row.dataset)}" tabindex="0" role="button" aria-expanded="false" title="Click to inspect model scores for ${escape(row.dataset)}">
            <td class="font-medium text-foreground"><span class="ds-name">${escape(row.dataset)}</span></td>
            <td><span class="badge-domain badge-${row.kind}">${row.domain}</span></td>
            <td class="text-right font-mono">${row.classes}</td>
            <td class="text-right font-mono">${row.testRows.toLocaleString('en-US')}</td>
            <td class="text-right font-mono font-semibold ${isJevWinner ? 'score-winner' : ''}">${jevZeroScore.toFixed(1)}% <span class="sd-sub">±${row.scores['Jev zero-shot'].sd.toFixed(1)}</span></td>
            <td class="text-right font-mono">${jevFewScore.toFixed(1)}% <span class="sd-sub">±${row.scores['Jev few-shot'].sd.toFixed(1)}</span></td>
            <td class="text-right font-mono font-semibold ${isClassicWinner ? 'score-winner' : ''}">${row.best.toFixed(1)}% <span class="best-model-label">(${escape(bestModelName)})</span></td>
            <td class="text-right font-mono"><span class="delta-chip ${deltaCls}">${deltaStr}</span></td>
            <td class="text-center"><span class="row-chevron" aria-hidden="true">▾</span></td>
          </tr>
          <tr class="detail-row hidden" id="detail-${idSafe}">
            <td colspan="9" class="detail-cell">
              <div class="detail-content">
                <div class="detail-header">
                  <div>
                    <h4 class="detail-title">${escape(row.dataset)} <span class="detail-badge">${row.domain} · ${row.classes} classes · ${row.testRows.toLocaleString('en-US')} test rows</span></h4>
                    <p class="detail-desc">${escape(row.description)}</p>
                  </div>
                  <div class="detail-meta-group">
                    <span class="meta-tag">Holdout Seed: <code>20260920</code></span>
                    <span class="meta-tag">Training Seeds: <code>2027, 2028, 2029</code></span>
                  </div>
                </div>
                <div class="detail-scores-grid">
                  ${scoreCardsHtml}
                </div>
              </div>
            </td>
          </tr>
        `;
      }).join('');
    }

    // 2. Render Full Matrix Table
    if (fullTableEl) {
      const models = ['Jev zero-shot', 'Jev few-shot', ...panelData.models.filter(m => !m.startsWith('Jev '))];
      fullTableEl.innerHTML = `
        <caption class="sr-only">${isAdjusted ? 'Threshold-adjusted' : 'Raw'} balanced accuracy: mean ± sample standard deviation, three seeds.</caption>
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
              <tr>
                <th scope="row" class="sticky-col font-medium">${escape(row.dataset)}</th>
                ${models.map(m => {
                  const s = row.scores[m];
                  const isBest = s.mean === best;
                  const cls = isBest ? 'font-semibold text-primary score-winner' : '';
                  return `<td class="text-right font-mono ${cls}">${s.mean.toFixed(1)} <span class="sd-sub">±${s.sd.toFixed(1)}</span></td>`;
                }).join('')}
              </tr>
            `;
          }).join('')}
        </tbody>
      `;
    }

    // 3. Render Analytical Bar Chart
    if (chartEl) {
      chartEl.innerHTML = rows.map(row => {
        const series = [
          ['zero', row.scores['Jev zero-shot'].mean, 'Jev zero-shot prompts'],
          ['few', row.scores['Jev few-shot'].mean, 'Jev few-shot prompts'],
          ['classic', row.best, 'Best classical: ' + row.bestModels.join(', ')]
        ];
        return `
          <div class="chart-row" data-kind="${row.kind}">
            <div class="chart-label">
              <strong class="chart-ds-title">${escape(row.dataset)}</strong>
              <span class="chart-ds-meta">${row.domain} · ${row.testRows.toLocaleString('en-US')} test rows</span>
            </div>
            <div class="bars">
              ${series.map(([key, val, lbl]) => `
                <div class="bar-line" style="--value:${val}%" aria-label="${escape(lbl)}: ${val.toFixed(1)}%" title="${escape(lbl)}: ${val.toFixed(1)}%">
                  <span class="bar ${key}"></span>
                  <span class="value font-mono">${val.toFixed(1)}%</span>
                </div>
              `).join('')}
            </div>
          </div>
        `;
      }).join('') + '<div class="axis font-mono" aria-hidden="true"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>';
    }

    // 4. Update Dynamic Analytical Takeaway
    if (resultTakeawayEl) {
      const wins = rows.filter(row => row.scores['Jev zero-shot'].mean > row.best);
      const lead = wins.length ? wins.reduce((a, b) => (a.scores['Jev zero-shot'].mean - a.best) > (b.scores['Jev zero-shot'].mean - b.best) ? a : b) : null;
      resultTakeawayEl.textContent = `${isAdjusted ? 'Adjusted zero-shot-prompt' : 'Raw zero-shot'} Jev leads the best classical pipeline mean on ${wins.length ? wins.map(r => r.dataset).join(' and ') : 'none of the selected datasets'}.${lead ? ` The largest advantage is ${lead.dataset}: +${(lead.scores['Jev zero-shot'].mean - lead.best).toFixed(1)} percentage points.` : ''} Across tabular datasets, classical tree and kernel models maintain an empirical lead.`;
    }

    // 5. Toggle View Visibility (Summary vs Full Matrix)
    const summaryWrapper = document.getElementById('summary-table-wrapper');
    const fullTableWrapper = document.getElementById('full-table-wrapper');
    if (summaryWrapper && fullTableWrapper) {
      if (viewMode === 'matrix') {
        summaryWrapper.classList.add('hidden');
        fullTableWrapper.classList.remove('hidden');
      } else {
        summaryWrapper.classList.remove('hidden');
        fullTableWrapper.classList.add('hidden');
      }
    }

    // Attach row expand click handlers
    attachRowExpanders();
  };

  const attachRowExpanders = () => {
    document.querySelectorAll('.summary-row').forEach(row => {
      const toggle = () => {
        const dsName = row.getAttribute('data-dataset');
        const idSafe = dsName.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase();
        const detailRow = document.getElementById(`detail-${idSafe}`);
        if (!detailRow) return;

        const isExpanded = row.classList.contains('expanded');
        if (isExpanded) {
          row.classList.remove('expanded');
          row.setAttribute('aria-expanded', 'false');
          detailRow.classList.add('hidden');
        } else {
          row.classList.add('expanded');
          row.setAttribute('aria-expanded', 'true');
          detailRow.classList.remove('hidden');
        }
      };

      row.onclick = toggle;
      row.onkeydown = (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          toggle();
        }
      };
    });
  };

  // --- Copy Code Snippets ---
  const initCopyButtons = () => {
    document.querySelectorAll('.copy-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const code = btn.getAttribute('data-copy');
        if (!code) return;
        try {
          await navigator.clipboard.writeText(code);
          const originalText = btn.textContent;
          btn.textContent = 'Copied!';
          btn.style.color = 'var(--foreground)';
          setTimeout(() => {
            btn.textContent = originalText;
            btn.style.color = '';
          }, 1500);
        } catch (err) {
          console.error('Clipboard copy failed', err);
        }
      });
    });
  };

  // --- Event Listeners ---
  document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initCopyButtons();

    const themeToggleBtn = document.getElementById('theme-toggle');
    if (themeToggleBtn) {
      themeToggleBtn.addEventListener('click', toggleTheme);
    }

    document.querySelectorAll('input[name="panel"]').forEach(el => el.addEventListener('change', render));
    document.querySelectorAll('input[name="domain-filter"]').forEach(el => el.addEventListener('change', render));
    document.querySelectorAll('input[name="view-mode"]').forEach(el => el.addEventListener('change', render));

    render();
  });
})();
