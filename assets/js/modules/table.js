export function renderTable(matrix, tableEl, opts = {}) {
    const { section = 'bt', helper = null, pvalue = '0.1' } = opts;
    const table = tableEl || document.getElementById('datatable');
    if (!table) return;

    table.innerHTML = '';
    if (!matrix?.length) {
        table.innerHTML = '<caption>No hay datos.</caption>';
        return;
    }

    const [head, ...rows] = matrix;
    const lowerHead = head.map(h => String(h).trim().toLowerCase());
    const idxHit = lowerHead.indexOf('hit rate');
    const idxErr = lowerHead.indexOf('error rate');

    const isNum = v => v !== '' && !Number.isNaN(Number(v));
    const fmtNum = n => Number.isInteger(n) ? String(n) : n.toFixed(6);
    const clamp01 = x => Math.max(0, Math.min(1, x));
    const heat = (value, max, kind) => {
        const t = clamp01((Number(value) || 0) / max);
        const eased = Math.pow(t, 1.8);         // contraste alto en zona alta
        const alpha = 0.05 + 0.55 * eased;      // 0.05..0.60
        if (kind === 'green') return `rgba(34,197,94,${alpha})`;
        if (kind === 'red') return `rgba(239,68,68,${alpha})`;
        return '';
    };

    let spans = null;
    if (section === 'bt') {
        spans = Array(3).fill(null).map(() => Array(rows.length).fill(1));
        for (let c = 0; c < 3; c++) {
            let r = 0;
            while (r < rows.length) {
                const val = String(rows[r]?.[c] ?? '').trim();
                if (val !== '') {
                    let k = r + 1;
                    while (k < rows.length && String(rows[k]?.[c] ?? '').trim() === '') k++;
                    const span = k - r;
                    spans[c][r] = span;              
                    for (let i = r + 1; i < k; i++) spans[c][i] = 0;
                    r = k;
                } else {
                    spans[c][r] = 0;
                    r++;
                }
            }
        }
    }

    const thead = document.createElement('thead');
    const trh = document.createElement('tr');
    head.forEach(h => {
        const th = document.createElement('th');
        th.textContent = String(h).trim();
        trh.appendChild(th);
    });
    thead.appendChild(trh);

    const tbody = document.createElement('tbody');

    const highlightFirstCol = ['wilcoxon', 'hr', 'summary', 'significant'].includes(section);
    const highlightFirst3 = (section === 'bt');

    for (let ri = 0; ri < rows.length; ri++) {
        const r = rows[ri];
        const tr = document.createElement('tr');

        let maxIdx = -1;
        if (section === 'bt') {
            let max = -Infinity;
            for (let c = 1; c < r.length; c++) {
                const v = Number(r[c]);
                if (!Number.isNaN(v) && v > max) { max = v; maxIdx = c; }
            }
        }

        for (let ci = 0; ci < r.length; ci++) {
            if (section === 'bt' && ci <= 2 && spans && spans[ci][ri] === 0) continue;

            const td = document.createElement('td');

            if ((highlightFirst3 && ci <= 2) || (highlightFirstCol && ci === 0)) {
                td.classList.add('keycol');
            }

            // Contenido
            const cell = r[ci];

            if (ci > 0 && isNum(cell)) {
                const n = Number(cell);
                let text = fmtNum(n);

                if (section === 'wilcoxon') {
                    let stars = '';
                    if (n < 0.01) stars = '***';
                    else if (n < 0.05) stars = '**';
                    else if (n < 0.1) stars = '*';
                    td.textContent = text + (stars ? stars : '');
                    if (stars) td.classList.add('sig');
                }
                else if (section === 'hr') {
                    td.textContent = text;
                    if (helper && helper[ri + 1] && isNum(helper[ri + 1][ci])) {
                        const threshold = Number(pvalue); // '0.1' | '0.05' | '0.01'
                        const wv = Number(helper[ri + 1][ci]);
                        if (!Number.isNaN(wv) && wv < threshold) td.classList.add('sig');
                    }

                }
                else if (section === 'summary') {
                    td.textContent = text;
                    if (ci === idxHit) { td.style.background = heat(n, 10, 'green'); td.style.boxShadow = 'inset 0 0 0 1px rgba(16,185,129,0.25)'; }
                    if (ci === idxErr) { td.style.background = heat(n, 10, 'red'); td.style.boxShadow = 'inset 0 0 0 1px rgba(239,68,68,0.25)'; }
                }
                else if (section === 'bt') {
                    td.textContent = text;
                    if (ci === maxIdx) td.classList.add('max'); 
                }
                else {
                    td.textContent = text; 
                }
            } else {
                td.textContent = cell;
            }

            if (section === 'bt' && ci <= 2 && spans) {
                const span = spans[ci][ri];
                if (span > 1) td.rowSpan = span;
            }

            tr.appendChild(td);
        }

        tbody.appendChild(tr);
    }

    table.appendChild(thead);
    table.appendChild(tbody);
}
