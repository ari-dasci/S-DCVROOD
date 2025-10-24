import { state } from './state.js';
import { parseCSV } from './csv.js';
import { renderTable } from './table.js';

function dom() {
    const view = document.querySelector('article.card:not(.hidden)');
    return {
        status: view?.querySelector('[data-status]') || null,
        table: view?.querySelector('[data-table]') || null
    };
}

function setStatus(html) {
    const { status } = dom();
    if (status) { status.classList.remove('hidden'); status.innerHTML = html; }
}
function clearStatus() {
    const { status } = dom();
    if (status) { status.classList.add('hidden'); status.innerHTML = ''; }
}

export function csvUrl() {
    switch (state.section) {
        case 'bt': return `./data/bt-${state.metric}.csv`;
        case 'hr': return `./data/hr-${state.metric}-${state.pvalue}.csv`;
        case 'wilcoxon': return `./data/wilcoxon-${state.metric}.csv`;
        case 'summary': return `./data/summary-${state.pvalue}.csv`;
        case 'significant': return `./data/significant.csv`;
        default: return `./data/unknown.csv`;
    }
}

export async function loadAndRender() {
    clearStatus();
    setStatus(`<span class="row"><span class="spinner"></span><span>Cargando <code>${csvUrl()}</code>…</span></span>`);
    try {
        const fetchCSV = async (url) => {
            const res = await fetch(url, { cache: 'no-store' });
            if (!res.ok) throw new Error(`HTTP ${res.status} en ${url}`);
            const text = await res.text();
            const rows = parseCSV(text);
            if (!rows.length) throw new Error(`CSV vacío en ${url}`);
            return rows;
        };

        let rows = null;
        let helper = null;

        if (state.section === 'hr') {
            const [hrRows, wilRows] = await Promise.all([
                fetchCSV(csvUrl()),
                fetchCSV(`./data/wilcoxon-${state.metric}.csv`)
            ]);
            rows = hrRows;
            helper = wilRows;
        } else {
            rows = await fetchCSV(csvUrl());
        }

        const { table } = dom();
        renderTable(rows, table, { section: state.section, helper, pvalue: state.pvalue });
        clearStatus();
    } catch (err) {
        setStatus(
            `<div role="alert">
        No se pudo cargar <code>${csvUrl()}</code>.<br>
        <small>${err.message}</small><br>
        <em>Revisa nombre y ubicación en <code>/data</code>.</em>
       </div>`
        );
    }
}

export async function loadDatasetsSummary() {
    const card = document.getElementById('view-datasets');
    if (!card) return;

    const status = card.querySelector('[data-status]');
    const table = card.querySelector('table[data-table="datasets"]');

    const set = html => { if (status) { status.classList.remove('hidden'); status.innerHTML = html; } };
    const clear = () => { if (status) { status.classList.add('hidden'); status.innerHTML = ''; } };

    set(`<span class="row"><span class="spinner"></span><span>Loading <code>./data/datasets.csv</code>…</span></span>`);

    try {
        const res = await fetch('./data/datasets.csv', { cache: 'no-store' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const text = await res.text();
        const rows = parseCSV(text);
        if (!rows.length) throw new Error('Empty CSV');

        // Sin highlighting especial en esta sección
        renderTable(rows, table, { section: 'datasets' });
        clear();
    } catch (err) {
        set(`<div role="alert">Could not load <code>./data/datasets.csv</code>.<br><small>${err.message}</small></div>`);
    }
}


