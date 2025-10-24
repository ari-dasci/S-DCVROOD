import { state } from './state.js';

const metricLabels = {
    auroc: 'AUROC',
    aupr: 'AUPR',
    tprfpr: 'TPR5',
    //fprtpr: 'FPR@TPR',
    //acc99: 'Acc-99',
    //acc95: 'Acc-95',
    acc90: 'Acc-90'
};

function template(html, data = {}) {
    return html.replace(/{{\s*(\w+)\s*}}/g, (_, k) => (k in data ? data[k] : ''));
}

function paramsFor(section) {
    return {
        metricLabel: metricLabels[state.metric] || state.metric,
        pvalue: state.pvalue
    };
}

export async function loadPartialFor(section) {
    const card = document.getElementById(`view-${section}`);
    if (!card) return;
    const host = card.querySelector('[data-partial]');
    if (!host) return;

    const url = `./assets/content/${section}.html`;
    try {
        const res = await fetch(url, { cache: 'no-store' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const html = await res.text();
        host.innerHTML = template(html, paramsFor(section));
    } catch {
        host.innerHTML = '';
    }
}
