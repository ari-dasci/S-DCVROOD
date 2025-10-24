// assets/js/modules/router.js
import { state } from './state.js';
import { loadAndRender, loadDatasetsSummary } from './data.js';
import { loadPartialFor } from './content.js';

const sections = ['home', 'datasets', 'algorithms', 'metrics', 'bt', 'hr', 'wilcoxon', 'summary', 'significant', 'metric'];

const viewId = s => `view-${s}`;

const metricLabels = {
    auroc: 'AUROC',
    aupr: 'AUPR',
    tprfpr: 'TPR5',
    //fprtpr: 'FPR@TPR',
    //acc99: 'Acc-99',
    //acc95: 'Acc-95',
    acc90: 'Acc-90'
};

export function labelForSection(s) {
    switch (s) {
        case 'home': return 'Home';
        case 'datasets': return 'Datasets';
        case 'algorithms': return 'Algorithms';
        case 'metrics': return 'Metrics';
        case 'bt': return 'Benchmark Truth';
        case 'hr': return 'Hit rate';
        case 'wilcoxon': return 'Wilcoxon p-values';
        case 'summary': return 'Results summary';
        case 'significant': return 'Significant differences';
        case 'metric': return `Metric · ${metricLabels[state.metric] || '—'}`;
        default: return 'Section';
    }
}


const iconForSection = (s) => {
    switch (s) {
        case 'home': return 'fa-house';
        case 'datasets': return 'fa-database';
        case 'algorithms': return 'fa-gears';
        case 'metrics': return 'fa-chart-column';
        case 'bt': return 'fa-chart-line';
        case 'wilcoxon': return 'fa-flask';
        case 'hr': return 'fa-bullseye';
        case 'summary': return 'fa-list-check';
        case 'significant': return 'fa-scale-balanced';
        case 'metric': return 'fa-sliders';
        default: return 'fa-chart-line';
    }
};


function updateBreadcrumb(target) {
    const crumb = document.getElementById('crumb');
    if (crumb) crumb.textContent = labelForSection(target);

    const iconEl = document.getElementById('crumb-icon');
    if (iconEl) iconEl.className = `fa-solid ${iconForSection(target)}`;
}

function showView(target) {
    sections.forEach(s => {
        const el = document.getElementById(viewId(s));
        if (el) el.classList.toggle('hidden', s !== target);
    });
    updateBreadcrumb(target);
    loadPartialFor(target);
}

function resetDefaultsFor(section) {
    const card = document.getElementById(`view-${section}`);
    if (!card) return;

    if (['bt', 'wilcoxon', 'hr'].includes(section)) {
        state.metric = 'auroc';
        card.querySelectorAll('.chip[data-group="metric"]')
            .forEach(b => b.dataset.active = (b.dataset.value === 'auroc') ? 'true' : 'false');
    }

    if (section === 'hr') {
        state.pvalue = '0.05';
        card.querySelectorAll('.chip[data-group="pvalue"]')
            .forEach(b => b.dataset.active = (b.dataset.value === '0.05') ? 'true' : 'false');
    }

    if (section === 'summary') {
        state.pvalue = '0.05';
        card.querySelectorAll('.chip[data-group="pvalue"]')
            .forEach(b => b.dataset.active = (b.dataset.value === '0.05') ? 'true' : 'false');
    }
}


export function initRouter() {
    updateBreadcrumb(state.section);
    loadPartialFor(state.section);

    if (state.section === 'datasets') {
        loadDatasetsSummary();
    }


    document.querySelectorAll('.link[data-section]').forEach(a => {
        a.addEventListener('click', e => {
            e.preventDefault();

            const target = a.dataset.section;

            if (target === 'metric') {
                const metricKey = a.dataset.metric;
                if (!metricKey) return;

                state.metric = metricKey;
                state.section = 'bt'; 

                document.querySelectorAll('.link[data-section]').forEach(x => x.removeAttribute('aria-current'));
                a.setAttribute('aria-current', 'page');

                showView('metric');

                const card = document.getElementById('view-metric');
                if (card) {
                    card.querySelectorAll('.chip[data-group="section"]').forEach(b => {
                        b.dataset.active = (b.dataset.value === state.section) ? 'true' : 'false';
                    });

                    const pGroup = card.querySelector('[data-role="pvalue-group"]');
                    if (pGroup) {
                        const show = (state.section === 'hr');
                        pGroup.classList.toggle('hidden', !show);
                        pGroup.querySelectorAll('.chip[data-group="pvalue"]').forEach(b => {
                            b.dataset.active = (b.dataset.value === state.pvalue) ? 'true' : 'false';
                        });
                    }
                }

                loadAndRender();
                return;
            }

            if (!sections.includes(target)) return;

            state.section = target;

            if (['bt', 'wilcoxon', 'hr', 'summary'].includes(target)) {
                showView(target);
                resetDefaultsFor(target);
            } else {
                showView(target);
            }


            if (target === 'summary') {
                state.pvalue = '0.1';
            }


            document.querySelectorAll('.link[data-section]').forEach(x => x.removeAttribute('aria-current'));
            a.setAttribute('aria-current', 'page');

            if(!['home', 'datasets'].includes(target)){
                showView(target);
            }

            if (target === 'datasets') {
                loadDatasetsSummary();
                return; 
            }

            if (target === 'summary') {
                const card = document.getElementById('view-summary');
                if (card) {
                    card.querySelectorAll('.chip[data-group="pvalue"]')
                        .forEach(b => b.dataset.active = (b.dataset.value === '0.1') ? 'true' : 'false');
                }
            }



            if (!['home', 'datasets'].includes(target)) {
                loadAndRender();
            }
        });
    });
}
