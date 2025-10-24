import { state } from './state.js';
import { loadAndRender } from './data.js';
import { initRouter } from './router.js';

const REPO_ZIP_URL = 'https://github.com/ari-dasci/S-DCVROOD/archive/refs/heads/main.zip'; 
const DATA_ZIP_URL = './downloads/data-v1.0.zip';

function triggerDownload(url, filename) {
    const a = document.createElement('a');
    a.href = url;
    if (filename) a.download = filename; // respeta si el navegador lo permite
    a.rel = 'noopener';
    document.body.appendChild(a);
    a.click();
    a.remove();
}

function initTopbarDownloads() {
    const dd = document.getElementById('downloads');
    if (!dd) return;

    const toggle = dd.querySelector('.dropdown-toggle');
    const menu = dd.querySelector('.dropdown-menu');

    toggle.addEventListener('click', () => {
        dd.classList.toggle('open');
        const open = dd.classList.contains('open');
        menu.setAttribute('aria-hidden', String(!open));
    });

    document.addEventListener('click', (e) => {
        if (!dd.contains(e.target)) {
            dd.classList.remove('open');
            menu.setAttribute('aria-hidden', 'true');
        }
    });

    menu.addEventListener('click', (e) => {
        const btn = e.target.closest('.dropdown-item');
        if (!btn) return;
        const action = btn.dataset.dl;

        if (action === 'scripts') {
            triggerDownload(REPO_ZIP_URL);
        } else if (action === 'tables') {
            triggerDownload(DATA_ZIP_URL);
        } else if (action === 'all') {
            triggerDownload(REPO_ZIP_URL);
            setTimeout(() => triggerDownload(DATA_ZIP_URL), 400);
        }

        dd.classList.remove('open');
        menu.setAttribute('aria-hidden', 'true');
    });
}


function togglePvalueGroupInVisibleCard() {
    const view = document.querySelector('article.card:not(.hidden)');
    if (!view) return;
    const pGroup = view.querySelector('[data-role="pvalue-group"]');
    if (!pGroup) return;
    pGroup.classList.toggle('hidden', state.section !== 'hr');
}

function initControls() {
    document.addEventListener('click', ev => {
        const chip = ev.target.closest('.chip');
        if (!chip) return;

        const group = chip.dataset.group;   // 'metric' | 'pvalue' | 'section'
        const value = chip.dataset.value;
        if (!group || !value) return;

        const scope = chip.closest('.card') || document;
        scope.querySelectorAll(`.chip[data-group="${group}"]`)
            .forEach(b => b.dataset.active = 'false');
        chip.dataset.active = 'true';

        if (group === 'metric') state.metric = value;
        if (group === 'pvalue') state.pvalue = value;
        if (group === 'section') {
            state.section = value;           // 'bt' | 'wilcoxon' | 'hr'
            togglePvalueGroupInVisibleCard();
        }

        loadAndRender();
    });
}


function applySidebarState(collapsed) {
    document.body.classList.toggle('sidebar-collapsed', collapsed);
    const btn = document.getElementById('sidebar-toggle');
    if (btn) btn.setAttribute('aria-expanded', String(!collapsed));
}

function initSidebarToggle() {
    const saved = localStorage.getItem('sidebarCollapsed');
    const initial = saved === 'true';
    applySidebarState(initial);

    // Click en botón
    const btn = document.getElementById('sidebar-toggle');
    if (btn) {
        btn.addEventListener('click', () => {
            const nowCollapsed = !document.body.classList.contains('sidebar-collapsed');
            applySidebarState(nowCollapsed);
            localStorage.setItem('sidebarCollapsed', String(nowCollapsed));
        });
    }
}



document.addEventListener('DOMContentLoaded', () => {
    initRouter();
    initControls();
    initSidebarToggle();
    initTopbarDownloads();
    if (!['home', 'datasets'].includes(state.section)) {
        loadAndRender();
    }
});

