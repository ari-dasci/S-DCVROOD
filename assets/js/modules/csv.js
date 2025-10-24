export function parseCSV(text) {
    const rows = []; let i = 0, field = '', row = [], inQuotes = false;
    while (i < text.length) {
        const c = text[i++];
        if (inQuotes) {
            if (c === '"') { if (text[i] === '"') { field += '"'; i++; } else { inQuotes = false; } }
            else field += c;
        } else {
            if (c === '\n' || c === '\r') {
                if (field.length || row.length) { row.push(field); rows.push(row); }
                field = ''; row = [];
                if (c === '\r' && text[i] === '\n') i++;
            } else if (c === ',') { row.push(field); field = ''; }
            else if (c === '"') { inQuotes = true; }
            else field += c;
        }
    }
    if (field.length || row.length) { row.push(field); rows.push(row); }
    return rows.filter(r => r.length && r.some(v => v !== ''));
}
