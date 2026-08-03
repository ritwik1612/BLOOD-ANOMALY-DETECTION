const form = document.querySelector('#predict-form');
const input = document.querySelector('#image');
const zone = document.querySelector('#drop-zone');
const message = document.querySelector('#message');
const submit = document.querySelector('#submit');

function setFile(file) { if (!file) return; const transfer = new DataTransfer(); transfer.items.add(file); input.files = transfer.files; document.querySelector('#filename').textContent = file.name; }
input.addEventListener('change', () => setFile(input.files[0]));
['dragenter','dragover'].forEach(event => zone.addEventListener(event, value => { value.preventDefault(); zone.classList.add('drag'); }));
['dragleave','drop'].forEach(event => zone.addEventListener(event, value => { value.preventDefault(); zone.classList.remove('drag'); }));
zone.addEventListener('drop', event => setFile(event.dataTransfer.files[0]));

form.addEventListener('submit', async event => {
  event.preventDefault(); message.textContent = ''; submit.disabled = true; submit.textContent = 'Analysing…';
  try {
    const response = await fetch('/api/predict', { method: 'POST', body: new FormData(form) });
    const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Analysis failed.');
    document.querySelector('#detected').textContent = data.cells_detected;
    document.querySelector('#abnormal').textContent = data.abnormal_cells;
    document.querySelector('#annotated').src = `${data.annotated_url}?t=${Date.now()}`;
    const reconstruction = document.querySelector('#reconstruction-figure'); reconstruction.hidden = !data.reconstruction_url;
    if (data.reconstruction_url) document.querySelector('#reconstruction').src = `${data.reconstruction_url}?t=${Date.now()}`;
    document.querySelector('#results').hidden = false;
  } catch (error) { message.textContent = error.message; }
  finally { submit.disabled = false; submit.textContent = 'Analyse image'; }
});
