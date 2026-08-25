const form = document.querySelector('#predict-form');
const input = document.querySelector('#image');
const zone = document.querySelector('#drop-zone');
const message = document.querySelector('#message');
const submit = document.querySelector('#submit');
const filename = document.querySelector('#filename');
const preview = document.querySelector('#preview');
const previewWrap = document.querySelector('#preview-wrap');
const dropEmpty = document.querySelector('#drop-empty');
const uploadPage = document.querySelector('#upload-page');
const resultsPage = document.querySelector('#results-page');
const newSample = document.querySelector('#new-sample');
let previewUrl = null;
let hasResults = false;

function showUploadPage() {
  uploadPage.hidden = false;
  resultsPage.hidden = true;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function showResultsPage() {
  uploadPage.hidden = true;
  resultsPage.hidden = false;
  window.scrollTo({ top: 0, behavior: 'auto' });
}

function displayImageUrl(url) {
  return url.startsWith('data:') ? url : `${url}?t=${Date.now()}`;
}

function resetUpload() {
  form.reset();
  input.value = '';
  submit.disabled = true;
  filename.textContent = 'NO FILE SELECTED';
  message.textContent = '';
  zone.classList.remove('has-preview');
  previewWrap.hidden = true;
  dropEmpty.hidden = false;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  showUploadPage();
  history.pushState({ view: 'upload' }, '', '#upload');
}

function setFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    message.textContent = 'Choose a PNG or JPEG image.';
    return;
  }
  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  filename.textContent = file.name;
  submit.disabled = false;
  message.textContent = '';
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  previewWrap.hidden = false;
  dropEmpty.hidden = true;
  zone.classList.add('has-preview');
}

input.addEventListener('change', () => setFile(input.files[0]));
['dragenter', 'dragover'].forEach(event => zone.addEventListener(event, value => {
  value.preventDefault();
  zone.classList.add('drag');
}));
['dragleave', 'drop'].forEach(event => zone.addEventListener(event, value => {
  value.preventDefault();
  zone.classList.remove('drag');
}));
zone.addEventListener('drop', event => setFile(event.dataTransfer.files[0]));
newSample.addEventListener('click', resetUpload);

document.querySelector('.brand').addEventListener('click', event => {
  if (hasResults) {
    event.preventDefault();
    resetUpload();
  }
});

document.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !resultsPage.hidden && input.files.length && !submit.disabled && document.activeElement?.tagName !== 'BUTTON') {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener('submit', async event => {
  event.preventDefault();
  if (!input.files.length) {
    message.textContent = 'Choose a PNG or JPEG smear image first.';
    return;
  }
  message.textContent = '';
  submit.disabled = true;
  submit.textContent = 'ANALYSING...';
  try {
    const response = await fetch('/api/predict', { method: 'POST', body: new FormData(form) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Analysis failed.');
    document.querySelector('#detected').textContent = data.cells_detected;
    document.querySelector('#abnormal').textContent = data.abnormal_cells;
    document.querySelector('#annotated').src = displayImageUrl(data.annotated_url);
    const reconstruction = document.querySelector('#reconstruction-figure');
    reconstruction.hidden = !data.reconstruction_url;
    if (data.reconstruction_url) document.querySelector('#reconstruction').src = displayImageUrl(data.reconstruction_url);
    hasResults = true;
    history.pushState({ view: 'results' }, '', '#results');
    showResultsPage();
  } catch (error) {
    message.textContent = error.message;
  } finally {
    submit.disabled = false;
    submit.textContent = 'RUN ANALYSIS ↗';
  }
});

window.addEventListener('popstate', () => {
  if (location.hash === '#results' && hasResults) showResultsPage();
  else showUploadPage();
});
