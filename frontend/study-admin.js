const $ = id => document.getElementById(id);
const fields = ['title', 'task', 'info', 'instructions', 'debrief'];
const publishedMode = document.body.dataset.adminMode === 'published';
let token;
let saved;
let busy = false;
let dirty = false;

async function request(path, options = {}, format = 'json') {
  const headers = {...options.headers};
  if (token) headers['X-Admin-Token'] = token;
  const response = await fetch(`/study/admin/${path}`, {
    ...options, headers,
  });
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    const detail = data?.detail;
    if (Array.isArray(detail)) throw new Error(detail.map(error => error.loc.slice(1).join('.') + ' — ' + error.msg).join('; '));
    throw new Error(detail || 'The study setup request failed. Try again.');
  }
  return format === 'blob' ? response.blob() : response.json();
}

async function loadVersions() {
  const select = $('export-run');
  const {runs} = await request('runs');
  select.replaceChildren(...runs.map(run => new Option(
    `Version ${run.id} · ${run.title}${run.preview ? ' · Preview' : ''}`,
    run.id, false, run.current,
  )));
  $('export').disabled = !runs.length;
}

function downloadFile(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function status(message, error = false) {
  $('setup-status').textContent = message;
  $('setup-status').classList.toggle('error', error);
}

function markDirty() {
  dirty = true;
  $('download').disabled = true;
  status('Unsaved changes');
}

function updateCounts() {
  const documents = $('documents').childElementCount;
  const questions = $('questions').childElementCount;
  $('document-count').textContent = `${documents} ${documents === 1 ? 'document' : 'documents'}`;
  $('question-count').textContent = `${questions} ${questions === 1 ? 'question' : 'questions'}`;
}

function addDocument(documentData) {
  const group = document.createElement('div');
  group.className = 'material-entry';
  group.innerHTML = `<details><summary></summary><label>Document name<input class="doc-name" required></label>
    <label>Extracted text<textarea class="doc-text" rows="3" required></textarea></label></details>
    <button class="button" type="button" data-remove>Remove document</button>`;
  group.querySelector('.doc-name').value = documentData.name;
  group.querySelector('.doc-text').value = documentData.text;
  group.querySelector('summary').textContent = documentData.name;
  $('documents').append(group);
}

function addQuestion(item) {
  const group = document.createElement('div');
  group.className = 'material-entry';
  group.dataset.id = item.id;
  group.innerHTML = `<details><summary></summary><label>Question<textarea class="prompt" rows="2" required></textarea></label>
    <label>Grading notes<textarea class="answer-key" rows="2"></textarea></label></details>
    <button class="button" type="button" data-remove>Remove question</button>`;
  group.querySelector('.prompt').value = item.prompt;
  group.querySelector('.answer-key').value = item.answer_key;
  group.querySelector('summary').textContent = item.prompt || 'New question';
  $('questions').append(group);
  return group;
}

function render(result) {
  saved = result;
  dirty = false;
  const materials = result.materials;
  for (const key of fields) $(key).value = materials[key];
  $('practice-prompt').value = materials.practice.prompt;
  $('practice-key').value = materials.practice.answer_key;
  $('questions').replaceChildren();
  materials.items.forEach(addQuestion);
  $('documents').replaceChildren();
  materials.documents.forEach(addDocument);
  updateCounts();
  $('download').disabled = !saved.saved;
}

function currentEdit() {
  const value = query => query.value.trim();
  const items = Array.from($('questions').children, group => ({
    id: group.dataset.id, prompt: value(group.querySelector('.prompt')),
    answer_key: value(group.querySelector('.answer-key')),
    choices: saved.materials.items.find(item => item.id === group.dataset.id)?.choices || null,
  }));
  const documents = Array.from($('documents').children, group => ({
    name: value(group.querySelector('.doc-name')), text: group.querySelector('.doc-text').value,
  }));
  if (!items.length || !documents.length) throw new Error('Add at least one task question and one document.');
  return {
    revision: saved.revision, ...Object.fromEntries(fields.map(key => [key, value($(key))])),
    practice: {...saved.materials.practice, prompt: value($('practice-prompt')), answer_key: value($('practice-key'))},
    items, documents,
  };
}

$('unlock').addEventListener('submit', async event => {
  event.preventDefault();
  const button = event.submitter;
  button.disabled = true;
  token = $('token').value;
  try {
    render(await request('materials'));
    $('token').value = '';
    $('unlock').hidden = true;
    $('editor').hidden = false;
    $('title').focus();
    status(saved.saved ? 'Saved draft loaded' : 'Started from current study materials');
  } catch (error) {
    token = undefined;
    $('unlock-status').textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

async function editAction(work) {
  if (busy) return;
  busy = true;
  const controls = Array.from($('editor').elements).filter(element => !element.disabled);
  controls.forEach(element => { element.disabled = true; });
  try {
    await work();
  } catch (error) {
    status(error.message, true);
  } finally {
    controls.forEach(element => { element.disabled = false; });
    $('download').disabled = !saved.saved || dirty;
    busy = false;
  }
}

$('editor').addEventListener('input', event => {
  const field = event.target;
  if (field === $('files')) return;
  if (field.matches('.doc-name, .prompt')) {
    const fallback = field.matches('.doc-name') ? 'Unnamed document' : 'New question';
    field.closest('details').querySelector('summary').textContent = field.value.trim() || fallback;
  }
  markDirty();
});
$('editor').addEventListener('invalid', event => {
  const details = event.target.closest('details');
  if (details) details.open = true;
}, true);
$('editor').addEventListener('click', event => {
  if (!event.target.matches('[data-remove]')) return;
  const entry = event.target.closest('.material-entry');
  const fallback = entry.parentElement.id === 'documents' ? $('add-documents') : $('add-question');
  const nextControl = (entry.nextElementSibling ?? entry.previousElementSibling)?.querySelector('[data-remove]');
  entry.remove();
  (nextControl ?? fallback).focus();
  updateCounts();
  markDirty();
});
$('add-question').addEventListener('click', () => {
  const group = addQuestion({id: 'item-' + crypto.randomUUID(), prompt: '', answer_key: ''});
  group.querySelector('details').open = true;
  updateCounts();
  markDirty();
  group.querySelector('.prompt').focus();
});

$('add-documents').addEventListener('click', () => $('files').click());
$('files').addEventListener('change', () => {
  const files = Array.from($('files').files);
  $('files').value = '';
  if (!files.length) return;
  editAction(async () => {
    const problems = [];
    for (const file of files) {
      status(`Reading ${file.name}…`);
      try {
        const form = new FormData();
        form.append('file', file);
        const data = await request('extract', {method: 'POST', body: form});
        if (Array.from(document.querySelectorAll('.doc-name')).some(input => input.value === data.name)) {
          throw new Error('A document with that name is already in the draft.');
        }
        addDocument(data);
        updateCounts();
        markDirty();
      } catch (error) {
        problems.push(`${file.name}: ${error.message}`);
      }
    }
    if (problems.length) throw new Error(problems.join(' '));
  });
});

$('editor').addEventListener('submit', event => {
  event.preventDefault();
  editAction(async () => {
    status(publishedMode ? 'Saving changes…' : 'Saving draft…');
    const result = await request('materials', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(currentEdit()),
    });
    render(result);
    if (publishedMode) {
      await loadVersions();
      status(result.evaluation_reset
        ? 'New preview published. Prepare new shared answers before collecting data.'
        : 'Changes are live for new participants.');
    } else {
      status(result.evaluation_reset
        ? 'Draft saved. Prepare new shared answer examples before the study.'
        : 'Draft saved for the next run.');
    }
  });
});

$('download').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(saved.materials, null, 2)], {type: 'application/json'});
  downloadFile(blob, 'study-draft.json');
});

$('export').addEventListener('click', () => {
  editAction(async () => {
    const message = $('export-status');
    message.textContent = 'Preparing your download…';
    try {
      const version = publishedMode ? $('export-run').value : '';
      const archive = await request(version ? `export.zip?run=${version}` : 'export.zip', {}, 'blob');
      downloadFile(archive, version ? `live-study-version-${version}.zip` : 'live-study-export.zip');
      message.textContent = 'Export downloaded.';
    } catch (error) {
      message.textContent = error.message;
    }
  });
});

if (publishedMode) {
  $('unlock').hidden = true;
  $('setup-description').textContent = 'Saved changes apply to new participants. Existing sessions keep their version.';
  $('editor').querySelector('button[type=submit]').textContent = 'Save changes';
  $('download').textContent = 'Download study setup';
  $('export-versions').hidden = false;
  $('export').disabled = true;
  request('materials').then(result => {
    render(result);
    $('editor').hidden = false;
    return loadVersions();
  }).catch(error => {
    $('setup-description').textContent = error.message;
  });
}
