const $ = id => document.getElementById(id);
const esc = text => String(text).replace(/[&<>"']/g, char => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
}[char]));
let state = {phase: 'consent'};
let information;
let busy = false;
let serverOffset = 0;
let draftTimer;
let viewId;
let pendingAsk;
let pendingSubmit;
let pendingJudgment;
let expirySubmitted = false;
let lastDraft = Promise.resolve();

async function api(path, body, keepalive = false) {
  const response = await fetch('/study' + path, body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body), keepalive,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(typeof error?.detail === 'string' ? error.detail : 'This request could not be completed. Try again.');
  }
  const data = await response.json();
  return data;
}

function setBusy(value) {
  busy = value;
  document.querySelectorAll('button, input, textarea, select').forEach(el => { el.disabled = value; });
}

async function action(work) {
  if (busy) return;
  $('notice').textContent = '';
  setBusy(true);
  try { await work(); }
  catch (error) { $('notice').textContent = error.message; }
  finally { setBusy(false); }
}

function confidence(score) {
  if (!Number.isFinite(score)) return '';
  const value = Math.min(1, Math.max(0, score));
  return `<details class="verdict"><summary><span>System confidence</span>
    <span class="agreement-bar" role="meter" aria-label="System confidence" aria-valuemin="0" aria-valuemax="1" aria-valuenow="${value}" aria-valuetext="${value.toFixed(2)} out of 1"><span class="agreement-fill" style="width:${value * 100}%"></span></span>
    <span class="agreement-score">${value.toFixed(2)}</span><span class="signal-info">How it’s estimated</span></summary>
    <p>Estimated from agreement between two searches of your documents. Higher scores mean more agreement. Check the sources to assess the answer’s accuracy.</p></details>`;
}

function answerText(text, target, kind, sources) {
  return esc(text).replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\[(\d+)\]/g, (match, n) => +n >= 1 && +n <= sources
      ? `<a href="#" data-view="${esc(target)}" data-kind="${kind}" aria-label="Read source ${n}">${n}</a>` : match)
    .split(/\n\s*\n/).map(block => `<p>${block.replace(/\n/g, '<br>')}</p>`).join('');
}

function transcript(turns = state.transcript || []) {
  return turns.map(turn => `<div class="q">${esc(turn.query)}</div><div class="a">
    <div class="body">${turn.answer ? answerText(turn.answer.text, turn.request_id, 'citation', turn.passages.length) : esc(turn.answer_error || 'The assistant could not answer.')}</div>
    ${confidence(turn.score)}<p class="by"><a href="#" class="link" data-view="${esc(turn.request_id)}" data-kind="citation">View ${turn.passages.length} source passages</a></p></div>`).join('');
}

function ratings(name, selected, low = 'Not at all confident', high = 'Completely confident') {
  return `<div class="rating">${Array.from({length: 7}, (_, i) => `<label><input type="radio" name="${name}" value="${i + 1}" ${selected === i + 1 ? 'checked' : ''}>${i + 1}</label>`).join('')}</div><div class="rating-caption">1 — ${esc(low)} · 7 — ${esc(high)}</div>`;
}

function responseFields() {
  const items = state.phase === 'practice' ? [state.practice] : state.items;
  return items.map(item => {
    const draft = state.draft[item.id] || {};
    return `<fieldset><label class="response-label" for="answer-${item.id}">${esc(item.prompt)}</label>
      <textarea class="response-text" id="answer-${item.id}" data-answer="${item.id}" maxlength="4000">${esc(draft.text || '')}</textarea>
      <div role="group" aria-label="Confidence in your response"><p class="rating-caption">How confident are you that your response is correct?</p>${ratings('confidence-' + item.id, draft.confidence)}</div></fieldset>`;
  }).join('');
}

function collectAnswers() {
  const answers = {};
  $('response-fields').querySelectorAll('[data-answer]').forEach(field => {
    const id = field.dataset.answer;
    const checked = $('response-fields').querySelector(`input[name="confidence-${id}"]:checked`);
    answers[id] = {text: field.value, confidence: checked ? Number(checked.value) : null};
  });
  return answers;
}

function adopt(next) {
  if (state.phase !== next.phase) {
    document.querySelectorAll('dialog[open]').forEach(dialog => dialog.close());
  }
  state = next;
  serverOffset = state.server_time ? state.server_time * 1000 - Date.now() : 0;
  expirySubmitted = false;
  render();
}

function render() {
  const phase = state.phase;
  $('preview').hidden = !(state.preview ?? information.preview);
  $('session-code').textContent = state.participant_id || '';
  $('documents').hidden = !['practice', 'task', 'evaluation'].includes(phase);
  $('withdraw').hidden = ['consent', 'complete', 'withdrawn'].includes(phase);
  $('timer').hidden = phase !== 'task';
  const screen = $('screen');
  if (phase === 'consent') {
    screen.innerHTML = `<section class="stage"><h1>${esc(information.title)}</h1><p>${esc(information.info)}</p>
      <form id="consent-form"><label class="check-consent"><input type="checkbox" id="consent" required><span>I have read this information and agree to take part.</span></label><button class="button primary">Agree and continue</button></form></section>`;
  } else if (phase === 'instructions') {
    screen.innerHTML = `<section class="stage"><div class="stage-label">Before you begin</div><h1>How the session works</h1><p>${esc(state.instructions)}</p>
      ${state.confidence_note ? `<p>${esc(state.confidence_note)}</p>` : ''}<button class="button primary" id="begin">Start practice</button></section>`;
  } else if (['practice', 'task'].includes(phase)) {
    const practice = phase === 'practice';
    screen.innerHTML = `<div class="study-workspace">
      <section class="task-panel" aria-labelledby="task-title"><h1 id="task-title">${practice ? 'Practice' : esc(state.title)}</h1>
        <p class="task-brief">${practice ? 'This item is unscored.' : esc(state.task)}</p>
        <a class="link panel-jump" href="#assistant-title">Go to assistant</a>
        <form id="response-form"><div id="response-fields">${responseFields()}</div>
          <div class="response-actions"><button class="button primary" type="submit">${practice ? 'Finish practice' : 'Submit responses'}</button><span id="draft-status" role="status"></span></div>
        </form>
      </section>
      <section class="assistant-panel" aria-labelledby="assistant-title">
        <div class="assistant-intro"><h2 id="assistant-title">Assistant</h2><p>Use the assistant if you choose.</p><a class="link panel-jump" href="#task-title">Back to task</a></div>
        <div class="assistant-scroll" id="assistant-scroll"><div id="transcript" role="log" aria-label="Assistant conversation">${transcript()}</div></div>
        <form id="ask-form" class="composer"><div class="box"><textarea id="query" maxlength="4000" rows="1" placeholder="Ask about the documents…" aria-label="Question for the assistant"></textarea><div class="composer-actions"><span id="ask-status" class="hint" role="status"></span><button class="button primary">Ask</button></div></div></form>
      </section>
    </div>`;
  } else if (phase === 'evaluation') {
    const item = state.evaluation;
    screen.innerHTML = `<section class="stage"><div class="stage-label">Answer evaluation · ${item.position} of ${item.total}</div><h1>Assess this answer</h1><p>You may consult the documents and cited passages.</p>
      <h2 class="q">${esc(item.query)}</h2><div class="evaluation-answer"><div class="body">${answerText(item.answer, item.id, 'evaluation', item.passages.length)}</div>${confidence(item.score)}<p class="by"><a href="#" class="link" data-view="${esc(item.id)}" data-kind="evaluation">View source passages</a></p></div>
      <form id="judgment-form"><fieldset><legend>Is the assistant’s answer correct?</legend><div class="rating">${['correct', 'incorrect', 'unsure'].map(value => `<label><input type="radio" name="judgment" value="${value}" required>${value === 'unsure' ? 'Unsure' : value[0].toUpperCase() + value.slice(1)}</label>`).join('')}</div></fieldset>
      <fieldset><legend>How confident are you in that judgment?</legend>${ratings('judgment-confidence', null)}</fieldset><button class="button primary">Continue</button></form></section>`;
  } else if (phase === 'survey') {
    screen.innerHTML = `<section class="stage"><h1>Your experience</h1><p>Think about the main task. You may skip any item or choose Not applicable.</p><form id="survey-form">${state.survey.map(item => `<fieldset><legend>${esc(item.text)}</legend>${item.type === 'scale' ? ratings('survey-' + item.id, null, item.low, item.high) : `<select name="survey-${item.id}"><option value="">Choose a response</option>${item.options.map(option => `<option value="${esc(option)}">${esc(option)}</option>`).join('')}</select>`}<label class="hint"><input type="checkbox" name="skip-${item.id}"> Not applicable / Prefer not to answer</label></fieldset>`).join('')}<button class="button primary">Finish questionnaire</button></form></section>`;
  } else if (phase === 'interview') {
    screen.innerHTML = `<section class="stage"><h1>Ready for your interview</h1><p>Your responses have been saved. Please let the researcher know you are ready for the short interview.</p><p>Participant code <strong>${esc(state.participant_id)}</strong></p><button class="button primary" id="debrief">After the interview, view the study explanation</button></section>`;
  } else {
    screen.innerHTML = `<section class="stage"><h1>${phase === 'withdrawn' ? 'Your session has stopped' : 'Thank you for taking part'}</h1><p>${phase === 'withdrawn' ? 'Your session is marked as withdrawn and will be excluded from analysis. Give your participant code to the researcher if you also want to request deletion of the stored records.' : esc(state.debrief)}</p><p>Participant code <strong>${esc(state.participant_id)}</strong></p></section>`;
  }
  screen.scrollTop = 0;
  tick();
}

async function submitPhase(extra = {}) {
  clearTimeout(draftTimer);
  await lastDraft;
  const request = {phase: state.phase, ...extra};
  const signature = JSON.stringify(request);
  if (!pendingSubmit || pendingSubmit.signature !== signature) pendingSubmit = {signature, id: crypto.randomUUID()};
  const next = await api('/submit', {...request, request_id: pendingSubmit.id});
  const expired = state.phase === 'task' && (Date.now() + serverOffset) / 1000 >= state.deadline;
  pendingSubmit = null;
  adopt(next);
  if (expired) $('notice').textContent = 'Task time ended. Your current responses were saved and submitted.';
}

document.addEventListener('submit', event => {
  event.preventDefault();
  const form = event.target;
  const formData = new FormData(form);
  action(async () => {
    if (form.id === 'consent-form') adopt(await api('/session', {consent: $('consent').checked}));
    if (form.id === 'ask-form') {
      const query = $('query').value.trim();
      if (!query) return;
      if (!pendingAsk || pendingAsk.query !== query || pendingAsk.phase !== state.phase) pendingAsk = {query, phase: state.phase, request_id: crypto.randomUUID()};
      $('ask-status').textContent = 'Searching the documents and writing an answer…';
      try {
        const turn = await api('/ask', pendingAsk);
        pendingAsk = null;
        state.transcript.push(turn);
        $('transcript').insertAdjacentHTML('beforeend', transcript([turn]));
        $('query').value = '';
        $('assistant-scroll').scrollTop = $('assistant-scroll').scrollHeight;
      } finally { $('ask-status').textContent = ''; }
    }
    if (form.id === 'response-form') await submitPhase({answers: collectAnswers()});
    if (form.id === 'judgment-form') {
      const data = formData;
      if (!data.get('judgment-confidence')) throw new Error('Choose your confidence rating.');
      const judgment = {item_id: state.evaluation.id, judgment: data.get('judgment'), confidence: Number(data.get('judgment-confidence'))};
      const signature = JSON.stringify(judgment);
      if (!pendingJudgment || pendingJudgment.signature !== signature) pendingJudgment = {signature, id: crypto.randomUUID()};
      const next = await api('/judge', {...judgment, request_id: pendingJudgment.id});
      pendingJudgment = null;
      adopt(next);
    }
    if (form.id === 'survey-form') {
      const data = formData;
      const values = {};
      state.survey.forEach(item => {
        const value = data.get('survey-' + item.id);
        values[item.id] = data.has('skip-' + item.id) || !value ? null : item.type === 'scale' ? Number(value) : value;
      });
      await submitPhase({survey: values});
    }
  });
});

document.addEventListener('click', event => {
  const close = event.target.closest('[data-close]');
  if (close) close.closest('dialog').close();
  const source = event.target.closest('[data-view]');
  if (source) {
    event.preventDefault();
    action(async () => {
      const result = await api('/view', {request_id: crypto.randomUUID(), kind: source.dataset.kind, target: source.dataset.view});
      viewId = result.view_id;
      $('reader-content').innerHTML = result.passages.map((passage, i) => `<article class="hit"><div class="src"><b>${i + 1}. ${esc(passage.name)}</b></div><div class="txt">${esc(passage.text)}</div></article>`).join('');
      $('reader').showModal();
    });
  }
  if (event.target.id === 'begin' || event.target.id === 'debrief') action(() => submitPhase());
});

$('documents').addEventListener('click', () => action(async () => {
  const documents = await api('/documents');
  $('document-list').innerHTML = documents.map(doc => `<button class="document-row" data-view="${doc.doc_id}" data-kind="document">${esc(doc.source)}</button>`).join('');
  $('documents-dialog').showModal();
}));
$('withdraw').addEventListener('click', () => {
  if (confirm('Stop taking part? Your session will be marked as withdrawn.')) action(async () => adopt(await api('/withdraw', {})));
});

$('screen').addEventListener('input', event => {
  if (!event.target.closest('#response-fields')) return;
  clearTimeout(draftTimer);
  const status = $('draft-status');
  status.textContent = 'Unsaved changes…';
  draftTimer = setTimeout(() => {
    const body = {phase: state.phase, answers: collectAnswers()};
    state.draft = body.answers;
    lastDraft = lastDraft.then(async () => {
      try { await api('/draft', body); status.textContent = 'Draft saved.'; }
      catch (error) { status.textContent = 'Draft could not be saved. Keep this page open and submit again.'; }
    });
  }, 500);
});

function closeView() {
  if (!viewId) return;
  const id = viewId;
  viewId = null;
  api('/view/close', {request_id: id}, true).catch(() => {});
}
$('reader').addEventListener('close', closeView);
document.addEventListener('visibilitychange', () => {
  if (document.hidden && $('reader').open) $('reader').close();
});
window.addEventListener('pagehide', closeView);

function tick() {
  if (state.phase !== 'task' || !state.deadline) return;
  const remaining = Math.max(0, Math.ceil(state.deadline - (Date.now() + serverOffset) / 1000));
  $('timer').textContent = `${Math.floor(remaining / 60)}:${String(remaining % 60).padStart(2, '0')} remaining`;
  if (!remaining && !busy && !expirySubmitted) {
    expirySubmitted = true;
    action(async () => {
      try { await submitPhase({answers: collectAnswers()}); }
      catch (error) { throw new Error(`Time has ended. Select Submit responses to retry. ${error.message}`); }
    });
  }
}
setInterval(tick, 1000);

(async () => {
  try {
    information = await api('/info');
    adopt(await api('/session'));
  } catch (error) { $('notice').textContent = `Could not load the study. ${error.message} Refresh to retry.`; }
})();
