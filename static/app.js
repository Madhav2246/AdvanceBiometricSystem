const statusPill = document.getElementById("status-pill");
const fpsLabel = document.getElementById("fps-label");
const consentBtn = document.getElementById("toggle-consent");
const blurBtn = document.getElementById("toggle-blur");
const enrollBtn = document.getElementById("enroll-btn");
const enrollName = document.getElementById("enroll-name");
const enrollStatus = document.getElementById("enroll-status");
const faceList = document.getElementById("face-list");
const faceEmpty = document.getElementById("face-empty");
const consentVal = document.getElementById("consent-val");
const blurVal = document.getElementById("blur-val");
const faceCount = document.getElementById("face-count");
const fpsVal = document.getElementById("fps-val");
const chart = document.getElementById("rate-chart");
const chartCtx = chart ? chart.getContext("2d") : null;
const timelineChart = document.getElementById("timeline-chart");
const timelineCtx = timelineChart ? timelineChart.getContext("2d") : null;
const consentState = document.getElementById("consent-state");
const blurState = document.getElementById("blur-state");
const cameraIndex = document.getElementById("camera-index");
const applyCamera = document.getElementById("apply-camera");
const exportAudit = document.getElementById("export-audit");
const legendUser = document.getElementById("legend-user");
const legendEmotion = document.getElementById("legend-emotion");
const legendEmotionConf = document.getElementById("legend-emotion-conf");
const legendLiveness = document.getElementById("legend-liveness");
const legendConsent = document.getElementById("legend-consent");
const legendBlur = document.getElementById("legend-blur");
const legendEnroll = document.getElementById("legend-enroll");
const lightingVal = document.getElementById("lighting-val");
const distanceVal = document.getElementById("distance-val");
const brightnessVal = document.getElementById("brightness-val");
const modelShape = document.getElementById("model-shape");
const modelRec = document.getElementById("model-rec");
const modelEmotion = document.getElementById("model-emotion");
const enrollList = document.getElementById("enroll-list");
const enrollEmpty = document.getElementById("enroll-empty");
const enrollSearch = document.getElementById("enroll-search");
const refreshEnrollments = document.getElementById("refresh-enrollments");
const unknownList = document.getElementById("unknown-list");
const unknownEmpty = document.getElementById("unknown-empty");
const quoteEmotion = document.getElementById("quote-emotion");
const quoteText = document.getElementById("quote-text");
let lastQuoteEmotion = null;

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  return res.json();
}

consentBtn.addEventListener("click", async () => {
  await postJson("/api/consent");
});

blurBtn.addEventListener("click", async () => {
  await postJson("/api/blur");
});

if (applyCamera) {
  applyCamera.addEventListener("click", async () => {
    const idx = cameraIndex ? cameraIndex.value : "0";
    const data = await postJson("/api/camera", { index: idx });
    if (!data.ok) {
      alert(data.error || "Failed to switch camera");
    }
  });
}

if (exportAudit) {
  exportAudit.addEventListener("click", async () => {
    const res = await fetch("/api/audit");
    const text = await res.text();
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "events.jsonl";
    a.click();
    URL.revokeObjectURL(url);
  });
}
enrollBtn.addEventListener("click", async () => {
  const name = enrollName.value.trim();
  if (!name) {
    enrollStatus.textContent = "Enter a name first.";
    enrollStatus.style.color = "#ff6b6b";
    return;
  }
  enrollStatus.textContent = "Starting enrollment...";
  enrollStatus.style.color = "#4ef0b0";
  const data = await postJson("/api/enroll", { name });
  if (!data.ok) {
    enrollStatus.textContent = data.error || "Enrollment failed.";
    enrollStatus.style.color = "#ff6b6b";
  }
});

function renderFaces(faces) {
  faceList.innerHTML = "";
  if (!faces || faces.length === 0) {
    faceEmpty.style.display = "block";
    return;
  }
  faceEmpty.style.display = "none";
  faces.forEach((face) => {
    const li = document.createElement("li");
    li.className = "face-item";
    const left = document.createElement("span");
    left.textContent = `${face.user_id}`;
    const right = document.createElement("span");
    right.textContent = `${face.emotion} • ${face.liveness}`;
    li.appendChild(left);
    li.appendChild(right);
    faceList.appendChild(li);
  });
}

function renderUnknowns(unknowns) {
  if (!unknownList) return;
  unknownList.innerHTML = "";
  if (!unknowns || unknowns.length === 0) {
    if (unknownEmpty) unknownEmpty.style.display = "block";
    return;
  }
  if (unknownEmpty) unknownEmpty.style.display = "none";
  unknowns.forEach((u) => {
    const li = document.createElement("li");
    li.className = "face-item";
    const left = document.createElement("span");
    left.textContent = u.label;
    const actions = document.createElement("div");
    actions.className = "face-actions";
    const useBtn = document.createElement("button");
    useBtn.className = "mini-btn";
    useBtn.textContent = "Use";
    useBtn.addEventListener("click", () => {
      enrollName.value = u.label.replace("Unknown", "").trim() ? u.label : "";
    });
    actions.appendChild(useBtn);
    li.appendChild(left);
    li.appendChild(actions);
    unknownList.appendChild(li);
  });
}

async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    statusPill.textContent = "Live";
    statusPill.style.borderColor = "rgba(78,240,176,0.6)";
    statusPill.style.color = "#4ef0b0";
    fpsLabel.textContent = `FPS: ${data.fps ?? "--"}`;
    // Button labels are inside the toggle cards; avoid overwriting inner HTML.
    if (consentState) consentState.textContent = data.consent ? "On" : "Off";
    if (blurState) blurState.textContent = data.blur_unknown ? "On" : "Off";
    if (consentState && consentState.parentElement?.parentElement) {
      consentState.parentElement.parentElement.classList.toggle("active", !!data.consent);
    }
    if (blurState && blurState.parentElement?.parentElement) {
      blurState.parentElement.parentElement.classList.toggle("active", !!data.blur_unknown);
    }
    consentVal.textContent = data.consent ? "On" : "Off";
    blurVal.textContent = data.blur_unknown ? "On" : "Off";
    faceCount.textContent = data.faces ? data.faces.length : 0;
    fpsVal.textContent = data.fps ?? "--";
    renderFaces(data.faces || []);
    renderUnknowns(data.unknowns || []);
    if (legendConsent) legendConsent.textContent = data.consent ? "On" : "Off";
    if (legendBlur) legendBlur.textContent = data.blur_unknown ? "On" : "Off";
    if (data.calibration) {
      if (lightingVal) lightingVal.textContent = data.calibration.lighting;
      if (distanceVal) distanceVal.textContent = data.calibration.distance;
      if (brightnessVal) brightnessVal.textContent = data.calibration.brightness;
    }
    if (data.enroll && enrollStatus) {
      enrollStatus.textContent = data.enroll.message || "Idle";
      if (data.enroll.status === "error") {
        enrollStatus.style.color = "#ff6b6b";
      } else if (data.enroll.status === "done") {
        enrollStatus.style.color = "#4ef0b0";
      } else {
        enrollStatus.style.color = "#9fb0c3";
      }
    }
    if (legendEnroll && data.enroll) {
      legendEnroll.textContent = data.enroll.message || "Idle";
    }
    if (legendEmotionConf) {
      legendEmotionConf.textContent = data.emotion_confidence ?? "--";
    }
    if (data.faces && data.faces.length > 0) {
      const primary = data.faces[0];
      if (legendUser) legendUser.textContent = primary.user_id || "--";
      if (legendEmotion) legendEmotion.textContent = primary.emotion || "--";
      if (legendLiveness) legendLiveness.textContent = primary.liveness || "--";
      triggerQuote(primary.emotion || "neutral");
    } else {
      if (legendUser) legendUser.textContent = "--";
      if (legendEmotion) legendEmotion.textContent = "--";
      if (legendEmotionConf) legendEmotionConf.textContent = "--";
      if (legendLiveness) legendLiveness.textContent = "--";
      triggerQuote("neutral");
    }
  } catch (err) {
    statusPill.textContent = "Offline";
    statusPill.style.borderColor = "rgba(255,255,255,0.1)";
    statusPill.style.color = "#9fb0c3";
  }
  setTimeout(pollStatus, 700);
}

pollStatus();

async function triggerQuote(emotion) {
  if (!quoteText || !quoteEmotion) return;
  const normalized = (emotion || "neutral").toLowerCase();
  if (normalized === "unknown") return;
  if (normalized === lastQuoteEmotion) return;
  lastQuoteEmotion = normalized;
  quoteEmotion.textContent = `Emotion: ${normalized}`;
  try {
    const res = await fetch(`/api/quote?emotion=${encodeURIComponent(normalized)}`);
    const data = await res.json();
    quoteText.textContent = data.quote || "Keep going. You’ve got this.";
  } catch (err) {
    quoteText.textContent = "Keep going. You’ve got this.";
  }
}

async function fetchEnrollments() {
  if (!enrollList) return;
  try {
    const res = await fetch("/api/enrollments");
    const data = await res.json();
    renderEnrollments(data.users || []);
  } catch (err) {
    renderEnrollments([]);
  }
}

function renderEnrollments(users) {
  enrollList.innerHTML = "";
  let filtered = users;
  const q = enrollSearch ? enrollSearch.value.trim().toLowerCase() : "";
  if (q) {
    filtered = users.filter((u) => u.name.toLowerCase().includes(q));
  }
  if (!filtered.length) {
    if (enrollEmpty) enrollEmpty.style.display = "block";
    return;
  }
  if (enrollEmpty) enrollEmpty.style.display = "none";
  filtered.forEach((u) => {
    const li = document.createElement("li");
    li.className = "face-item";
    const left = document.createElement("span");
    left.textContent = `${u.name} (${u.samples})`;
    const actions = document.createElement("div");
    actions.className = "face-actions";
    const del = document.createElement("button");
    del.className = "mini-btn";
    del.textContent = "Delete";
    del.addEventListener("click", async () => {
      const ok = confirm(`Delete ${u.name}?`);
      if (!ok) return;
      await postJson("/api/enrollments/delete", { name: u.name });
      fetchEnrollments();
    });
    actions.appendChild(del);
    li.appendChild(left);
    li.appendChild(actions);
    enrollList.appendChild(li);
  });
}

if (refreshEnrollments) {
  refreshEnrollments.addEventListener("click", fetchEnrollments);
}
if (enrollSearch) {
  enrollSearch.addEventListener("input", fetchEnrollments);
}

setInterval(fetchEnrollments, 5000);
fetchEnrollments();

function drawChart(counts) {
  if (!chartCtx) return;
  const w = chart.width;
  const h = chart.height;
  chartCtx.clearRect(0, 0, w, h);

  chartCtx.fillStyle = "rgba(255,255,255,0.02)";
  chartCtx.fillRect(0, 0, w, h);

  const maxVal = Math.max(1, ...counts);
  const padding = 18;
  const innerW = w - padding * 2;
  const innerH = h - padding * 2;

  chartCtx.strokeStyle = "rgba(255,255,255,0.08)";
  chartCtx.lineWidth = 1;
  chartCtx.beginPath();
  chartCtx.moveTo(padding, padding);
  chartCtx.lineTo(padding, h - padding);
  chartCtx.lineTo(w - padding, h - padding);
  chartCtx.stroke();

  const barW = innerW / counts.length;
  counts.forEach((val, i) => {
    const x = padding + i * barW + 3;
    const barH = (val / maxVal) * innerH;
    const y = h - padding - barH;
    chartCtx.fillStyle = "rgba(78,240,176,0.7)";
    chartCtx.fillRect(x, y, barW - 6, barH);
  });
}

async function pollMetrics() {
  if (!chartCtx) return;
  try {
    const res = await fetch("/api/metrics");
    const data = await res.json();
    drawChart(data.counts || []);
  } catch (err) {
    drawChart([]);
  }
  setTimeout(pollMetrics, 2000);
}

pollMetrics();

function drawLineChart(ctx, counts) {
  if (!ctx) return;
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "rgba(255,255,255,0.02)";
  ctx.fillRect(0, 0, w, h);
  const maxVal = Math.max(1, ...counts);
  const padding = 18;
  const innerW = w - padding * 2;
  const innerH = h - padding * 2;
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding, padding);
  ctx.lineTo(padding, h - padding);
  ctx.lineTo(w - padding, h - padding);
  ctx.stroke();

  ctx.strokeStyle = "rgba(63,182,255,0.8)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  counts.forEach((val, i) => {
    const x = padding + (innerW * i) / Math.max(1, counts.length - 1);
    const y = h - padding - (val / maxVal) * innerH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

async function pollTimeline() {
  if (!timelineCtx) return;
  try {
    const res = await fetch("/api/timeline");
    const data = await res.json();
    drawLineChart(timelineCtx, data.counts || []);
  } catch (err) {
    drawLineChart(timelineCtx, []);
  }
  setTimeout(pollTimeline, 1500);
}

async function pollModelStatus() {
  try {
    const res = await fetch("/api/model_status");
    const data = await res.json();
    if (modelShape) modelShape.textContent = data.dlib_shape_predictor ? "Loaded" : "Missing";
    if (modelRec) modelRec.textContent = data.dlib_recognition_model ? "Loaded" : "Missing";
    if (modelEmotion) modelEmotion.textContent = data.emotion_model ? "Loaded" : "Missing";
  } catch (err) {
    if (modelShape) modelShape.textContent = "--";
    if (modelRec) modelRec.textContent = "--";
    if (modelEmotion) modelEmotion.textContent = "--";
  }
  setTimeout(pollModelStatus, 5000);
}

pollTimeline();
pollModelStatus();
