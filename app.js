const video = document.querySelector('#video');
const canvas = document.querySelector('#canvas');
const ctx = canvas.getContext('2d');
const statusText = document.querySelector('#statusText');
const statusDot = document.querySelector('#statusDot');
const cameraMessage = document.querySelector('#cameraMessage');
const vectorOutput = document.querySelector('#vectorOutput');
const eventList = document.querySelector('#eventList');
const cameraButton = document.querySelector('#cameraButton');
const latestVectors = { value: '' };
const PROFILE_STORAGE_KEY = 'ekyc-face-profiles-v1';
const SIGNATURE_POINTS = [1, 10, 13, 14, 33, 61, 152, 199, 234, 263, 291, 454];
const LIVENESS_STEPS = ['blink', 'turn-left', 'turn-right'];

const options = {
  mesh: document.querySelector('#meshToggle'),
  points: document.querySelector('#pointsToggle'),
  oval: document.querySelector('#ovalToggle'),
  mirror: document.querySelector('#mirrorToggle'),
  vectors: document.querySelector('#vectorsToggle'),
  density: document.querySelector('#densitySelect')
};

let latestFaces = [];
let camera = null;
let faceMesh = null;
let stream = null;
let lastFaceCount = -1;
let frameCount = 0;
let fpsStartedAt = performance.now();
let adaptiveConnections = [];
let profiles = loadProfiles();
const liveness = {
  stepIndex: 0,
  stepStartedAt: 0,
  openEyesSeen: false,
  eyesClosed: false,
  verified: false,
  lastResetReason: ''
};

function logEvent(message) {
  const item = document.createElement('div');
  item.className = 'event';
  const time = document.createElement('span');
  time.className = 'event-time';
  time.textContent = new Date().toLocaleTimeString();
  const text = document.createElement('span');
  text.textContent = message;
  item.append(time, text);
  eventList.prepend(item);
  while (eventList.children.length > 5) eventList.lastElementChild.remove();
}

function setStatus(message, live = false) {
  statusText.textContent = message;
  statusDot.classList.toggle('live', live);
}

function livenessStepLabel() {
  return {
    blink: 'Blink once',
    'turn-left': 'Turn your head toward screen left',
    'turn-right': 'Turn your head toward screen right'
  }[LIVENESS_STEPS[liveness.stepIndex]];
}

function updateLivenessUi(message, state = 'pending') {
  const value = document.querySelector('#livenessValue');
  const instruction = document.querySelector('#livenessInstruction');
  value.textContent = state === 'verified' ? 'Verified' : state === 'failed' ? 'Reset required' : 'In progress';
  value.className = state;
  instruction.textContent = message;
}

function resetLiveness(reason = 'Show one face to begin the active challenge.') {
  liveness.stepIndex = 0;
  liveness.stepStartedAt = 0;
  liveness.openEyesSeen = false;
  liveness.eyesClosed = false;
  liveness.verified = false;
  liveness.lastResetReason = reason;
  updateLivenessUi(reason);
  document.querySelector('#enrollButton').disabled = true;
}

function distanceBetween(first, second) {
  return Math.hypot(first.x - second.x, first.y - second.y);
}

function eyeAspectRatio(points, upper, lower, corners) {
  const [leftCorner, rightCorner] = corners.map((index) => points[index]);
  const [upperPoint, lowerPoint] = [points[upper], points[lower]];
  if (!leftCorner || !rightCorner || !upperPoint || !lowerPoint) return 0;
  return distanceBetween(upperPoint, lowerPoint) / Math.max(distanceBetween(leftCorner, rightCorner), 0.001);
}

function eyesOpenRatio(points) {
  const left = eyeAspectRatio(points, 159, 145, [33, 133]);
  const right = eyeAspectRatio(points, 386, 374, [362, 263]);
  return (left + right) / 2;
}

function headTurnDirection(points) {
  const nose = points[1];
  const leftEye = points[33];
  const rightEye = points[263];
  if (!nose || !leftEye || !rightEye) return null;
  const eyeCenter = (leftEye.x + rightEye.x) / 2;
  const eyeWidth = Math.max(Math.abs(rightEye.x - leftEye.x), 0.001);
  const yaw = (nose.x - eyeCenter) / eyeWidth;
  if (yaw < -0.12) return 'turn-left';
  if (yaw > 0.12) return 'turn-right';
  return null;
}

function advanceLiveness() {
  liveness.stepIndex += 1;
  liveness.stepStartedAt = performance.now();
  liveness.openEyesSeen = false;
  liveness.eyesClosed = false;
  if (liveness.stepIndex >= LIVENESS_STEPS.length) {
    liveness.verified = true;
    updateLivenessUi('Live response verified. Enrollment and matching are enabled.', 'verified');
    document.querySelector('#enrollButton').disabled = false;
    logEvent('Active liveness challenge passed');
    return;
  }
  updateLivenessUi(livenessStepLabel());
}

function updateLiveness(faces) {
  if (liveness.verified) return;
  if (faces.length !== 1) {
    const message = faces.length > 1 ? 'Only one face may complete the liveness challenge.' : 'Show one face to begin the active challenge.';
    if (liveness.lastResetReason !== message) resetLiveness(message);
    return;
  }

  const points = faces[0].landmarks || faces[0].points || [];
  if (points.length < 455) return;
  if (!liveness.stepStartedAt) {
    liveness.stepStartedAt = performance.now();
    updateLivenessUi(livenessStepLabel());
  }
  if (performance.now() - liveness.stepStartedAt > 12000) {
    resetLiveness('Challenge timed out. Try again when ready.');
    return;
  }

  const step = LIVENESS_STEPS[liveness.stepIndex];
  if (step === 'blink') {
    const ratio = eyesOpenRatio(points);
    if (ratio > 0.2) liveness.openEyesSeen = true;
    if (liveness.openEyesSeen && ratio < 0.14) liveness.eyesClosed = true;
    if (liveness.eyesClosed && ratio > 0.2) advanceLiveness();
  } else if (headTurnDirection(points) === step) {
    advanceLiveness();
  }
}

function loadProfiles() {
  try {
    const saved = JSON.parse(localStorage.getItem(PROFILE_STORAGE_KEY) || '[]');
    return Array.isArray(saved) ? saved : [];
  } catch (error) {
    console.warn('Saved face profiles could not be loaded', error);
    return [];
  }
}

function saveProfiles() {
  localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify(profiles));
}

function faceSignature(points) {
  if (!Array.isArray(points) || points.length < 455) return null;
  const selected = SIGNATURE_POINTS.map((index) => points[index]);
  const xs = selected.map((point) => point.x);
  const ys = selected.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const width = Math.max(maxX - minX, 0.001);
  const height = Math.max(maxY - minY, 0.001);
  return selected.flatMap((point) => [
    (point.x - minX) / width,
    (point.y - minY) / height,
    (point.z || 0) / width
  ]);
}

function matchFace(points) {
  const signature = faceSignature(points);
  if (!signature || !profiles.length) return null;
  let best = null;
  for (const profile of profiles) {
    if (!Array.isArray(profile.signature) || profile.signature.length !== signature.length) continue;
    const distance = Math.sqrt(signature.reduce((sum, value, index) => {
      return sum + (value - profile.signature[index]) ** 2;
    }, 0) / signature.length);
    if (!best || distance < best.distance) best = { profile, distance };
  }
  return best && best.distance <= 0.12 ? best : null;
}

function updateIdentity(faces) {
  if (!liveness.verified) {
    document.querySelector('#identityName').textContent = 'Liveness pending';
    document.querySelector('#identityDetails').textContent = 'Complete the active challenge before identity matching.';
    const badge = document.querySelector('#matchBadge');
    badge.textContent = 'Liveness required';
    badge.classList.remove('matched');
    return;
  }
  const matches = faces.map((face) => matchFace(face.landmarks || face.points || [])).filter(Boolean);
  const name = document.querySelector('#identityName');
  const details = document.querySelector('#identityDetails');
  const badge = document.querySelector('#matchBadge');
  if (!matches.length) {
    name.textContent = 'Unknown face';
    details.textContent = profiles.length ? 'No enrolled profile matched this face.' : 'Enroll a profile, then look at the camera.';
    badge.textContent = profiles.length ? 'No match' : 'No profiles';
    badge.classList.remove('matched');
    return;
  }
  const { profile } = matches[0];
  name.textContent = profile.name;
  details.textContent = [profile.id, profile.details].filter(Boolean).join(' • ') || 'Profile matched';
  badge.textContent = matches.length > 1 ? `${matches.length} matches` : 'Matched';
  badge.classList.add('matched');
}

function resizeCanvas() {
  const bounds = video.getBoundingClientRect();
  const width = Math.max(1, Math.round(bounds.width));
  const height = Math.max(1, Math.round(bounds.height));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function applyMirror() {
  video.style.transform = options.mirror.checked ? 'scaleX(-1)' : 'none';
}

function normalizeFaces(results) {
  if (!results) return [];
  if (Array.isArray(results.multiFaceLandmarks)) {
    return results.multiFaceLandmarks.map((landmarks) => ({ landmarks }));
  }
  if (Array.isArray(results.faces)) return results.faces;
  if (Array.isArray(results)) return results;
  return results.landmarks || results.points ? [results] : [];
}

function pointAt(point, width, height) {
  const x = Math.max(0, Math.min(width, point.x * width));
  const y = Math.max(0, Math.min(height, point.y * height));
  return { x, y };
}

function drawConnections(points, connections, width, height, color, lineWidth) {
  ctx.beginPath();
  for (const [start, end] of connections) {
    const a = points[start];
    const b = points[end];
    if (!a || !b) continue;
    const first = pointAt(a, width, height);
    const second = pointAt(b, width, height);
    ctx.moveTo(first.x, first.y);
    ctx.lineTo(second.x, second.y);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
}

const MESH_REGION_COLORS = {
  eyes: '#ffe680',
  mouth: '#f6c453',
  nose: '#ffd34e',
  outline: '#fff3a3'
};

const EYE_INDICES = new Set([
  33, 133, 157, 158, 159, 160, 161, 246, 263, 362, 384, 385, 386, 387, 388, 466
]);

const MOUTH_INDICES = new Set([
  61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 78, 95, 88, 178,
  87, 14, 317, 402, 318, 324, 308
]);

const NOSE_INDICES = new Set([
  1, 2, 4, 5, 6, 19, 44, 45, 48, 64, 94, 98, 168, 197, 209, 217, 275, 278,
  279, 294, 327, 326, 351, 360, 393, 419, 440, 456
]);

function meshColorForConnection(start, end) {
  if (EYE_INDICES.has(start) || EYE_INDICES.has(end)) return MESH_REGION_COLORS.eyes;
  if (MOUTH_INDICES.has(start) || MOUTH_INDICES.has(end)) return MESH_REGION_COLORS.mouth;
  if (NOSE_INDICES.has(start) || NOSE_INDICES.has(end)) return MESH_REGION_COLORS.nose;
  return MESH_REGION_COLORS.outline;
}

function drawColorMesh(points, connections, width, height, lineWidth) {
  const grouped = new Map();
  for (const [start, end] of connections) {
    const color = meshColorForConnection(start, end);
    if (!grouped.has(color)) grouped.set(color, []);
    grouped.get(color).push([start, end]);
  }
  for (const [color, regionConnections] of grouped) {
    drawConnections(points, regionConnections, width, height, color, lineWidth);
  }
}
function getAdaptiveConnections() {
  if (!adaptiveConnections.length) {
    const tessellation = window.FACEMESH_TESSELATION || [];
    adaptiveConnections = tessellation.filter(([start, end], index) => {
      const regionChange = Math.abs(start - end) > 18;
      return index % 4 === 0 || regionChange;
    });
  }
  return adaptiveConnections;
}

function drawHexagon(points, width, height) {
  const xs = points.map((point) => point.x * width);
  const ys = points.map((point) => point.y * height);
  const left = Math.min(...xs);
  const right = Math.max(...xs);
  const top = Math.min(...ys);
  const bottom = Math.max(...ys);
  const centerX = (left + right) / 2;
  const centerY = (top + bottom) / 2;
  const radiusX = (right - left) * 0.58;
  const radiusY = (bottom - top) * 0.68;
  const vertices = Array.from({ length: 6 }, (_, index) => {
    const angle = -Math.PI / 2 + index * Math.PI / 3;
    return {
      x: centerX + Math.cos(angle) * radiusX,
      y: centerY + Math.sin(angle) * radiusY
    };
  });

  ctx.save();
  ctx.beginPath();
  vertices.forEach((vertex, index) => {
    if (index === 0) ctx.moveTo(vertex.x, vertex.y);
    else ctx.lineTo(vertex.x, vertex.y);
  });
  ctx.closePath();
  ctx.shadowColor = 'rgba(255, 139, 29, 0.95)';
  ctx.shadowBlur = 18;
  ctx.strokeStyle = 'rgba(255, 171, 66, 0.98)';
  ctx.lineWidth = 2.2;
  ctx.stroke();
  ctx.shadowBlur = 5;
  ctx.strokeStyle = 'rgba(255, 231, 178, 0.9)';
  ctx.lineWidth = 0.8;
  ctx.stroke();
  ctx.restore();
}

function drawFaceOverlay(face, width, height) {
  const points = face.landmarks || face.points || [];
  if (points.length < 2) return;

  const adaptiveMesh = getAdaptiveConnections();
  const contours = window.FACEMESH_CONTOURS || [];
  const density = options.density.value;
  const connections = density === 'contours' ? contours : adaptiveMesh;

  ctx.save();
  if (options.mirror.checked) {
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
  }

  if (options.mesh.checked && connections.length) {
    ctx.shadowColor = 'rgba(255, 218, 92, 0.65)';
    ctx.shadowBlur = 5;
    drawColorMesh(points, connections, width, height, density === 'adaptive' ? 1.05 : 1.4);
    ctx.shadowBlur = 0;
  }

  if (options.points.checked) {
    ctx.fillStyle = '#39d353';
    ctx.shadowColor = 'rgba(57, 211, 83, 0.8)';
    ctx.shadowBlur = 6;
    const stride = density === 'keypoints' ? 4 : density === 'adaptive' ? 3 : 1;
    for (let index = 0; index < points.length; index += stride) {
      const point = pointAt(points[index], width, height);
      ctx.beginPath();
      ctx.arc(point.x, point.y, density === 'adaptive' ? 1.6 : 2.1, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.shadowBlur = 0;
  }

  if (options.oval.checked) {
    drawHexagon(points, width, height);
  }
  ctx.restore();
}

function drawResults(results) {
  latestFaces = normalizeFaces(results);
  const width = video.clientWidth || 640;
  const height = video.clientHeight || 480;
  ctx.clearRect(0, 0, width, height);
  latestFaces.forEach((face) => drawFaceOverlay(face, width, height));

  const count = latestFaces.length;
  document.querySelector('#faceCount').textContent = `${count} face${count === 1 ? '' : 's'}`;
  document.querySelector('#peopleValue').textContent = String(count);
  document.querySelector('#trackingValue').textContent = count ? 'Active' : 'Searching';
  updateLiveness(latestFaces);
  updateIdentity(latestFaces);
  if (count !== lastFaceCount) {
    lastFaceCount = count;
    if (count) logEvent(`Detected ${count} face${count > 1 ? 's' : ''}`);
  }

  const vectors = latestFaces.map((face, index) => {
    const points = face.landmarks || face.points || [];
    const values = points.slice(0, 5).flatMap((point) => [point.x, point.y, point.z || 0]);
    return `Face ${index + 1}: ${values.map((value) => value.toFixed(3)).join(', ')}`;
  });
  latestVectors.value = vectors.join('\n');
  vectorOutput.textContent = options.vectors.checked && vectors.length ? vectors.join('\n') : 'Vector console disabled';
}

function updateFps() {
  frameCount += 1;
  const elapsed = performance.now() - fpsStartedAt;
  if (elapsed < 1000) return;
  document.querySelector('#fpsValue').textContent = `${Math.round(frameCount * 1000 / elapsed)} FPS`;
  frameCount = 0;
  fpsStartedAt = performance.now();
}

function onResults(results) {
  const startedAt = performance.now();
  drawResults(results);
  document.querySelector('#latencyValue').textContent = `${Math.round(performance.now() - startedAt)} ms`;
  updateFps();
}

async function startCamera() {
  if (camera) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Camera access is not supported in this browser or context');
  }
  if (typeof FaceMesh !== 'function' || typeof Camera !== 'function') {
    throw new Error('Face Mesh libraries did not load. Check your network connection and reload');
  }

  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
    audio: false
  });
  video.srcObject = stream;
  await video.play();
  resizeCanvas();

  faceMesh = new FaceMesh({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
  });
  faceMesh.setOptions({
    maxNumFaces: 6,
    refineLandmarks: true,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.6
  });
  faceMesh.onResults(onResults);
  camera = new Camera(video, {
    onFrame: async () => faceMesh.send({ image: video }),
    width: 1280,
    height: 720
  });
  camera.start();
  cameraMessage.classList.add('hidden');
  cameraButton.textContent = 'Camera enabled';
  cameraButton.disabled = true;
  document.querySelector('#cameraState').textContent = 'Live';
  document.querySelector('#resolution').textContent = `${video.videoWidth} x ${video.videoHeight}`;
  setStatus('Live tracking', true);
  logEvent('Camera connected; multi-face mesh ready');
}

function showStartupError(error) {
  setStatus('Camera unavailable');
  document.querySelector('#cameraState').textContent = 'Unavailable';
  cameraMessage.classList.remove('hidden');
  cameraMessage.textContent = error.message || 'Unable to start camera';
  cameraButton.disabled = false;
  logEvent(`Startup error: ${error.message || error}`);
  console.error(error);
}

window.addEventListener('resize', resizeCanvas);
cameraButton.addEventListener('click', () => startCamera().catch(showStartupError));
document.querySelector('#copyButton').addEventListener('click', async () => {
  if (!latestVectors.value || !navigator.clipboard) return;
  await navigator.clipboard.writeText(latestVectors.value);
});
document.querySelector('#resetLivenessButton').addEventListener('click', () => {
  resetLiveness();
  logEvent('Active liveness challenge reset');
});
Object.values(options).forEach((control) => {
  control.addEventListener('change', () => {
    if (control === options.mirror) applyMirror();
    if (control === options.vectors) vectorOutput.hidden = !options.vectors.checked;
    if (latestFaces.length) drawResults({ faces: latestFaces });
  });
});

document.querySelector('#enrollButton').addEventListener('click', () => {
  const points = latestFaces[0]?.landmarks || latestFaces[0]?.points;
  const name = document.querySelector('#profileName').value.trim();
  if (!liveness.verified || !points || !name) {
    logEvent(!liveness.verified ? 'Enrollment requires verified liveness' : 'Enrollment requires a visible face and a name');
    return;
  }
  const profile = {
    name,
    id: document.querySelector('#profileId').value.trim(),
    details: document.querySelector('#profileDetails').value.trim(),
    signature: faceSignature(points),
    createdAt: new Date().toISOString()
  };
  profiles = profiles.filter((item) => item.name.toLowerCase() !== name.toLowerCase());
  profiles.push(profile);
  saveProfiles();
  document.querySelector('#profileName').value = '';
  document.querySelector('#profileId').value = '';
  document.querySelector('#profileDetails').value = '';
  logEvent(`Enrolled local profile: ${name}`);
  updateIdentity(latestFaces);
});

document.querySelector('#clearProfilesButton').addEventListener('click', () => {
  profiles = [];
  saveProfiles();
  updateIdentity(latestFaces);
  logEvent('Cleared all saved face profiles');
});

resizeCanvas();
applyMirror();
startCamera().catch(showStartupError);
