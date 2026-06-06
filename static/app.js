// ============================================================
// SwiftPlate — Application State
// ============================================================
let currentMode = 'upload';     // 'upload' | 'webcam'
let webcamStream = null;
let webcamInterval = null;
let isProcessingFrame = false;
let historyLog = [];            // Full detection history

// Bounding box overlay state for webcam
let activeDetections = [];
let detectionsExpiryTime = 0;
let animationFrameId = null;

// FPS tracking
let lastFrameTime = 0;
let frameCount = 0;
let fpsValue = 0;
let fpsUpdateInterval = null;

// ============================================================
// DOM Elements
// ============================================================
const dropzone           = document.getElementById('dropzone');
const fileInput          = document.getElementById('file-input');
const previewContainer   = document.getElementById('preview-container');
const annotatedPreview   = document.getElementById('annotated-preview');
const clearUploadBtn     = document.getElementById('clear-upload-btn');

const webcamVideo        = document.getElementById('webcam-video');
const webcamCanvas       = document.getElementById('webcam-canvas');
const webcamPlaceholder  = document.getElementById('webcam-placeholder');
const webcamControls     = document.getElementById('webcam-controls');
const startWebcamBtn     = document.getElementById('start-webcam-btn');
const stopWebcamBtn      = document.getElementById('stop-webcam-btn');
const fpsCounter         = document.getElementById('fps-counter');
const fpsValueEl         = document.getElementById('fps-value');

const noResults          = document.getElementById('no-results');
const activeResults      = document.getElementById('active-results');
const platesContainer    = document.getElementById('plates-container');
const statLatency        = document.getElementById('stat-latency');
const statCount          = document.getElementById('stat-count');

const historyList        = document.getElementById('history-list');
const clearHistoryBtn    = document.getElementById('clear-history-btn');
const appOverlay         = document.getElementById('app-overlay');
const overlayText        = document.getElementById('overlay-text');

// Offscreen canvas for grabbing webcam frames
const offscreenCanvas    = document.createElement('canvas');
const offscreenCtx       = offscreenCanvas.getContext('2d');

// ============================================================
// Bootstrap — Event Listeners
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    // Dropzone interactions
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
    fileInput.addEventListener('change', handleFileSelect);

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) processImageFile(e.dataTransfer.files[0]);
    });

    clearUploadBtn.addEventListener('click', resetUploadView);
    startWebcamBtn.addEventListener('click', startWebcam);
    stopWebcamBtn.addEventListener('click', stopWebcam);
    clearHistoryBtn.addEventListener('click', clearHistory);

    // Global keyboard shortcut: Space triggers detection when image is loaded
    document.addEventListener('keydown', handleGlobalKeydown);
});

// ============================================================
// Keyboard Shortcut Handler
// ============================================================
function handleGlobalKeydown(e) {
    // Space to trigger file picker if in upload mode with dropzone visible
    if (e.code === 'Space' && currentMode === 'upload') {
        const dropzoneVisible = dropzone.style.display !== 'none';
        if (dropzoneVisible && document.activeElement !== dropzone) {
            e.preventDefault();
            fileInput.click();
        }
    }
}

// ============================================================
// Mode Switching
// ============================================================
function switchMode(mode) {
    if (mode === currentMode) return;
    currentMode = mode;

    document.getElementById('mode-upload').classList.toggle('active', mode === 'upload');
    document.getElementById('mode-upload').setAttribute('aria-pressed', mode === 'upload');
    document.getElementById('mode-webcam').classList.toggle('active', mode === 'webcam');
    document.getElementById('mode-webcam').setAttribute('aria-pressed', mode === 'webcam');

    document.getElementById('upload-container').classList.toggle('active', mode === 'upload');
    document.getElementById('webcam-container').classList.toggle('active', mode === 'webcam');

    document.getElementById('feed-badge').innerText = mode === 'upload' ? 'UPLOAD MODE' : 'LIVE MODE';

    if (mode !== 'webcam') {
        stopWebcam();
    } else {
        resetUploadView();
    }
}

// ============================================================
// Loading Overlay
// ============================================================
function showOverlay(show, text = 'Processing...') {
    overlayText.innerText = text;
    appOverlay.style.display = show ? 'flex' : 'none';
}

// ============================================================
// Upload Mode Helpers
// ============================================================
function resetUploadView() {
    fileInput.value = '';
    annotatedPreview.src = '';
    previewContainer.style.display = 'none';
    dropzone.style.display = 'flex';
    resetResultsView();
}

function resetResultsView() {
    activeResults.style.display = 'none';
    noResults.style.display = 'flex';
    if (platesContainer) platesContainer.innerHTML = '';
}

function handleFileSelect(e) {
    if (e.target.files.length > 0) processImageFile(e.target.files[0]);
}

// ============================================================
// Image Processing
// ============================================================
async function processImageFile(file) {
    if (!file.type.match('image.*')) {
        showToast('error', 'Invalid File', 'Please select an image file (PNG, JPEG, or WEBP).');
        return;
    }

    showOverlay(true, 'Analyzing Image...');
    resetResultsView();

    const formData = new FormData();
    formData.append('file', file);
    const startTime = performance.now();

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(err.detail || response.statusText);
        }

        const data = await response.json();
        const latency = Math.round(performance.now() - startTime);

        if (data.success) {
            annotatedPreview.src = data.image;
            dropzone.style.display = 'none';
            previewContainer.style.display = 'flex';
            updateResultsPanel(data.detections, latency);

            if (!data.detections || data.detections.length === 0) {
                showToast('warning', 'No Plates Detected', 'No license plates were found in this image. Try a clearer photo.');
            } else {
                showToast('success', `${data.detections.length} Plate${data.detections.length > 1 ? 's' : ''} Detected`, `Detection completed in ${latency}ms.`);
            }
        } else {
            showToast('error', 'Detection Failed', 'The plate detection process returned an error.');
            resetUploadView();
        }
    } catch (error) {
        console.error(error);
        showToast('error', 'Connection Error', error.message || 'Failed to reach the detection server.');
        resetUploadView();
    } finally {
        showOverlay(false);
    }
}

// ============================================================
// Webcam
// ============================================================
async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width:  { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'environment'
            }
        });

        webcamVideo.srcObject = webcamStream;
        webcamVideo.muted = true;
        webcamVideo.setAttribute('playsinline', true);
        await webcamVideo.play();

        webcamPlaceholder.style.display = 'none';
        webcamControls.style.display = 'block';
        fpsCounter.style.display = 'flex';
        resetResultsView();

        activeDetections = [];
        detectionsExpiryTime = 0;
        lastFrameTime = performance.now();
        frameCount = 0;
        fpsValue = 0;

        drawLocalStream();

        webcamInterval = setInterval(captureAndProcessWebcamFrame, 500);

        // Update FPS display every second
        fpsUpdateInterval = setInterval(() => {
            fpsValueEl.textContent = fpsValue;
        }, 1000);

        showToast('success', 'Camera Started', 'Live detection is now active.');
    } catch (error) {
        console.error('Camera access failed:', error);
        showToast('error', 'Camera Access Denied', 'Unable to access camera. Check browser permissions and use localhost or HTTPS.');
    }
}

// Draw the local camera feed and bounding box overlays
function drawLocalStream() {
    if (!webcamStream || currentMode !== 'webcam') return;

    if (webcamVideo.readyState >= webcamVideo.HAVE_ENOUGH_DATA) {
        const ctx = webcamCanvas.getContext('2d');

        // Dynamically match canvas to current layout size for correct scaling
        const displayWidth  = webcamCanvas.clientWidth  || 640;
        const vidRatio      = webcamVideo.videoHeight / webcamVideo.videoWidth;
        const displayHeight = Math.round(displayWidth * vidRatio) || 480;

        if (webcamCanvas.width !== displayWidth || webcamCanvas.height !== displayHeight) {
            webcamCanvas.width  = displayWidth;
            webcamCanvas.height = displayHeight;
        }

        ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);

        // Draw bounding boxes if valid and not expired
        if (activeDetections.length > 0 && Date.now() < detectionsExpiryTime) {
            const scaleX = webcamCanvas.width  / webcamVideo.videoWidth;
            const scaleY = webcamCanvas.height / webcamVideo.videoHeight;

            activeDetections.forEach(det => {
                const [x1, y1, x2, y2] = det.box;
                const rx1 = x1 * scaleX;
                const ry1 = y1 * scaleY;
                const rw  = (x2 - x1) * scaleX;
                const rh  = (y2 - y1) * scaleY;

                const isYellow = det.board_type && det.board_type.includes('Yellow');
                const isGreen  = det.board_type && det.board_type.includes('Green');
                const boxColor  = isGreen ? '#22c55e' : isYellow ? '#f59e0b' : '#10d9a0';
                const glowColor = isGreen ? 'rgba(34,197,94,0.4)' : isYellow ? 'rgba(245,158,11,0.4)' : 'rgba(16,217,160,0.4)';

                // Glow then sharp box
                ctx.shadowColor  = glowColor;
                ctx.shadowBlur   = 10;
                ctx.strokeStyle  = boxColor;
                ctx.lineWidth    = 2.5;
                ctx.lineJoin     = 'round';
                ctx.strokeRect(rx1, ry1, rw, rh);
                ctx.shadowBlur   = 0;

                // Label
                const displayPlate = det.plate_formatted || det.plate_number || 'Plate';
                const unreadable   = !det.plate_number;
                const boardShort   = isGreen ? 'EV' : isYellow ? 'Comm' : 'Priv';
                const label        = unreadable ? `UNREADABLE (${boardShort})` : `${displayPlate} (${boardShort})`;

                ctx.font = 'bold 12px "Outfit", -apple-system, sans-serif';
                const textWidth   = ctx.measureText(label).width;
                const badgeH      = 22;
                const badgePad    = 8;

                ctx.fillStyle = unreadable ? '#f43f5e' : boxColor;
                ctx.fillRect(rx1 - 1, ry1 - badgeH - 4, textWidth + badgePad * 2, badgeH);

                ctx.fillStyle = '#000000';
                ctx.fillText(label, rx1 + badgePad, ry1 - 9);
            });
        }

        // Track FPS
        frameCount++;
        const now = performance.now();
        if (now - lastFrameTime >= 1000) {
            fpsValue = frameCount;
            frameCount = 0;
            lastFrameTime = now;
        }
    }

    animationFrameId = requestAnimationFrame(drawLocalStream);
}

function stopWebcam() {
    if (webcamInterval)     { clearInterval(webcamInterval);     webcamInterval = null; }
    if (fpsUpdateInterval)  { clearInterval(fpsUpdateInterval);  fpsUpdateInterval = null; }
    if (animationFrameId)   { cancelAnimationFrame(animationFrameId); animationFrameId = null; }
    if (webcamStream)       { webcamStream.getTracks().forEach(t => t.stop()); webcamStream = null; }

    webcamVideo.srcObject = null;
    webcamCanvas.getContext('2d').clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);

    webcamPlaceholder.style.display = 'flex';
    webcamControls.style.display = 'none';
    fpsCounter.style.display = 'none';
    isProcessingFrame = false;
    activeDetections = [];
    resetResultsView();
}

// Capture offscreen frame and post to backend
async function captureAndProcessWebcamFrame() {
    if (isProcessingFrame) return;
    if (webcamVideo.readyState < webcamVideo.HAVE_ENOUGH_DATA) return;

    isProcessingFrame = true;
    offscreenCanvas.width  = webcamVideo.videoWidth;
    offscreenCanvas.height = webcamVideo.videoHeight;
    offscreenCtx.drawImage(webcamVideo, 0, 0);

    const base64Frame = offscreenCanvas.toDataURL('image/jpeg', 0.82);
    const startTime = performance.now();

    try {
        const response = await fetch('/detect-frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Frame })
        });

        if (!response.ok) throw new Error('Frame API error: ' + response.statusText);

        const data = await response.json();
        const latency = Math.round(performance.now() - startTime);

        if (data.success && currentMode === 'webcam') {
            activeDetections      = data.detections;
            detectionsExpiryTime  = Date.now() + 1400;
            updateResultsPanel(data.detections, latency);
        }
    } catch (error) {
        console.error('Frame processing failed:', error);
    } finally {
        isProcessingFrame = false;
    }
}

// ============================================================
// Results Panel — multi-plate cards
// ============================================================
function updateResultsPanel(detections, latency, isHistoryView = false) {
    if (!detections || detections.length === 0) {
        if (currentMode === 'upload') resetResultsView();
        return;
    }

    noResults.style.display = 'none';
    activeResults.style.display = 'flex';

    statLatency.innerText = `${latency}ms`;
    statCount.innerText   = detections.length;

    platesContainer.innerHTML = detections.map((det, idx) => {
        const plateFormatted = det.plate_formatted || det.plate_number || '';
        const isUnreadable   = !det.plate_number;
        const isYellow       = det.board_type && det.board_type.includes('Yellow');
        const isGreen        = det.board_type && det.board_type.includes('Green');
        const boardColor     = isGreen ? '#22c55e' : isYellow ? '#f59e0b' : '#10d9a0';
        const plateColor     = isUnreadable ? '#f43f5e' : boardColor;
        const plateDisplay   = isUnreadable ? 'UNREADABLE' : plateFormatted;
        const stateName      = det.state_name || 'Unknown Registration';
        const boardType      = det.board_type || 'White Board (Private)';
        const confPct        = (det.confidence * 100).toFixed(1);
        const isValid        = det.is_valid_format;
        const seriesLabel    = det.plate_series === 'bh' ? 'BH Series' : 'Standard';
        const validBadge     = isValid
            ? `<span class="format-badge valid">✓ VALID FORMAT</span>`
            : `<span class="format-badge invalid">~ PARTIAL READ</span>`;
        const pulseClass     = isValid && !isUnreadable ? 'valid-pulse' : '';

        const rtoName = det.rto_name || '—';
        const hsrpBg         = isUnreadable ? '#f43f5e' : isGreen ? '#16a34a' : isYellow ? '#f59e0b' : '#ffffff';
        const hsrpText       = isUnreadable || isGreen ? '#ffffff' : '#000000';

        return `
            <div class="plate-card ${pulseClass}" style="--plate-color:${plateColor}; --plate-glow:${plateColor}40;">
                <div class="plate-card-header">
                    <div class="plate-card-left">
                        <span class="plate-index">#${idx + 1}</span>
                        ${!isUnreadable ? validBadge : ''}
                    </div>
                    ${!isUnreadable ? `
                    <button class="copy-plate-btn" title="Copy plate number"
                        onclick="copyPlate(this, '${plateFormatted}')">
                        <i class="fa-regular fa-copy"></i>
                    </button>` : ''}
                </div>
                <div class="hsrp-plate" style="--plate-bg: ${hsrpBg}; --plate-text: ${hsrpText}; box-shadow: 0 0 18px ${plateColor}45;">
                    <div class="hsrp-ind-strip">
                        <div class="hsrp-chakra"></div>
                        <span class="hsrp-ind-text">IND</span>
                    </div>
                    <div class="hsrp-number">${plateDisplay}</div>
                </div>
                ${det.crop_image ? `
                <div class="plate-crop-container" style="text-align:center; margin: 12px 0;">
                    <img src="${det.crop_image}" style="max-height: 80px; max-width: 100%; border-radius: 6px; border: 2px solid ${plateColor};" alt="Cropped Plate">
                </div>
                ` : ''}
                <div class="plate-details-grid">
                    <div class="plate-detail-item">
                        <span class="detail-label"><i class="fa-solid fa-map-location-dot"></i> State / UT</span>
                        <span class="detail-value">${isUnreadable ? '—' : stateName}</span>
                    </div>
                    <div class="plate-detail-item">
                        <span class="detail-label"><i class="fa-solid fa-tag"></i> Board Type</span>
                        <span class="detail-value" style="color:${boardColor};">${boardType}</span>
                    </div>
                    <div class="plate-detail-item">
                        <span class="detail-label"><i class="fa-solid fa-building-columns"></i> RTO Office</span>
                        <span class="detail-value rto-value">${isUnreadable ? '—' : rtoName}</span>
                    </div>
                    <div class="plate-detail-item">
                        <span class="detail-label"><i class="fa-solid fa-layer-group"></i> Series</span>
                        <span class="detail-value">${isUnreadable ? '—' : seriesLabel}</span>
                    </div>
                    <div class="plate-detail-item">
                        <span class="detail-label"><i class="fa-solid fa-fingerprint"></i> Raw OCR</span>
                        <span class="detail-value" style="font-size:0.78rem; font-family: var(--font-mono);">${det.raw_text || '—'}</span>
                    </div>
                    <div class="plate-detail-item">
                        <span class="detail-label"><i class="fa-solid fa-shield-halved"></i> HSRP</span>
                        <span class="detail-value" style="color:${det.is_hsrp ? '#10d9a0' : '#f59e0b'};">${det.is_hsrp ? 'Yes (Detected)' : 'No'}</span>
                    </div>
                </div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar-label">
                        <span><i class="fa-solid fa-percent"></i> Detection Confidence</span>
                        <span class="conf-value">${confPct}%</span>
                    </div>
                    <div class="conf-bar-track">
                        <div class="conf-bar-fill" style="width:${confPct}%; background:${plateColor};"></div>
                    </div>
                </div>
            </div>
        `;
    }).join('');

    // Log valid plates to history
    if (!isHistoryView) {
        detections.forEach(det => {
            if (det.plate_number && det.plate_number.length >= 4) {
                const confPct = (det.confidence * 100).toFixed(1) + '%';
                addToHistoryLog(
                    det.plate_number,
                    det.plate_formatted || det.plate_number,
                    det.state_name,
                    det.board_type,
                    confPct,
                    det
                );
            }
        });
    }
}

// ============================================================
// Copy Plate to Clipboard
// ============================================================
function copyPlate(btn, plateText) {
    if (!plateText) return;
    navigator.clipboard.writeText(plateText).then(() => {
        btn.classList.add('copied');
        btn.innerHTML = '<i class="fa-solid fa-check"></i>';
        showToast('success', 'Copied!', `"${plateText}" copied to clipboard.`, 2000);
        setTimeout(() => {
            btn.classList.remove('copied');
            btn.innerHTML = '<i class="fa-regular fa-copy"></i>';
        }, 2000);
    }).catch(() => {
        showToast('error', 'Copy Failed', 'Could not copy to clipboard. Try manually selecting the text.');
    });
}

// ============================================================
// History Log
// ============================================================
function addToHistoryLog(rawPlate, formattedPlate, state, board, confidence, fullDetection) {
    const cleanPlate = rawPlate.toUpperCase().replace(/\s/g, '');
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    // Prevent duplicate logs within 5 seconds
    const existingIdx = historyLog.findIndex(item => {
        const itemClean = item.rawPlate.replace(/\s/g, '');
        const timeDiff  = (now - item.timestamp) / 1000;
        return itemClean === cleanPlate && timeDiff < 5;
    });

    if (existingIdx !== -1) {
        // Update timestamp and time display for the existing entry
        historyLog[existingIdx].timestamp = now;
        historyLog[existingIdx].time = timeStr;
        if (fullDetection) historyLog[existingIdx].fullDetection = fullDetection;
        renderHistory();
        return;
    }

    historyLog.unshift({
        rawPlate: cleanPlate,
        plate: formattedPlate,
        state,
        board,
        confidence,
        time: timeStr,
        timestamp: now,
        fullDetection: fullDetection
    });

    if (historyLog.length > 20) historyLog.pop();
    renderHistory();
}

function renderHistory() {
    if (historyLog.length === 0) {
        historyList.innerHTML = `<div class="history-empty">No detections logged yet</div>`;
        return;
    }

    historyList.innerHTML = historyLog.map((item, idx) => {
        const isYellow  = item.board && item.board.includes('Yellow');
        const isGreen   = item.board && item.board.includes('Green');
        const borderClr = isGreen ? '#22c55e' : isYellow ? '#f59e0b' : '#7c3aed';
        const boardShort = isGreen ? 'EV' : isYellow ? 'Comm' : 'Priv';
        return `
            <div class="history-item" style="border-left-color:${borderClr}; cursor: pointer;" onclick="viewHistoryItem(${idx})" title="Click to view details">
                <div class="history-left">
                    <span class="history-plate">${item.plate}</span>
                    <span class="history-state" title="${item.state}">${item.state} • ${boardShort}</span>
                </div>
                <div class="history-right">
                    <span class="history-time">${item.time}</span>
                    <span class="history-conf">${item.confidence}</span>
                </div>
            </div>
        `;
    }).join('');
}

function viewHistoryItem(idx) {
    const item = historyLog[idx];
    if (item && item.fullDetection) {
        updateResultsPanel([item.fullDetection], '—', true);
        window.scrollTo({ top: 0, behavior: 'smooth' }); // Scroll up to see results
    }
}

function clearHistory() {
    historyLog = [];
    renderHistory();
    showToast('info', 'History Cleared', 'Detection log has been cleared.');
}

// ============================================================
// Export History as CSV
// ============================================================
async function exportHistoryAsCSV() {
    if (historyLog.length === 0) {
        showToast('warning', 'Nothing to Export', 'Detect some plates first before exporting.');
        return;
    }

    // Build payload matching backend ExportData model
    const payload = historyLog.map(item => ({
        plate_formatted:  item.plate,
        plate_number:     item.rawPlate,
        state_code:       item.state ? item.state.substring(0, 2) : '',
        state_name:       item.state || '',
        board_type:       item.board || '',
        plate_series:     'standard',
        confidence:       item.confidence,
        is_valid_format:  true,
        raw_text:         item.rawPlate,
        timestamp:        item.time,
    }));

    try {
        const response = await fetch('/export-csv', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ detections: payload })
        });

        if (!response.ok) throw new Error('Export failed: ' + response.statusText);

        const blob = await response.blob();
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href     = url;
        a.download = `swiftplate_history_${new Date().toISOString().slice(0,10)}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        showToast('success', 'CSV Exported', `${historyLog.length} records saved successfully.`);
    } catch (error) {
        console.error(error);
        showToast('error', 'Export Failed', error.message || 'Could not export history. Try again.');
    }
}

// ============================================================
// Toast Notification System
// ============================================================
const TOAST_ICONS = {
    success: 'fa-solid fa-circle-check',
    error:   'fa-solid fa-circle-xmark',
    info:    'fa-solid fa-circle-info',
    warning: 'fa-solid fa-triangle-exclamation',
};

function showToast(type = 'info', title = '', message = '', duration = 4000) {
    const container = document.getElementById('toast-container');
    const id        = `toast-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    const icon      = TOAST_ICONS[type] || TOAST_ICONS.info;

    const toast = document.createElement('div');
    toast.id = id;
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <i class="${icon} toast-icon"></i>
        <div class="toast-body">
            ${title ? `<div class="toast-title">${title}</div>` : ''}
            ${message ? `<div class="toast-msg">${message}</div>` : ''}
        </div>
        <div class="toast-progress" style="animation-duration: ${duration}ms;"></div>
    `;

    // Click to dismiss
    toast.addEventListener('click', () => dismissToast(toast));

    container.appendChild(toast);

    // Auto-dismiss
    setTimeout(() => dismissToast(toast), duration);
}

function dismissToast(toast) {
    if (!toast || !toast.parentNode) return;
    toast.classList.add('toast-hide');
    toast.addEventListener('animationend', () => toast.remove(), { once: true });
}
