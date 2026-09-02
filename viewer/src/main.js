import { Viewer } from './viewer.js';
import * as api from './api.js';

let viewerInstance = null;
let currentProcessId = null;
let activeEventSource = null;
let selectedFile = null;

const STATES = {
    IDLE: 'IDLE',
    UPLOADED: 'UPLOADED',
    QUEUED: 'QUEUED',
    PROCESSING: 'PROCESSING',
    COMPLETED: 'COMPLETED',
    FAILED: 'FAILED'
};

let currentState = STATES.IDLE;

document.addEventListener('DOMContentLoaded', () => {
    const ui = {
        controlPanel: document.getElementById('control-panel'),
        statusOverlay: document.getElementById('status-overlay'),
        statusText: document.getElementById('status-text'),
        hud: document.getElementById('hud'),
        
        imageInput: document.getElementById('image-input'),
        selectBtn: document.getElementById('select-btn'),
        fileInfo: document.getElementById('file-info'),
        imagePreview: document.getElementById('image-preview'),
        filename: document.getElementById('filename'),
        generateBtn: document.getElementById('generate-btn'),
        
        processSection: document.getElementById('process-section'),
        processId: document.getElementById('process-id'),
        processStatus: document.getElementById('process-status'),
        logMessages: document.getElementById('log-messages'),
        errorMessage: document.getElementById('error-message'),
        newProcessBtn: document.getElementById('new-process-btn'),
        
        uploadSection: document.getElementById('upload-section')
    };

    function setState(newState) {
        currentState = newState;
        updateUI();
    }

    function updateUI() {
        // Hide errors by default
        ui.errorMessage.classList.add('hidden');
        
        switch (currentState) {
            case STATES.IDLE:
                ui.uploadSection.classList.remove('hidden');
                ui.processSection.classList.add('hidden');
                ui.fileInfo.classList.add('hidden');
                ui.generateBtn.disabled = true;
                break;
                
            case STATES.UPLOADED:
                ui.fileInfo.classList.remove('hidden');
                ui.generateBtn.disabled = false;
                ui.generateBtn.textContent = 'Generate';
                break;
                
            case STATES.QUEUED:
            case STATES.PROCESSING:
                ui.uploadSection.classList.add('hidden');
                ui.processSection.classList.remove('hidden');
                ui.processId.textContent = currentProcessId;
                ui.processStatus.textContent = currentState;
                ui.processStatus.className = `badge ${currentState.toLowerCase()}`;
                ui.newProcessBtn.classList.add('hidden');
                break;
                
            case STATES.COMPLETED:
                ui.processStatus.textContent = 'COMPLETED';
                ui.processStatus.className = 'badge completed';
                ui.newProcessBtn.classList.remove('hidden');
                ui.controlPanel.classList.add('hidden');
                ui.statusOverlay.classList.remove('hidden');
                ui.statusText.innerText = 'Click to enter flythrough';
                break;
                
            case STATES.FAILED:
                ui.processStatus.textContent = 'FAILED';
                ui.processStatus.className = 'badge failed';
                ui.newProcessBtn.classList.remove('hidden');
                break;
        }
    }

    function addLog(msg, type = 'info') {
        const div = document.createElement('div');
        div.className = `log-entry ${type}`;
        
        const time = new Date().toLocaleTimeString('en-US', { hour12: false });
        
        div.innerHTML = `
            <span class="log-time">${time}</span>
            <span class="log-msg">${msg}</span>
        `;
        ui.logMessages.appendChild(div);
        
        // Auto-scroll to bottom
        const container = document.getElementById('log-container');
        container.scrollTop = container.scrollHeight;
    }

    function showError(msg) {
        ui.errorMessage.textContent = msg;
        ui.errorMessage.classList.remove('hidden');
    }

    // Event Listeners
    ui.selectBtn.addEventListener('click', () => {
        ui.imageInput.click();
    });

    ui.imageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        selectedFile = file;
        ui.filename.textContent = file.name;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            ui.imagePreview.src = e.target.result;
            setState(STATES.UPLOADED);
        };
        reader.readAsDataURL(file);
    });

    ui.generateBtn.addEventListener('click', async () => {
        if (!selectedFile) return;
        
        try {
            ui.generateBtn.disabled = true;
            ui.generateBtn.textContent = 'Uploading...';
            
            const result = await api.uploadImage(selectedFile);
            currentProcessId = result.process.id;
            
            ui.logMessages.innerHTML = ''; // Clear logs
            addLog(`Process created: ${currentProcessId}`);
            
            setState(STATES.QUEUED);
            startPolling();
            
        } catch (err) {
            showError(err.message);
            ui.generateBtn.disabled = false;
            ui.generateBtn.textContent = 'Generate';
        }
    });

    ui.newProcessBtn.addEventListener('click', () => {
        selectedFile = null;
        currentProcessId = null;
        ui.imageInput.value = '';
        ui.controlPanel.classList.remove('hidden');
        ui.statusOverlay.classList.add('hidden');
        
        // If we have a viewer, show it behind the UI
        if (viewerInstance) {
            viewerInstance.clearTerrain();
        }
        
        setState(STATES.IDLE);
    });
    
    // Status Overlay (Enter flythrough)
    ui.statusOverlay.addEventListener('click', () => {
        if (currentState === STATES.COMPLETED && viewerInstance) {
            viewerInstance.lockPointer();
        }
    });

    function loadModel() {
        const url = api.getModelUrl(currentProcessId);
        
        if (viewerInstance) {
            viewerInstance.loadNewTerrain(url);
        } else {
            initViewer(url);
        }
    }

    function initViewer(url) {
        ui.statusOverlay.classList.remove('hidden');
        ui.statusText.innerText = 'Loading terrain...';
        
        try {
            viewerInstance = new Viewer(url, {
                debugGeometry: false,
                onProgress: (percent) => {
                    ui.statusText.innerText = `Loading terrain... ${Math.round(percent)}%`;
                },
                onLoad: () => {
                    ui.statusText.innerText = 'Click to enter flythrough';
                },
                onError: (err) => {
                    console.error("Terrain load error:", err);
                    ui.statusText.innerText = 'Failed to load generated model.';
                },
                onLock: () => {
                    ui.statusOverlay.classList.add('hidden');
                    ui.hud.style.display = 'block';
                },
                onUnlock: () => {
                    ui.statusOverlay.classList.remove('hidden');
                    ui.statusText.innerText = 'Click to resume flythrough';
                    ui.hud.style.display = 'none';
                    // Also bring back the control panel so user can hit Generate Another
                    ui.controlPanel.classList.remove('hidden');
                }
            });
        } catch (e) {
            console.error("Viewer initialization failed", e);
            ui.statusText.innerText = 'WebGL initialization failed.';
        }
    }

    function startPolling() {
        if (activeEventSource) {
            activeEventSource.close();
        }

        activeEventSource = api.subscribeToLogs(currentProcessId, {
            onLog: (data) => {
                let msg = data.message;
                if (data.progress !== null && data.progress !== undefined) {
                    msg += ` (${data.progress}%)`;
                }
                
                let type = 'info';
                if (data.level === 'ERROR') type = 'error';
                else if (data.level === 'WARNING') type = 'warning';
                
                addLog(msg, type);
                
                if (currentState === STATES.QUEUED && data.stage) {
                    setState(STATES.PROCESSING);
                }
            },
            onComplete: () => {
                setState(STATES.COMPLETED);
                addLog("Generation complete. Loading model...", "info");
                loadModel();
            },
            onError: (err) => {
                if (err && err.message === "Process was cancelled") {
                    addLog("Process was cancelled", "warning");
                    setState(STATES.FAILED);
                } else if (err && err.error) {
                    addLog(`Error: ${err.error}`, "error");
                    setState(STATES.FAILED);
                } else {
                    // Just a network drop, fallback to polling status
                }
            }
        });
        
        // Also start a periodic status check just in case SSE fails silently
        const statusInterval = setInterval(async () => {
            if (currentState === STATES.COMPLETED || currentState === STATES.FAILED) {
                clearInterval(statusInterval);
                return;
            }
            
            try {
                const process = await api.getProcess(currentProcessId);
                
                if (process.status === 'COMPLETED' && currentState !== STATES.COMPLETED) {
                    setState(STATES.COMPLETED);
                    addLog("Generation complete (polled). Loading model...", "info");
                    loadModel();
                    clearInterval(statusInterval);
                    if (activeEventSource) activeEventSource.close();
                } else if (process.status === 'FAILED' && currentState !== STATES.FAILED) {
                    setState(STATES.FAILED);
                    addLog(`Process failed: ${process.error_message}`, "error");
                    showError(process.error_message || "Process failed");
                    clearInterval(statusInterval);
                    if (activeEventSource) activeEventSource.close();
                }
            } catch (err) {
                console.warn("Failed to poll status", err);
            }
        }, 3000);
    }
});
