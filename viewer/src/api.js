export const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Upload an image to start a new DepthWizard process.
 * @param {File} file 
 * @returns {Promise<Object>} The API response containing the process UUID
 */
export async function uploadImage(file) {
    const formData = new FormData();
    formData.append('image', file);

    const response = await fetch(`${API_BASE}/api/v1/process`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP Error ${response.status}`);
    }

    return await response.json();
}

/**
 * Get the current status of a process.
 * @param {string} processId 
 * @returns {Promise<Object>}
 */
export async function getProcess(processId) {
    const response = await fetch(`${API_BASE}/api/v1/process/${processId}`);
    if (!response.ok) {
        throw new Error(`HTTP Error ${response.status}`);
    }
    return await response.json();
}

/**
 * Get the list of artifacts for a process.
 * @param {string} processId 
 * @returns {Promise<Object>}
 */
export async function getArtifacts(processId) {
    const response = await fetch(`${API_BASE}/api/v1/process/${processId}/artifacts`);
    if (!response.ok) {
        throw new Error(`HTTP Error ${response.status}`);
    }
    return await response.json();
}

/**
 * Get the model URL for a specific process.
 * @param {string} processId 
 * @returns {string} The full URL to the model endpoint
 */
export function getModelUrl(processId) {
    return `${API_BASE}/api/v1/process/${processId}/model`;
}

/**
 * Subscribe to real-time process logs via SSE.
 * @param {string} processId 
 * @param {Object} callbacks
 * @param {function(Object)} callbacks.onLog - Called when a new log arrives
 * @param {function(Object)} callbacks.onComplete - Called when process completes successfully
 * @param {function(Object)} callbacks.onError - Called when process fails or SSE errors
 * @returns {EventSource} The SSE connection to allow for manual closing
 */
export function subscribeToLogs(processId, callbacks) {
    const eventSource = new EventSource(`${API_BASE}/api/v1/process/${processId}/logs/stream`);

    eventSource.addEventListener('log', (e) => {
        try {
            const data = JSON.parse(e.data);
            if (callbacks.onLog) callbacks.onLog(data);
        } catch (err) {
            console.error("Failed to parse log event", err);
        }
    });

    eventSource.addEventListener('complete', (e) => {
        try {
            const data = JSON.parse(e.data);
            if (callbacks.onComplete) callbacks.onComplete(data);
            eventSource.close();
        } catch (err) {
            console.error("Failed to parse complete event", err);
        }
    });

    eventSource.addEventListener('error', (e) => {
        try {
            // Note: SSE triggers a generic 'error' event on network disconnects
            // We check if there's actual data indicating a backend error vs a network drop
            if (e.data) {
                const data = JSON.parse(e.data);
                if (callbacks.onError) callbacks.onError(data);
            } else {
                console.warn("SSE connection error/drop.");
                // We don't automatically trigger onError for drops, since we might reconnect.
                // The main application should handle polling process status as a fallback.
            }
        } catch (err) {
            console.error("Failed to parse error event", err);
        }
    });

    eventSource.addEventListener('cancelled', (e) => {
        try {
            const data = JSON.parse(e.data);
            if (callbacks.onError) callbacks.onError({ message: "Process was cancelled", ...data });
            eventSource.close();
        } catch (err) {
            console.error("Failed to parse cancelled event", err);
        }
    });

    return eventSource;
}
