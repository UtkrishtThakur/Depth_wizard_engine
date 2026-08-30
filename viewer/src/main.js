import { Viewer } from './viewer.js';

// Resolves to engine/processed via Vite's publicDir config
export const DEBUG_GEOMETRY = false;
const TERRAIN_ASSET = DEBUG_GEOMETRY ? '/terrain.glb' : '/textured_terrain.glb';

const timestamp = new Date().getTime();
const TERRAIN_URL = `${TERRAIN_ASSET}?t=${timestamp}`;

document.addEventListener('DOMContentLoaded', () => {
    console.log(`[Main] Requesting terrain from URL: ${TERRAIN_URL}`);
    const statusOverlay = document.getElementById('status-overlay');
    const statusText = document.getElementById('status-text');
    const hud = document.getElementById('hud');

    try {
        const viewer = new Viewer(TERRAIN_URL, {
            debugGeometry: DEBUG_GEOMETRY,
            onProgress: (percent) => {
                statusText.innerText = `Loading terrain... ${Math.round(percent)}%`;
            },
            onLoad: () => {
                statusText.innerText = 'Click to enter flythrough';
                
                statusOverlay.addEventListener('click', () => {
                    viewer.lockPointer();
                });
            },
            onError: (err) => {
                console.error("Terrain load error:", err);
                statusText.innerText = 'Failed to load terrain.glb';
            },
            onLock: () => {
                statusOverlay.classList.add('hidden');
                hud.style.display = 'block';
            },
            onUnlock: () => {
                statusOverlay.classList.remove('hidden');
                statusText.innerText = 'Click to resume flythrough';
                hud.style.display = 'none';
            }
        });
    } catch (e) {
        console.error("Viewer initialization failed", e);
        statusText.innerText = 'WebGL initialization failed.';
    }
});
