import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { PointerLockControls } from 'three/examples/jsm/controls/PointerLockControls.js';

export class Viewer {
    constructor(terrainUrl, callbacks) {
        this.callbacks = callbacks;
        
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a1a);
        this.scene.fog = new THREE.FogExp2(0x1a1a1a, 0.002);

        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 10000);
        
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        document.body.appendChild(this.renderer.domElement);

        this.setupLighting();
        this.setupControls();
        this.loadTerrain(terrainUrl);

        window.addEventListener('resize', this.onWindowResize.bind(this));
        
        this.clock = new THREE.Clock();
        this.animate = this.animate.bind(this);
        this.animate();
    }

    setupLighting() {
        const ambient = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambient);

        const directional = new THREE.DirectionalLight(0xffffff, 1.2);
        directional.position.set(100, 200, 50);
        this.scene.add(directional);
    }

    setupControls() {
        this.controls = new PointerLockControls(this.camera, document.body);

        this.controls.addEventListener('lock', () => {
            if (this.callbacks.onLock) this.callbacks.onLock();
        });

        this.controls.addEventListener('unlock', () => {
            if (this.callbacks.onUnlock) this.callbacks.onUnlock();
        });

        this.move = {
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false
        };
        
        this.speedModifier = 1.0;

        document.addEventListener('keydown', (e) => this.onKeyDown(e));
        document.addEventListener('keyup', (e) => this.onKeyUp(e));
    }

    onKeyDown(event) {
        switch (event.code) {
            case 'ArrowUp':
            case 'KeyW': this.move.forward = true; break;
            case 'ArrowLeft':
            case 'KeyA': this.move.left = true; break;
            case 'ArrowDown':
            case 'KeyS': this.move.backward = true; break;
            case 'ArrowRight':
            case 'KeyD': this.move.right = true; break;
            case 'KeyE': this.move.up = true; break;
            case 'KeyQ': this.move.down = true; break;
            case 'ShiftLeft':
            case 'ShiftRight': this.speedModifier = 3.0; break;
            case 'ControlLeft':
            case 'ControlRight': this.speedModifier = 0.25; break;
        }
    }

    onKeyUp(event) {
        switch (event.code) {
            case 'ArrowUp':
            case 'KeyW': this.move.forward = false; break;
            case 'ArrowLeft':
            case 'KeyA': this.move.left = false; break;
            case 'ArrowDown':
            case 'KeyS': this.move.backward = false; break;
            case 'ArrowRight':
            case 'KeyD': this.move.right = false; break;
            case 'KeyE': this.move.up = false; break;
            case 'KeyQ': this.move.down = false; break;
            case 'ShiftLeft':
            case 'ShiftRight':
            case 'ControlLeft':
            case 'ControlRight': this.speedModifier = 1.0; break;
        }
    }

    loadTerrain(url) {
        const loader = new GLTFLoader();
        
        loader.load(
            url,
            (gltf) => {
                const model = gltf.scene;
                this.currentTerrain = model;
                this.scene.add(model);
                
                // --- DIAGNOSTIC LOGGING ---
                model.updateMatrixWorld(true);
                console.log("MODEL TRANSFORM:");
                console.log("position =", model.position);
                console.log("rotation =", model.rotation);
                console.log("scale =", model.scale);

                const DEBUG_GEOMETRY = this.callbacks.debugGeometry || false;

                model.traverse((child) => {
                    if (child.isMesh) {
                        console.log(`Mesh [${child.name || 'unnamed'}] TRANSFORM:`);
                        console.log("position =", child.position);
                        console.log("rotation =", child.rotation);
                        console.log("scale =", child.scale);
                        
                        if (DEBUG_GEOMETRY) {
                            child.material = new THREE.MeshBasicMaterial({
                                color: 0x00ff00,
                                wireframe: true,
                                side: THREE.DoubleSide
                            });
                        }
                    }
                });

                if (DEBUG_GEOMETRY) {
                    const axesHelper = new THREE.AxesHelper(50);
                    this.scene.add(axesHelper);
                    
                    const gridHelper = new THREE.GridHelper(200, 20);
                    this.scene.add(gridHelper);
                }

                this.frameTerrain(model);
                if (this.callbacks.onLoad) this.callbacks.onLoad();
            },
            (xhr) => {
                if (xhr.total > 0 && this.callbacks.onProgress) {
                    this.callbacks.onProgress((xhr.loaded / xhr.total) * 100);
                }
            },
            (error) => {
                if (this.callbacks.onError) this.callbacks.onError(error);
            }
        );
    }

    clearTerrain() {
        if (this.currentTerrain) {
            this.scene.remove(this.currentTerrain);
            this.currentTerrain.traverse((child) => {
                if (child.isMesh) {
                    if (child.geometry) child.geometry.dispose();
                    if (child.material) {
                        if (Array.isArray(child.material)) {
                            child.material.forEach(mat => mat.dispose());
                        } else {
                            child.material.dispose();
                        }
                    }
                }
            });
            this.currentTerrain = null;
        }
    }

    loadNewTerrain(url) {
        this.clearTerrain();
        this.loadTerrain(url);
    }

    frameTerrain(model) {
        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        
        console.log("MODEL WORLD BOUNDS:");
        console.log("X =", size.x);
        console.log("Y =", size.y);
        console.log("Z =", size.z);
        console.log("CENTER =", center);

        const maxDim = Math.max(size.x, size.y, size.z);
        
        // Place camera above the terrain using positive Y and point toward center
        this.camera.position.set(
            center.x, 
            center.y + maxDim * 1.5, 
            center.z + maxDim * 1.5
        );
        this.camera.lookAt(center.x, center.y, center.z);
        
        console.log("CAMERA:");
        console.log("position =", this.camera.position);
        console.log("rotation =", this.camera.rotation);

        this.camera.near = maxDim / 10000;
        this.camera.far = maxDim * 100;
        this.camera.updateProjectionMatrix();
        
        // Normalize movement speed relative to terrain radius
        this.baseSpeed = maxDim * 0.5; 
        
        // Adjust fog distance relative to terrain radius
        this.scene.fog.density = 1.5 / maxDim;

        const DEBUG_GEOMETRY = this.callbacks.debugGeometry || false;
        if (DEBUG_GEOMETRY) {
            const sphereGeo = new THREE.SphereGeometry(maxDim * 0.05, 16, 16);
            const sphereMat = new THREE.MeshBasicMaterial({ color: 0xff0000 });
            const sphere = new THREE.Mesh(sphereGeo, sphereMat);
            sphere.position.copy(center);
            this.scene.add(sphere);
        }
    }

    lockPointer() {
        this.controls.lock();
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    animate() {
        requestAnimationFrame(this.animate);

        const delta = this.clock.getDelta();

        if (this.controls.isLocked) {
            const actualSpeed = this.baseSpeed * this.speedModifier * delta;
            
            if (this.move.forward) this.controls.moveForward(actualSpeed);
            if (this.move.backward) this.controls.moveForward(-actualSpeed);
            if (this.move.right) this.controls.moveRight(actualSpeed);
            if (this.move.left) this.controls.moveRight(-actualSpeed);
            
            // PointerLockControls maintains y implicitly, so we adjust manually
            if (this.move.up) this.camera.position.y += actualSpeed;
            if (this.move.down) this.camera.position.y -= actualSpeed;
        }

        this.renderer.render(this.scene, this.camera);
    }
}
