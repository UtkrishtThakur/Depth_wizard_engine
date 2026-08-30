import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  // Expose the engine/processed directory directly so /terrain.glb resolves automatically
  // without copying files or duplicating assets during development.
  publicDir: resolve(__dirname, '../engine/processed'),
  server: {
    fs: {
      // Allow serving files from the parent directory
      allow: ['..']
    }
  }
});
