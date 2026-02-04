// SPDX-License-Identifier: MIT
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['src/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['src/**/*.ts'],
      exclude: ['src/**/*.test.ts', 'src/**/*.d.ts', 'src/index.ts', 'src/types/index.ts', 'src/grpc.ts', 'src/generated/**', 'src/services/index.ts'],
      thresholds: {
        lines: 95,
        functions: 95,
        // Note: vitest v4 (CI) reports ~94% vs vitest v1 (local) ~96%
        // Uncovered branches are internal catch blocks in client.ts
        branches: 94,
        statements: 95,
      },
    },
  },
});
