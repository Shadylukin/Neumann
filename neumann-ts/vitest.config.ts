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
        // vitest v4 reports 94.43% due to strict branch counting on optional fields
        // Improved from 85.98% with extensive edge case tests
        branches: 94,
        statements: 95,
      },
    },
  },
});
