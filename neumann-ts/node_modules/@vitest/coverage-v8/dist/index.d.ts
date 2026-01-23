import { V8CoverageProvider } from './provider.js';
import 'vitest/coverage';
import 'vitest';
import 'vitest/node';

declare const _default: {
    getProvider(): Promise<V8CoverageProvider>;
    startCoverage(): void;
    takeCoverage(): Promise<unknown>;
    stopCoverage(): void;
};

export { _default as default };
