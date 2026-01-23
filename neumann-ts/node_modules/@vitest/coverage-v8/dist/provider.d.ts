import { BaseCoverageProvider } from 'vitest/coverage';
import { CoverageProvider, AfterSuiteRunMeta, ReportContext, ResolvedCoverageOptions } from 'vitest';
import { Vitest } from 'vitest/node';

interface TestExclude {
    new (opts: {
        cwd?: string | string[];
        include?: string | string[];
        exclude?: string | string[];
        extension?: string | string[];
        excludeNodeModules?: boolean;
        relativePath?: boolean;
    }): {
        shouldInstrument: (filePath: string) => boolean;
        glob: (cwd: string) => Promise<string[]>;
    };
}
type Options = ResolvedCoverageOptions<'v8'>;
type Filename = string;
type CoverageFilesByTransformMode = Record<AfterSuiteRunMeta['transformMode'], Filename[]>;
type ProjectName = NonNullable<AfterSuiteRunMeta['projectName']> | typeof DEFAULT_PROJECT;
declare const DEFAULT_PROJECT: unique symbol;
declare class V8CoverageProvider extends BaseCoverageProvider implements CoverageProvider {
    name: string;
    ctx: Vitest;
    options: Options;
    testExclude: InstanceType<TestExclude>;
    coverageFiles: Map<ProjectName, CoverageFilesByTransformMode>;
    coverageFilesDirectory: string;
    pendingPromises: Promise<void>[];
    initialize(ctx: Vitest): void;
    resolveOptions(): Options;
    clean(clean?: boolean): Promise<void>;
    onAfterSuiteRun({ coverage, transformMode, projectName }: AfterSuiteRunMeta): void;
    reportCoverage({ allTestsRun }?: ReportContext): Promise<void>;
    private getUntestedFiles;
    private getSources;
    private convertCoverage;
}

export { V8CoverageProvider };
