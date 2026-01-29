# Contributing to Neumann TypeScript SDK

Thank you for your interest in contributing to the Neumann TypeScript SDK.

## Getting Started

### Prerequisites

- Node.js 18.0.0 or later
- npm 9.0.0 or later
- TypeScript knowledge

### Development Setup

```bash
# Clone the repository
git clone https://github.com/neumann-db/neumann.git
cd neumann/neumann-ts

# Install dependencies
npm install

# Run tests
npm test

# Run tests in watch mode
npm run test:watch

# Build all formats
npm run build

# Type checking
npm run typecheck

# Linting
npm run lint
```

### Project Structure

```
neumann-ts/
├── src/                    # Source code
│   ├── index.ts           # Public API exports
│   ├── client.ts          # NeumannClient implementation
│   ├── client.test.ts     # Test suite
│   └── types/             # Type definitions
│       ├── value.ts       # Value types
│       ├── query-result.ts # Result types
│       ├── errors.ts      # Error classes
│       └── grpc-web.d.ts  # gRPC-Web declarations
├── dist/                   # Build output
│   ├── esm/               # ES modules
│   ├── cjs/               # CommonJS modules
│   └── types/             # Type declarations
├── package.json
├── tsconfig.json          # Base TypeScript config
├── tsconfig.esm.json      # ESM build config
├── tsconfig.cjs.json      # CJS build config
├── tsconfig.types.json    # Type declarations config
├── vitest.config.ts       # Test configuration
└── eslint.config.js       # Linting configuration
```

## How to Contribute

### Reporting Bugs

1. Check existing [issues](https://github.com/neumann-db/neumann/issues)
2. Use the bug report template
3. Include:
   - Node.js version (`node --version`)
   - SDK version (`npm list @neumann/client`)
   - Operating system
   - Steps to reproduce
   - Expected vs actual behavior

### Suggesting Features

1. Open an issue with the feature request template
2. Describe the use case and motivation
3. Be open to discussion about implementation

### Submitting Code

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Ensure all checks pass:
   ```bash
   npm run lint
   npm run typecheck
   npm test
   npm run build
   ```
5. Commit with a clear message
6. Push and open a pull request

## Code Standards

### Style

- Follow ESLint TypeScript recommended rules
- Maximum line length: 100 characters
- Use strict TypeScript types
- Prefer explicit types over inference for public APIs

### Testing

- Write tests for all new functionality
- Use Vitest with async/await patterns
- Mock external dependencies (gRPC modules)
- Test edge cases and error conditions
- Maintain 95%+ coverage

### TypeScript Guidelines

```typescript
// Use explicit return types for public functions
export function calculate(a: number, b: number): number {
  return a + b;
}

// Use discriminated unions for result types
type Result = { type: 'success'; data: string } | { type: 'error'; message: string };

// Prefer type guards for narrowing
function isSuccess(result: Result): result is { type: 'success'; data: string } {
  return result.type === 'success';
}

// Use readonly where appropriate
interface Config {
  readonly apiKey: string;
  readonly timeout: number;
}
```

### Documentation

- Add JSDoc comments to public classes and functions
- Include `@param`, `@returns`, and `@example` tags
- Update README.md for new features

### Commit Messages

Write clear, imperative commit messages:

```
Add streaming support for large result sets

- Implement executeStream() method
- Add async iterator for chunked results
- Include tests for streaming behavior
```

## Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure CI passes (all checks green)
4. Request review from maintainers
5. Address feedback promptly

## Building and Testing

```bash
# Full build (ESM + CJS + Types)
npm run build

# Individual builds
npm run build:esm
npm run build:cjs
npm run build:types

# Run tests with coverage
npm test -- --coverage

# Run specific test file
npm test -- src/client.test.ts

# Type check without emitting
npm run typecheck
```

## Release Process

Releases are handled by maintainers:

1. Update version in `package.json`
2. Update CHANGELOG.md
3. Create git tag (`git tag ts-v0.1.0`)
4. CI builds and publishes to npm

## Governance

### Maintainer Responsibilities

- Review and merge pull requests
- Triage issues and feature requests
- Maintain documentation and examples
- Manage releases and versioning

### Decision Making

- Technical decisions are made through GitHub issues and PRs
- Major changes require RFC discussion
- Breaking changes require deprecation notice in previous version

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
