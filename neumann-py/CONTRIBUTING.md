# Contributing to Neumann Python SDK

Thank you for your interest in contributing to the Neumann Python SDK!

## Getting Started

### Prerequisites

- Python 3.10 or later
- Rust toolchain (for native bindings)
- pip and virtualenv

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Shadylukin/Neumann.git
cd Neumann/neumann-py

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,pandas,numpy]"

# Build native bindings (optional, requires Rust)
cd neumann-native
pip install maturin
maturin develop
cd ..
```

### Running Tests

```bash
# Run all tests
pytest tests -v

# Run with coverage
pytest tests --cov=neumann --cov-report=term-missing

# Run specific test file
pytest tests/test_client.py -v

# Run async tests only
pytest tests/test_aio_client.py -v
```

### Code Quality

```bash
# Linting
ruff check src tests

# Auto-fix linting issues
ruff check src tests --fix

# Type checking
mypy src/neumann --strict

# Format code (if using ruff format)
ruff format src tests
```

## Project Structure

```
neumann-py/
├── src/neumann/           # Main package
│   ├── __init__.py       # Public API exports
│   ├── client.py         # Sync client implementation
│   ├── aio/              # Async client
│   │   └── client.py
│   ├── types.py          # Data types and result handling
│   ├── errors.py         # Exception hierarchy
│   ├── config.py         # Configuration classes
│   ├── retry.py          # Retry logic
│   ├── transaction.py    # Transaction support
│   ├── integrations/     # Pandas/NumPy integrations
│   └── proto/            # Generated gRPC stubs
├── neumann-native/        # Rust/PyO3 bindings
│   ├── Cargo.toml
│   └── src/lib.rs
├── tests/                 # Test suite
└── pyproject.toml        # Package configuration
```

## How to Contribute

### Reporting Bugs

1. Check existing [issues](https://github.com/Shadylukin/Neumann/issues)
2. Use the bug report template
3. Include:
   - Python version (`python --version`)
   - SDK version (`pip show neumann-db`)
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
   ruff check src tests
   mypy src/neumann --strict
   pytest tests -v
   ```
5. Commit with a clear message
6. Push and open a pull request

## Code Standards

### Style

- Follow PEP 8 with Ruff for enforcement
- Maximum line length: 100 characters
- Use type hints for all public functions
- No emojis in code or commit messages

### Testing

- Write tests for all new functionality
- Use pytest with async support
- Mock external dependencies (gRPC, native bindings)
- Test edge cases and error conditions

### Documentation

- Add docstrings to public classes and functions
- Use Google-style docstrings
- Update README.md for new features
- Include examples in docstrings

### Commit Messages

Write clear, imperative commit messages:

```
Add streaming support for large result sets

- Implement execute_stream() method
- Add async iterator for chunked results
- Include tests for streaming behavior
```

## Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure CI passes (all checks green)
4. Request review from maintainers
5. Address feedback promptly

## Regenerating Proto Stubs

If the proto files change:

```bash
cd neumann-py
python -m grpc_tools.protoc \
    -I../neumann_server/proto \
    --python_out=src/neumann/proto \
    --grpc_python_out=src/neumann/proto \
    ../neumann_server/proto/neumann.proto
```

## Release Process

Releases are handled by maintainers:

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag
4. CI builds and publishes to PyPI

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
