# Unit Tests Directory

This directory is prepared for unit tests that test individual components in isolation.

## Purpose

Unit tests focus on testing individual functions, classes, and methods in isolation from their dependencies. They should be fast, reliable, and test specific functionality.

## Test Structure

### Planned Test Files

```
tests/unit/
├── test_config/
│   ├── test_manager.py          # ConfigManager unit tests
│   └── test_models.py           # Configuration model tests
├── test_utils/
│   ├── test_logging_utils.py    # Logging utility tests
│   └── test_utils.py            # General utility tests
├── test_agents/
│   ├── test_schemas.py          # Schema validation tests
│   ├── test_memory.py           # Memory manager tests
│   ├── test_nodes.py            # Individual node tests
│   └── test_tools.py            # Agent tool tests
└── test_database/               # Database adapter tests (when implemented)
```

## Testing Guidelines

### What to Test

- **Individual Functions** - Test each function with various inputs
- **Class Methods** - Test class behavior in isolation
- **Edge Cases** - Test boundary conditions and error cases
- **Data Validation** - Test Pydantic model validation
- **Configuration Loading** - Test config parsing and validation

### Mocking Strategy

- **External Dependencies** - Mock LLM calls, database connections
- **File System** - Mock file operations for config loading
- **Environment Variables** - Mock environment variable access
- **Network Calls** - Mock any external API calls

### Example Test Structure

```python
import pytest
from unittest.mock import Mock, patch
from src.agents.schemas import Router

class TestRouter:
    def test_valid_classification(self):
        """Test router with valid classification."""
        router = Router(
            classification="echo_respond",
            reasoning="Simple greeting message"
        )
        assert router.classification == "echo_respond"
        assert len(router.reasoning) >= 10

    def test_invalid_classification(self):
        """Test router with invalid classification."""
        with pytest.raises(ValueError):
            Router(
                classification="invalid_type",
                reasoning="This should fail"
            )
```

## Running Unit Tests

```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run specific test file
python -m pytest tests/unit/test_agents/test_schemas.py -v

# Run with coverage
python -m pytest tests/unit/ --cov=src --cov-report=html
```

## Benefits of Unit Testing

- **Fast Feedback** - Quick to run and identify issues
- **Regression Prevention** - Catch breaking changes early
- **Documentation** - Tests serve as usage examples
- **Refactoring Safety** - Confidence when changing code
- **Bug Isolation** - Pinpoint exact location of issues
