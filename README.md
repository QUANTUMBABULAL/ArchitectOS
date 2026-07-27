# AI Research Operator

A production-quality orchestration system for managing multiple AI systems through browser automation.

## Overview

The AI Research Operator is NOT a chatbot. It's a sophisticated system designed to orchestrate multiple AI reasoning engines (powered by Ollama/Qwen) through Playwright-based browser automation. The system coordinates complex workflows across multiple agents with consensus mechanisms and intelligent task planning.

## Architecture

### Core Modules

- **brain/**: Reasoning and decision-making logic powered by Ollama
- **browser/**: Playwright-based browser automation wrapper
- **orchestrator/**: Workflow coordination and management
- **planner/**: Task planning and execution strategies
- **consensus/**: Multi-agent agreement mechanisms
- **memory/**: State and knowledge storage
- **api/**: REST API interfaces
- **workers/**: Background task processing
- **prompts/**: Prompt template management
- **utils/**: Shared utilities and helpers

### Foundation Components

- **config.py**: Pydantic-based configuration management
- **constants.py**: Application-wide constants
- **exceptions.py**: Custom exception hierarchy
- **logger.py**: Structured logging system

## Project Structure

```
AI-Research-Operator/
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py              # Configuration management
│   ├── constants.py           # Application constants
│   ├── exceptions.py          # Custom exceptions
│   ├── logger.py              # Logging system
│   ├── api/                   # API interfaces
│   ├── brain/                 # Reasoning engine
│   ├── browser/               # Browser automation
│   ├── consensus/             # Consensus mechanisms
│   ├── memory/                # State management
│   ├── orchestrator/          # Workflow coordination
│   ├── planner/               # Task planning
│   ├── prompts/               # Prompt management
│   ├── utils/                 # Utilities
│   │   ├── __init__.py
│   │   ├── decorators.py      # Reusable decorators
│   │   ├── types.py           # Type definitions
│   │   ├── validators.py      # Input validation
│   │   └── main.py
│   └── workers/               # Background tasks
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## Installation

### Prerequisites

- Python 3.9+
- Virtual environment (venv, conda, etc.)

### Setup

1. **Clone or create the project**:
   ```bash
   cd AI-Research-Operator
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Create required directories**:
   ```bash
   python -m src
   ```

## Configuration

Configuration is managed through `src/config.py` using Pydantic. Settings can be provided through:

1. **Environment variables** (highest priority)
2. **.env file** in project root
3. **Default values** in configuration class

### Key Settings

```
ENVIRONMENT=development|staging|production
DEBUG=true|false
API_HOST=0.0.0.0
API_PORT=8000
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen:4b
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR|CRITICAL
```

See `.env.example` for all available settings.

## Running the System

### Initialize System

```bash
python -m src
```

This validates configuration, creates required directories, and initializes logging.

### Run in Python

```python
from src.config import get_settings
from src.logger import get_logger, configure_logging
import logging

# Initialize logging
configure_logging(level=logging.INFO)

# Get settings
settings = get_settings()
print(f"API running on {settings.api_host}:{settings.api_port}")

# Get logger
logger = get_logger(__name__)
logger.info("System ready")
```

## Development Guidelines

### Code Style

- **Type hints**: All functions and variables must have type hints
- **Docstrings**: All modules, classes, and functions must have detailed docstrings
- **SOLID Principles**: Follow Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion
- **No placeholders**: Every line of code serves a purpose
- **No TODOs**: Complete implementations only

### Exception Handling

Use custom exceptions from `src.exceptions`:

```python
from src.exceptions import ValidationError, BrainError

try:
    validate_input(data)
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
except BrainError as e:
    logger.error(f"Brain operation failed: {e}")
```

### Logging

Use the logging system throughout:

```python
from src.logger import get_logger

logger = get_logger(__name__)
logger.info("Operation started")
logger.debug("Detailed information")
logger.warning("Potential issue")
logger.error("Error occurred")
```

### Decorators

Reusable decorators for common patterns:

```python
from src.utils.decorators import logged, timed, retry, cache

@logged
@timed
@retry(max_attempts=3)
def my_operation():
    """Automatically logged, timed, and retried."""
    pass

@cache(ttl=60)
def expensive_computation():
    """Result cached for 60 seconds."""
    pass
```

### Validation

Use built-in validators:

```python
from src.utils.validators import (
    validate_string,
    validate_integer,
    validate_list,
)

username = validate_string(value, min_length=3, max_length=50)
port = validate_integer(8000, min_value=1, max_value=65535)
items = validate_list(value, min_length=1, element_type=str)
```

### Type Definitions

Use provided type aliases:

```python
from src.utils.types import (
    JSONValue,
    AgentId,
    TaskId,
    Callback,
)

def process(data: JSONValue) -> JSONValue:
    """Process JSON-compatible data."""
    pass

def register_agent(agent_id: AgentId, callback: Callback) -> None:
    """Register an agent with callback."""
    pass
```

## Components Overview

### Configuration System

```python
from src.config import get_settings

settings = get_settings()
settings.create_directories()  # Create required directories
print(settings.api_host)
```

### Logging System

```python
from src.logger import configure_logging, get_logger
import logging

configure_logging(level=logging.DEBUG)
logger = get_logger(__name__)
logger.info("Message")
```

### Exception Hierarchy

```
AIResearchOperatorError (Base)
├── ConfigurationError
├── ValidationError
├── BrowserError
├── OrchestratorError
├── MemoryError
├── PlannerError
├── BrainError
├── ConsensusError
└── WorkerError
```

### Constants

All application constants are defined in `src/constants.py`:

```python
from src.constants import (
    DEFAULT_TIMEOUT,
    BROWSER_HEADLESS,
    CONSENSUS_THRESHOLD,
    MAX_RETRIES,
)
```

## Testing

Run tests to verify the foundation:

```bash
python -c "from src.config import get_settings; print(get_settings())"
python -c "from src.logger import get_logger; logger = get_logger('test'); logger.info('Test')"
python -c "from src.exceptions import ValidationError; raise ValidationError('test')"
```

## Next Steps

Once the foundation is established, implement:

1. **Brain Module**: LLM integration with Ollama
2. **Browser Module**: Playwright automation wrapper
3. **Orchestrator Module**: Workflow coordination
4. **Planner Module**: Task decomposition and planning
5. **Consensus Module**: Multi-agent agreement
6. **Memory Module**: State and knowledge storage
7. **API Module**: REST endpoints
8. **Workers Module**: Background task processing

## Performance Considerations

- **Logging**: DEBUG level has performance impact; use INFO in production
- **Caching**: Use `@cache` decorator for expensive operations
- **Retries**: Exponential backoff prevents thundering herd
- **Timeouts**: All operations have configurable timeouts
- **Memory**: Implement cleanup strategies for long-running systems

## Security Considerations

- All configuration sensitive to environment
- No secrets in code or version control
- Use `.env` file for local secrets
- Validate all inputs
- Log errors without exposing sensitive data
- Use type hints for runtime validation

## Troubleshooting

### Import Errors

Ensure Python path includes project root:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Configuration Not Loading

Check `.env` file exists and is readable:
```bash
ls -la .env
```

### Logging Not Appearing

Ensure `configure_logging()` is called before `get_logger()`:
```python
from src.logger import configure_logging, get_logger
import logging

configure_logging(level=logging.DEBUG)
logger = get_logger(__name__)
```

## Contributing

- Follow SOLID principles
- Add type hints to all code
- Write comprehensive docstrings
- Use custom exceptions
- Add logging for debugging
- No placeholder code
- No TODO comments

## License

MIT License

## Support

For issues or questions, check:
1. Configuration (.env settings)
2. Logging output (check log files)
3. Exception messages
4. Documentation in module docstrings
