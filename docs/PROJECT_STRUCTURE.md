# File Processing Optimization - Project Structure

## Overview
This document describes the organized project structure after cleanup and refactoring.

## Directory Structure

```
file-processing-optimization/
├── README.md                    # Main project documentation
├── setup.py                     # Package setup configuration
├── requirements.txt             # Python dependencies
├── requirements_web.txt         # Web-specific dependencies
├── pytest.ini                   # Test configuration
├── docker-compose.yml           # Docker composition
├── Dockerfile                   # Main Docker configuration
├── Dockerfile.cli               # CLI Docker configuration
├── main.py                      # Main CLI application
├── web_app.py                   # Web application entry point
│
├── config/                      # Configuration files
│   ├── config.json             # Default configuration
│   ├── test_config.json        # Test configuration
│   └── uzbek_config.json       # Uzbek-specific configuration
│
├── src/                         # Source code (Clean Architecture)
│   ├── __init__.py
│   ├── domain/                  # Core business logic
│   │   ├── __init__.py
│   │   ├── models.py           # Data models
│   │   ├── exceptions.py       # Domain exceptions
│   │   └── matching/           # Matching algorithms
│   │       ├── __init__.py
│   │       ├── base.py         # Base matching classes
│   │       ├── engine.py       # Matching engine
│   │       ├── exact_matcher.py
│   │       ├── fuzzy_matcher.py
│   │       ├── phonetic_matcher.py
│   │       ├── uzbek_normalizer.py
│   │       ├── blocking.py     # Blocking strategies
│   │       └── cache.py        # Caching mechanisms
│   │
│   ├── application/             # Application services
│   │   ├── __init__.py
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── file_service.py
│   │       ├── config_service.py
│   │       ├── batch_processor.py
│   │       ├── result_manager.py
│   │       ├── secure_file_service.py
│   │       └── cli_config_service.py
│   │
│   ├── infrastructure/          # External concerns
│   │   ├── __init__.py
│   │   ├── logging.py          # Structured logging
│   │   ├── metrics.py          # Prometheus metrics
│   │   ├── tracing.py          # Distributed tracing
│   │   ├── monitoring_integration.py
│   │   ├── monitoring_endpoints.py
│   │   ├── caching.py          # Caching infrastructure
│   │   ├── security.py         # Security utilities
│   │   ├── progress_tracker.py # Progress tracking
│   │   ├── parallel_processing.py
│   │   ├── memory_management.py
│   │   ├── gpu_acceleration.py
│   │   ├── redis_cache.py
│   │   ├── compressed_storage.py
│   │   ├── memory_mapped_files.py
│   │   ├── data_protection.py
│   │   ├── error_aggregation.py
│   │   ├── health_checks.py
│   │   └── message_queue.py
│   │
│   └── web/                     # Web-specific components
│       ├── __init__.py
│       ├── api_app.py          # API application
│       ├── middleware/
│       │   ├── __init__.py
│       │   └── auth_middleware.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── web_models.py
│       ├── routes/
│       │   ├── __init__.py
│       │   ├── main_routes.py
│       │   └── api_routes.py
│       └── services/
│           ├── __init__.py
│           ├── file_service.py
│           ├── job_service.py
│           ├── processing_service.py
│           └── websocket_progress_service.py
│
├── templates/                   # Web templates
│   ├── base.html
│   ├── index.html
│   ├── configure.html
│   ├── processing.html
│   ├── results.html
│   └── select_delimiter.html
│
├── tests/                       # Test suite
│   ├── fixtures/               # Test fixtures
│   ├── test_*.py              # Test files
│   └── ...
│
├── scripts/                     # Utility scripts
│   ├── cli.py                  # CLI interface
│   ├── cli_templates.py        # CLI templates
│   ├── cli_completion.bash     # Bash completion
│   └── generate_api_docs.py    # API documentation generator
│
├── docs/                        # Documentation
│   ├── api/                    # API documentation
│   ├── architecture/           # Architecture docs
│   ├── deployment/             # Deployment guides
│   ├── user-guide/            # User guides
│   ├── configuration-reference.md
│   ├── CONTRIBUTING.md
│   ├── FAQ.md
│   └── troubleshooting.md
│
├── deployment/                  # Deployment configurations
│   ├── kubernetes/             # K8s manifests
│   ├── grafana/               # Grafana dashboards
│   ├── prometheus/            # Prometheus config
│   └── scripts/               # Deployment scripts
│
├── data/                        # Data files
│   ├── 1/                     # Sample dataset 1
│   ├── 2/                     # Sample dataset 2
│   └── results/               # Processing results
│
├── logs/                        # Log files
│   ├── application.log
│   └── errors.log
│
├── uploads/                     # Temporary uploads (cleaned regularly)
│
└── .kiro/                      # Kiro IDE configuration
    └── specs/                  # Feature specifications
        └── file-processing-optimization/
            ├── requirements.md
            ├── design.md
            └── tasks.md
```

## Key Improvements

### 1. **Clean Architecture**
- Clear separation between domain, application, and infrastructure layers
- Domain logic is independent of external frameworks
- Infrastructure concerns are isolated

### 2. **Organized Configuration**
- All configuration files moved to `config/` directory
- Environment-specific configurations separated
- Clear naming conventions

### 3. **Consolidated Scripts**
- All CLI and utility scripts in `scripts/` directory
- Better organization of command-line tools

### 4. **Removed Unnecessary Files**
- Eliminated duplicate main files
- Removed temporary test files from root
- Cleaned up cache and temporary files
- Removed old documentation files

### 5. **Better Documentation Structure**
- Comprehensive documentation in `docs/` directory
- API documentation separated from user guides
- Architecture documentation for developers

## Entry Points

### Web Application
```bash
python web_app.py
```

### CLI Application
```bash
python main.py
# or
python scripts/cli.py
```

### API Server
```bash
python -m src.web.api_app
```

## Development Workflow

1. **Source Code**: All business logic in `src/` following clean architecture
2. **Configuration**: Environment-specific configs in `config/`
3. **Testing**: Comprehensive test suite in `tests/`
4. **Documentation**: Keep `docs/` updated with changes
5. **Deployment**: Use configurations in `deployment/`

## File Naming Conventions

- **Python files**: `snake_case.py`
- **Configuration files**: `descriptive_name.json`
- **Documentation**: `UPPERCASE.md` for important docs, `lowercase.md` for guides
- **Test files**: `test_*.py`
- **Scripts**: `descriptive_name.py`

This structure provides better maintainability, clearer separation of concerns, and easier navigation for developers.