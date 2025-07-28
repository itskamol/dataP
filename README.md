# File Processing Optimization System

A comprehensive file processing and data matching system with optimized fuzzy matching for Uzbek text, built with clean architecture principles.

## Overview

This system transforms the existing monolithic file matching application into a modular, scalable, and maintainable architecture. It supports CSV and JSON file processing with various matching algorithms including exact matching, fuzzy matching, and phonetic matching optimized for Uzbek text.

## Architecture

The system follows clean architecture principles with clear separation of concerns:

```
src/
├── domain/                 # Core business logic and models
│   ├── models.py          # Data models and value objects
│   └── exceptions.py      # Domain-specific exceptions
├── application/           # Application services and use cases
│   └── services/
│       ├── file_service.py      # File processing operations
│       └── config_service.py    # Configuration management
└── infrastructure/       # External concerns and frameworks
    ├── logging.py        # Structured logging with correlation IDs
    └── caching.py        # Performance optimization caching
```

## Features

### Core Functionality
- **Multi-format Support**: CSV, JSON, and Excel file processing
- **Advanced Matching**: Exact, fuzzy, and phonetic matching algorithms
- **Uzbek Text Optimization**: Specialized normalization and matching for Uzbek language
- **Flexible Configuration**: JSON-based configuration with schema validation
- **Real-time Progress**: WebSocket-based progress tracking
- **Multiple Export Formats**: CSV, JSON, Excel output support

### Performance Optimizations
- **Blocking Strategies**: Reduce unnecessary comparisons
- **LRU Caching**: Cache similarity calculations and normalized text
- **Parallel Processing**: Multi-threaded and multi-process support
- **Memory Management**: Streaming processing for large datasets
- **GPU Acceleration**: Optional CUDA support for large-scale operations

### Quality Assurance
- **Comprehensive Logging**: Structured logging with correlation IDs
- **Error Handling**: Robust exception hierarchy with context
- **Data Validation**: Schema validation and integrity checks
- **Testing Framework**: Unit, integration, and performance tests
- **Type Safety**: Full type hints and mypy support

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Basic Installation
```bash
# Clone the repository
git clone <repository-url>
cd file-processing-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Development Installation
```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Web Interface Installation
```bash
# Install with web dependencies
pip install -e ".[web]"
```

## Configuration

### Basic Configuration
Create a `config.json` file in the project root:

```json
{
  "file1": {
    "path": "data/file1.csv",
    "type": "csv",
    "delimiter": ",",
    "encoding": "utf-8"
  },
  "file2": {
    "path": "data/file2.csv",
    "type": "csv",
    "delimiter": ",",
    "encoding": "utf-8"
  },
  "matching": {
    "mappings": [
      {
        "source_field": "name",
        "target_field": "name",
        "algorithm": "fuzzy",
        "weight": 1.0,
        "normalization": true,
        "case_sensitive": false
      }
    ],
    "algorithms": [
      {
        "name": "exact",
        "parameters": {},
        "enabled": true,
        "priority": 1
      },
      {
        "name": "fuzzy",
        "parameters": {
          "threshold": 80
        },
        "enabled": true,
        "priority": 2
      }
    ],
    "thresholds": {
      "minimum_confidence": 75.0
    },
    "matching_type": "one-to-one",
    "confidence_threshold": 75.0
  },
  "output": {
    "format": "csv",
    "path": "results/matched_results",
    "include_unmatched": true,
    "include_confidence_scores": true
  }
}
```

### Environment Variables
```bash
# Logging level
export LOG_LEVEL=INFO

# Maximum memory usage (MB)
export MAX_MEMORY_MB=2048

# Number of worker processes
export MAX_WORKERS=4

# Cache configuration
export CACHE_SIZE=50000
export CACHE_MEMORY_MB=200
```

## Usage

### Command Line Interface
```bash
# Process files with default configuration
file-processor --config config.json

# Process with custom output format
file-processor --config config.json --output-format json

# Enable verbose logging
file-processor --config config.json --verbose

# Show help
file-processor --help
```

### Web Interface
```bash
# Start web server
file-processor-web --port 5000

# Start with custom configuration
file-processor-web --config config.json --host 0.0.0.0 --port 8080
```

### Python API
```python
from src.application.services.file_service import FileProcessingService
from src.application.services.config_service import ConfigurationManager

# Load configuration
config_manager = ConfigurationManager('config.json')
config = config_manager.load_config()

# Initialize file service
file_service = FileProcessingService()

# Load datasets
dataset1 = file_service.load_file(config.file1.path)
dataset2 = file_service.load_file(config.file2.path)

# Process matching (implementation in next tasks)
# matching_engine = MatchingEngine(config.matching)
# results = matching_engine.find_matches(dataset1, dataset2)
```

## Development

### Project Structure
```
├── src/                    # Source code (Clean Architecture)
│   ├── domain/            # Core business logic and models
│   ├── application/       # Application services and use cases
│   ├── infrastructure/    # External frameworks and infrastructure
│   └── web/               # Web-specific components
├── tests/                 # Comprehensive test suite
├── docs/                  # Documentation and guides
├── config/                # Configuration files
├── scripts/               # CLI and utility scripts
├── templates/             # Web templates
├── deployment/            # Deployment configurations
├── data/                  # Sample data files
├── logs/                  # Application logs
└── uploads/               # Temporary file uploads
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed structure documentation.

### Code Quality
```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Type checking
mypy src/

# Code formatting
black src/ tests/

# Linting
flake8 src/ tests/
```

### Adding New Features
1. Define domain models in `src/domain/models.py`
2. Add business logic in `src/application/services/`
3. Implement infrastructure concerns in `src/infrastructure/`
4. Add comprehensive tests
5. Update documentation

## Performance Tuning

### Memory Optimization
- Adjust `memory_limit_mb` in configuration
- Enable streaming processing for large files
- Configure cache sizes appropriately
- Use blocking strategies to reduce comparisons

### CPU Optimization
- Set `max_workers` based on CPU cores
- Enable parallel processing
- Use appropriate matching algorithms
- Configure blocking parameters

### I/O Optimization
- Use SSD storage for temporary files
- Configure appropriate chunk sizes
- Enable compression for large outputs
- Use memory-mapped files for very large datasets

## Monitoring and Logging

### Log Files
- `logs/application.log` - General application logs
- `logs/errors.log` - Error-specific logs
- `logs/performance.log` - Performance metrics

### Metrics
- Processing speed (records/second)
- Memory usage
- Cache hit rates
- Match accuracy
- Error rates

### Health Checks
```bash
# Check system health
curl http://localhost:5000/health

# Get performance metrics
curl http://localhost:5000/metrics
```

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size in configuration
   - Enable streaming processing
   - Increase system memory or use swap

2. **Performance Issues**
   - Enable caching
   - Use blocking strategies
   - Optimize field mappings
   - Consider GPU acceleration

3. **File Format Errors**
   - Validate file encoding
   - Check delimiter settings
   - Verify file structure

4. **Matching Accuracy**
   - Adjust confidence thresholds
   - Enable text normalization
   - Fine-tune algorithm parameters

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with profiling
python -m cProfile -o profile.stats main.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks
5. Submit a pull request

### Development Guidelines
- Follow clean architecture principles
- Write comprehensive tests
- Use type hints
- Document public APIs
- Follow PEP 8 style guide

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide
- Contact the development team

## Roadmap

### Version 1.1
- GPU acceleration support
- Advanced phonetic matching
- Real-time processing API
- Enhanced web interface

### Version 1.2
- Machine learning integration
- Distributed processing
- Advanced analytics
- Cloud deployment support

### Version 2.0
- Microservices architecture
- Kubernetes deployment
- Advanced monitoring
- Multi-language support