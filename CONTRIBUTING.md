# Contributing to MCP Crawl4AI RAG

Thank you for your interest in contributing to the MCP Crawl4AI RAG project! This document provides guidelines for contributing.

## 🤝 How to Contribute

### Reporting Issues
- Use the GitHub issue tracker to report bugs
- Include detailed information about the problem
- Provide steps to reproduce the issue
- Include your environment details (OS, Python version, database provider)

### Submitting Changes
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 🧪 Development Setup

### Prerequisites
- Python 3.12+
- uv package manager
- Docker (for testing database providers)

### Setup
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/mcp-crawl4ai-rag.git
cd mcp-crawl4ai-rag

# Install development dependencies
uv pip install -e ".[all]"
crawl4ai-setup

# Run tests
pytest tests/
```

## 📋 Code Standards

### Python Code Style
- Follow PEP 8
- Use type hints
- Add docstrings to public functions
- Keep functions focused and small

### Database Providers
When adding new database providers:
1. Implement the `VectorDatabaseProvider` interface
2. Add configuration to `src/database/config.py`
3. Register in `src/database/factory.py`
4. Add tests in `tests/test_database_providers.py`
5. Update documentation

### Testing
- Write tests for all new functionality
- Ensure compatibility across all database providers
- Test both success and error cases
- Include integration tests where appropriate

## 📖 Documentation

- Update README.md for new features
- Add docstrings to new functions
- Update .env.example for new configuration options
- Create guides for complex features

## 🐛 Bug Reports

When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, database provider)
- Relevant logs or error messages

## 💡 Feature Requests

- Check existing issues before creating new ones
- Describe the use case clearly
- Explain how it fits with existing functionality
- Consider backward compatibility

## ✅ Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
- [ ] PR description clearly explains changes

## 🔧 Database Provider Guidelines

When implementing new database providers:

1. **Interface Compliance**: Implement all methods from `VectorDatabaseProvider`
2. **Error Handling**: Gracefully handle connection errors and timeouts
3. **Performance**: Implement efficient batch operations
4. **Testing**: Add comprehensive tests with real database instances
5. **Documentation**: Include setup instructions and configuration examples

## 📦 Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release branch
4. Test across all database providers
5. Create GitHub release with detailed notes

## 🤔 Questions?

Feel free to open an issue for discussion or reach out to the maintainers.

Thank you for contributing! 🎉