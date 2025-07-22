# Git Workflow Guide - Milestone 1 Customer Churn Detection

## Overview

This document outlines the git workflow and commit organization strategy for Milestone 1 development of the Customer Churn Detection MVP.

## Quick Start

### Setup Development Environment
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg

# Verify setup
pre-commit --version
git config --get commit.template  # Should show .gitmessage
```

### Daily Development Workflow
```bash
# 1. Start feature work
git checkout -b feature/churn-detection-core

# 2. Make changes and commit using template
git add <files>
git commit  # Opens template editor

# 3. Push feature branch
git push -u origin feature/churn-detection-core

# 4. Create pull request when ready
gh pr create --title "feat(churn): implement core detection system"
```

## Commit Organization Patterns

### Pattern 1: Feature + Tests + Docs (Recommended)
```bash
feat(churn): add customer behavior analyzer with tests

- Implement customer behavior pattern detection algorithm
- Add comprehensive unit tests with 95% coverage  
- Include API documentation and usage examples
- Update core module imports

Files:
- src/tracer/core/customer_analyzer.py
- tests/unit/test_customer_analyzer.py
- docs/core/customer_analyzer.md
- src/tracer/core/__init__.py
```

### Pattern 2: API Development
```bash
feat(api): add churn prediction endpoints

- Create REST endpoints for churn score calculation
- Add request/response validation schemas
- Include integration tests for all endpoints
- Update API documentation

Files:
- src/tracer/api/churn_routes.py
- src/tracer/api/validators/churn_schemas.py
- tests/integration/test_churn_api.py
- docs/api/churn_endpoints.md
```

### Pattern 3: Bug Fix with Tests
```bash
fix(churn): resolve edge case in scoring algorithm

- Handle customers with insufficient historical data
- Add validation for minimum data requirements
- Include regression tests for edge cases
- Update error handling documentation

Files:
- src/tracer/core/churn_scorer.py
- tests/unit/test_churn_scorer_edge_cases.py
- docs/troubleshooting/data_requirements.md
```

## Commit Message Format

### Structure
```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

### Types for Milestone 1
- `feat`: New churn detection features
- `fix`: Bug fixes in prediction logic
- `refactor`: Code optimization without behavior change
- `test`: Adding or updating tests
- `docs`: Documentation changes
- `chore`: Dependency updates, build configuration
- `perf`: Performance improvements

### Scopes for Milestone 1
- `churn`: Core churn detection functionality
- `customer`: Customer data management
- `prediction`: ML prediction services
- `pipeline`: Data processing workflows
- `monitoring`: System monitoring and alerts
- `api`: API endpoints and routing
- `core`: Core business logic
- `data`: Data models and validation

### Subject Line Guidelines
- Use imperative mood ("add" not "added")
- Keep under 50 characters
- Don't capitalize first letter
- Don't end with period
- Be specific about what changed

### Examples
```bash
# Good
feat(churn): add real-time customer behavior tracking
fix(prediction): handle missing customer attributes gracefully  
test(pipeline): increase data validation test coverage
docs(api): add churn prediction endpoint examples

# Bad
feat: added some stuff
fix: bug fix
Update code
FIX: Fixed the thing that was broken
```

## Branch Management

### Branch Naming Convention
```bash
# Features
feature/churn-detection-core
feature/customer-data-pipeline  
feature/prediction-api
feature/monitoring-dashboard

# Bug fixes
fix/customer-data-validation
fix/churn-score-calculation
fix/api-error-handling

# Performance improvements
perf/data-processing-optimization
perf/api-response-caching

# Documentation
docs/api-documentation
docs/deployment-guide
```

### Branch Lifecycle
```bash
# 1. Create feature branch from master
git checkout master
git pull origin master
git checkout -b feature/churn-detection-core

# 2. Development with atomic commits
git add src/tracer/core/analyzer.py tests/unit/test_analyzer.py
git commit -m "feat(churn): add core behavior analyzer with tests"

git add src/tracer/api/routes.py src/tracer/api/validators/
git commit -m "feat(api): add churn prediction endpoints"

# 3. Keep branch updated
git checkout master
git pull origin master
git checkout feature/churn-detection-core
git rebase master

# 4. Final push and PR
git push origin feature/churn-detection-core
gh pr create --title "feat(churn): implement complete detection system"
```

### Merge Strategy
- **Feature branches**: Squash merge to master for clean history
- **Hotfixes**: Fast-forward merge to preserve urgency context
- **Release branches**: Merge commit to preserve release context

## File Staging Strategy

### Automatic Grouping Rules
1. **Component + Tests**: Always stage implementation with corresponding tests
2. **API + Validation**: Stage route handlers with validation schemas
3. **Migration + Models**: Database changes require both migration and model files
4. **Documentation**: Include relevant docs with significant changes

### Pre-Commit Checklist
- [ ] All tests pass for modified components
- [ ] Code follows formatting standards (black, isort)
- [ ] Type hints are present (mypy validation)
- [ ] No sensitive data committed
- [ ] Import statements are clean
- [ ] Documentation updated for public APIs
- [ ] Commit message follows conventional format

## Quality Standards

### Commit Quality
- **Atomic**: Each commit represents one logical change
- **Self-contained**: Commit should not break the build
- **Reviewable**: Changes should be easy to understand
- **Revertible**: Each commit should be safely revertible

### Code Quality
- Minimum 80% test coverage for new features
- All public APIs must have docstrings
- Type hints required for all function signatures
- Follow PEP 8 style guidelines (enforced by pre-commit)

### Review Process
- All changes require PR review before merging
- CI must pass (tests, linting, type checking)
- Security scan must pass
- At least one approval from team member

## Tools and Automation

### Pre-commit Hooks
- **Black**: Code formatting
- **isort**: Import sorting
- **Flake8**: Linting
- **MyPy**: Type checking
- **detect-secrets**: Security scanning
- **conventional-pre-commit**: Commit message validation

### Git Aliases (Optional)
```bash
# Add to ~/.gitconfig or use git config --global
[alias]
  co = checkout
  br = branch
  ci = commit
  st = status
  unstage = reset HEAD --
  last = log -1 HEAD
  visual = !gitk
  
  # Milestone 1 specific
  feat = "!f() { git commit -m \"feat(churn): $1\"; }; f"
  fix = "!f() { git commit -m \"fix(churn): $1\"; }; f"
  test = "!f() { git commit -m \"test(churn): $1\"; }; f"
```

## Troubleshooting

### Common Issues

#### Pre-commit Hook Failures
```bash
# Skip hooks temporarily (not recommended)
git commit --no-verify

# Fix formatting issues
black src/ tests/
isort src/ tests/

# Update pre-commit hooks
pre-commit autoupdate
```

#### Large File Errors
```bash
# Remove from staging
git reset HEAD large_file.pkl

# Add to .gitignore
echo "*.pkl" >> .gitignore
```

#### Merge Conflicts
```bash
# Rebase instead of merge for cleaner history
git rebase master
# Resolve conflicts, then:
git add <resolved_files>
git rebase --continue
```

## Getting Help

- Review this guide for standard patterns
- Check commit history for examples: `git log --oneline --grep="feat(churn)"`
- Ask team for review of complex commit organizations
- Use `git commit --amend` to fix commit messages before pushing

Remember: Good commit organization helps everyone understand the project history and makes debugging, code review, and maintenance much easier.