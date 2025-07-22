# Git Quick Reference - Milestone 1 Customer Churn Detection

## 🚀 Quick Start

```bash
# Setup (first time only)
./scripts/git_helpers.sh setup

# Start new feature
./scripts/git_helpers.sh feature churn-detection-core

# Make changes, then smart commit
git add <files>
./scripts/git_helpers.sh commit
```

## 📝 Commit Message Format

```
<type>(<scope>): <subject>
```

### Types
- `feat` - New features
- `fix` - Bug fixes  
- `test` - Adding tests
- `docs` - Documentation
- `refactor` - Code refactoring
- `perf` - Performance improvements
- `chore` - Build/tooling changes

### Scopes (Milestone 1)
- `churn` - Core churn detection
- `customer` - Customer data management
- `prediction` - ML prediction services
- `pipeline` - Data processing
- `monitoring` - System monitoring
- `api` - API endpoints
- `core` - Core business logic
- `data` - Data models

## 💡 Common Patterns

### Feature Development
```bash
feat(churn): add customer behavior analyzer with tests

Files:
- src/tracer/core/analyzer.py
- tests/unit/test_analyzer.py  
- docs/core/analyzer.md
```

### API Development
```bash
feat(api): add churn prediction endpoints

Files:
- src/tracer/api/churn_routes.py
- src/tracer/api/validators/churn.py
- tests/integration/test_churn_api.py
```

### Bug Fixes
```bash
fix(prediction): handle missing customer attributes

Files:
- src/tracer/core/predictor.py
- tests/unit/test_predictor_edge_cases.py
```

## 🌿 Branch Naming

```bash
feature/churn-detection-core
feature/customer-data-pipeline
fix/customer-data-validation
fix/api-error-handling
perf/data-processing-optimization
docs/api-documentation
```

## 🔧 Helper Commands

```bash
# Create feature branch
./scripts/git_helpers.sh feature <name>

# Create fix branch  
./scripts/git_helpers.sh fix <name>

# Smart commit with analysis
./scripts/git_helpers.sh commit

# Update branch with master
./scripts/git_helpers.sh update

# Show development stats
./scripts/git_helpers.sh stats

# Clean up merged branches
./scripts/git_helpers.sh cleanup
```

## ⚡ Quick Commands

```bash
# Check status
git status

# Stage files
git add <files>
git add .  # All files

# Commit with template
git commit  # Opens editor with template

# Push branch
git push origin <branch-name>

# Create PR
gh pr create --title "feat(churn): implement detection system"
```

## 🛡️ Pre-commit Hooks

Automatically run on every commit:
- **Code formatting** (Black, isort)
- **Linting** (Flake8)
- **Type checking** (MyPy)
- **Security scanning** (detect-secrets)
- **Commit message validation**

## 📊 File Grouping Rules

1. **Implementation + Tests** - Always together
2. **API + Validation** - Route handlers with schemas
3. **Database Changes** - Models with migrations
4. **Features + Docs** - Public APIs with documentation

## 🎯 Examples

### Good Commits
```bash
✅ feat(churn): add real-time behavior tracking
✅ fix(api): handle missing customer data gracefully
✅ test(pipeline): increase data validation coverage
✅ docs(api): add churn prediction examples
```

### Bad Commits
```bash
❌ feat: added some stuff
❌ bug fix
❌ Update code
❌ FIX: Fixed the thing that was broken
```

## 🆘 Troubleshooting

### Skip pre-commit (emergency only)
```bash
git commit --no-verify
```

### Fix formatting issues
```bash
black src/ tests/
isort src/ tests/
```

### Rebase instead of merge
```bash
git rebase master
```

### Amend last commit message
```bash
git commit --amend
```

---

**Remember**: Good commits help everyone understand the project history and make debugging, reviews, and maintenance easier! 🎉