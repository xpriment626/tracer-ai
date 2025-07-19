# Project Context: Tracer Framework - eSports SQM MVP

## Project Overview

Building the Tracer Framework - a Small Quantitative Model (SQM) framework for esports betting predictions, with Model 1 (Match Winner Prediction) as the flagship implementation. This is designed to be a pioneering framework for agentic AI applications.

## Key Development Principles

1. **Model 1 is our baby** - Build it with production-grade infrastructure while everything else can be scrappy
2. **Framework-first approach** - We're building the Tracer Framework for an SQM ecosystem, not just a single model
3. **Agent-ready from day one** - All APIs and outputs designed for AI agent consumption
4. **Budget-conscious** - Working within $500 for data and model training
5. **Speed to market** - 4-week sprint to get something deployable

## Technical Stack

- **ML Framework**: Python with scikit-learn, LightGBM, SHAP
- **API**: FastAPI with OpenAPI specs
- **Data Sources**: PandaScore API (free tier), Kaggle datasets, web scraping
- **Infrastructure**: Google Colab for training, basic cloud hosting for demo

## Project Structure

```
tracer-ai/
├── tracer_framework/      # Core Tracer Framework
│   ├── core/              # Base classes and interfaces
│   ├── interpretability/  # SHAP integration
│   ├── serving/           # API and agent interfaces
│   └── evaluation/        # Metrics and backtesting
├── models/                # Model implementations
│   └── match_winner/      # Model 1 implementation
├── data/                  # Data pipeline and storage
├── api/                   # FastAPI application
├── ui/                    # Basic web interface
└── ctx/                   # External context documents
```

## Current Sprint (Week 1-4)

### Week 1: Tracer Framework Architecture
- Build abstract SQM base classes for Tracer
- Design agent-friendly interfaces
- Set up model registry and versioning

### Week 2: Model 1 Implementation
- Implement match winner prediction using Tracer Framework
- Feature engineering pipeline
- Train within budget constraints

### Week 3: API & Integration
- FastAPI with agent-optimized endpoints
- Basic UI showcasing Tracer capabilities
- Batch prediction support

### Week 4: Testing & Documentation
- Comprehensive testing suite for Tracer Framework
- API documentation
- Deployment preparation

## Key Features for Model 1

- **Prediction accuracy**: Target >68%
- **Full interpretability**: SHAP values for every prediction
- **Agent-friendly output**: Semantic explanations with confidence scores
- **Version control**: Git-like model versioning
- **A/B testing**: Built-in framework for model comparison

## Development Guidelines

1. **Always prioritize Model 1** - This is our flagship demonstration of Tracer Framework
2. **Mock what's not ready** - UI can show "coming soon" for Models 2-5
3. **Document for agents** - All APIs should have clear semantic descriptions
4. **Test early and often** - Especially Model 1's predictions
5. **Stay within budget** - Monitor API usage carefully

## External Context

Additional context documents are stored in the `ctx/` folder:
- `mvp_sprint.md` - Detailed MVP sprint plan
- `esports_sqm.md` - Original comprehensive briefing document

## Testing Commands

```bash
# Run tests
pytest tests/

# Check code quality
flake8 tracer_framework/
black tracer_framework/

# Run model evaluation
python -m tracer_framework.evaluation.backtesting
```

## API Endpoints (Once Implemented)

- `POST /predict/match_winner` - Get match prediction with interpretability
- `GET /models/` - List available models and versions
- `GET /models/{model_id}/metadata` - Get model metadata
- `POST /evaluate/backtest` - Run backtesting on historical data

## Implementation Plan

### Phase 1: Core Data Structures
1. **Create basic types and dataclasses**
   - `tracer_framework/types.py` - Simple type definitions (ModelID, Version, etc.)
   - `tracer_framework/schemas.py` - Pydantic models for predictions, features, metadata

2. **Build prediction interface**
   - `tracer_framework/core/prediction.py` - Single prediction result class
   - Focus: Just the data structure, no logic yet

### Phase 2: Abstract Base Model
3. **Create minimal BaseSQM**
   - `tracer_framework/core/base_model.py` - Abstract base class
   - Only 3 methods: `predict()`, `get_metadata()`, `validate_input()`
   - No complex features yet

4. **Add model metadata**
   - `tracer_framework/core/metadata.py` - Simple metadata storage
   - Version string, model type, creation date

### Phase 3: Simple Feature Pipeline
5. **Basic feature definition**
   - `tracer_framework/features/base.py` - Feature abstract class
   - `tracer_framework/features/elo.py` - One concrete feature (ELO rating)
   - Test with hardcoded data first

### Phase 4: Minimal Model Implementation
6. **Create skeleton MatchWinnerSQM**
   - `models/match_winner/model.py` - Inherits from BaseSQM
   - Start with random predictions
   - Add one real feature at a time

### Phase 5: Basic Testing Infrastructure
7. **Unit tests for each component**
   - `tests/test_base_model.py`
   - `tests/test_prediction.py`
   - Test with mock data only

### Benefits of This Approach:
- Each file is <100 lines initially
- Can test each component in isolation
- Clear dependencies between components
- Easy to debug when something breaks
- Natural git commit boundaries

## Important Notes

- Model 1 should be built with the future state in mind - it's not just a model but the foundation of the Tracer Framework ecosystem
- We're pioneering the Tracer Framework to fit into the broader landscape of agentic applications
- Focus on interpretability and agent-friendly outputs from the start
- Keep infrastructure lightweight but extensible