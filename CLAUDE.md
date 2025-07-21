# Project Context: Tracer Framework - General Purpose SQM Platform

## Project Overview

Building the Tracer Framework - a General Purpose Small Quantitative Model (SQM) platform for quantitative prediction models across domains, with Model 1 (eSports Match Winner Prediction) as the flagship implementation and domain template. This is designed to be a pioneering framework for agentic AI applications with customer-in-the-loop evolution.

## Key Development Principles

1. **Model 1 is our template** - Build it with production-grade infrastructure as the blueprint for all domain implementations
2. **Framework-first approach** - We're building the Tracer Framework for a multi-domain SQM ecosystem, not just esports
3. **Customer-driven evolution** - Framework adapts and improves based on real-time customer feedback and usage patterns
4. **Agent-ready from day one** - All APIs and outputs designed for AI agent consumption across domains
5. **Budget-conscious** - Working within $500 for initial data and model training per domain
6. **Speed to market** - 4-week sprint to get domain-agnostic platform deployable

## Technical Stack

- **ML Framework**: Python with scikit-learn, LightGBM, SHAP for domain-agnostic modeling
- **API**: FastAPI with OpenAPI specs, auto-generated per domain
- **Configuration**: Dynamic domain configuration system for customer adaptations
- **Feedback Loop**: Real-time customer interaction tracking and model improvement pipeline
- **Data Sources**: Pluggable data pipeline patterns (starting with PandaScore API, Kaggle datasets, web scraping)
- **Infrastructure**: Google Colab for training, basic cloud hosting for multi-domain demo

## Project Structure

```
tracer-ai/
├── tracer_framework/      # Core Tracer Framework
│   ├── core/              # Domain-agnostic base classes and interfaces
│   ├── domains/           # Domain-specific implementations
│   │   └── esports/       # eSports domain (Model 1)
│   ├── config/            # Dynamic configuration and customer adaptations
│   ├── feedback/          # Customer-in-the-loop systems
│   ├── interpretability/  # SHAP integration across domains
│   ├── serving/           # API and agent interfaces
│   └── evaluation/        # Domain-agnostic metrics and backtesting
├── models/                # Domain-specific model implementations
│   └── match_winner/      # Model 1 - eSports match winner (template)
├── data/                  # Domain-agnostic data pipeline and storage
├── api/                   # Multi-domain FastAPI application
├── ui/                    # Domain marketplace and demo interface
└── ctx/                   # External context documents
```

## Current Sprint (Week 1-4)

### Week 1: Multi-Domain Framework Architecture
- Build domain-agnostic SQM base classes
- Design agent-friendly interfaces with customer feedback loops
- Set up model registry and domain template system

### Week 2: Model 1 + Domain Abstraction
- Implement eSports match winner as domain template
- Create feature engineering pipeline for domain reuse
- Build customer feedback collection infrastructure

### Week 3: Multi-Tenant API & Productization
- FastAPI with tier-based access and domain routing
- Customer onboarding pipeline implementation
- Basic UI showcasing domain marketplace concept

### Week 4: Testing & Production Readiness
- Comprehensive testing across domain abstraction layer
- API documentation for agent consumption
- MVP deployment with subscription tier enforcement

## Key Platform Features

### Multi-Domain Support
- **Domain marketplace**: Easy integration of new prediction domains (finance, sports, logistics, etc.)
- **Template-driven**: New domains built from proven esports template
- **Rapid deployment**: Domain-specific APIs and UIs auto-generated from configuration

### Customer-in-the-Loop Evolution
- **Real-time feedback**: Customer interactions continuously improve model performance
- **Dynamic configuration**: Models adapt to customer-specific use cases and preferences
- **A/B testing**: Built-in framework for comparing model variants across domains

### Universal Model Features (Starting with Model 1 - eSports)
- **Prediction accuracy**: Target >68% across all domains
- **Full interpretability**: SHAP values for every prediction, domain-contextualized
- **Agent-friendly output**: Semantic explanations with confidence scores
- **Version control**: Git-like model versioning across domains
- **Cross-domain learning**: Insights from one domain inform others

## Development Guidelines

1. **Model 1 as domain template** - Build it as the blueprint for all future domain implementations
2. **Framework-first, domain-agnostic** - All core components must work across domains
3. **Mock future domains** - UI can show marketplace with "coming soon" for other domains
4. **Document for agents** - All APIs should have clear semantic descriptions across domains
5. **Test domain abstraction** - Ensure new domains can be added with minimal code
6. **Customer feedback priority** - Build feedback loops before advanced features
7. **Stay within budget** - Monitor API usage carefully across all domains

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

## API Endpoints (Multi-Domain + Productization)

### Domain-Agnostic Endpoints
- `POST /predict/{domain}` - Get domain-specific predictions with interpretability
- `POST /predict/natural` - Natural language prediction queries for agents
- `GET /domains/` - List available domain templates and customer models
- `GET /models/{model_id}/metadata` - Get model metadata and performance metrics

### Customer Management
- `POST /customers/onboard` - Customer onboarding and domain assessment
- `GET /customers/usage` - Usage analytics and billing information
- `POST /feedback/` - Submit prediction feedback for model improvement

### Marketplace & Training
- `POST /train/custom` - Train custom model on customer data
- `GET /marketplace/domains` - Browse available domain templates
- `POST /evaluate/backtest` - Run backtesting across domains

## Implementation Plan

### Phase 1: Domain Abstraction Layer
1. **Abstract existing domain-specific code**
   - `tracer_framework/types.py` - Generalize to domain-agnostic base types
   - `tracer_framework/schemas.py` - Create generic prediction schemas
   - `tracer_framework/domains/esports/` - Move esports-specific types and schemas

2. **Create domain registry and configuration**
   - `tracer_framework/config/domain_config.py` - Dynamic domain configuration
   - `tracer_framework/core/domain_registry.py` - Domain plugin system
   - `tracer_framework/domains/base.py` - Abstract domain interface

### Phase 2: Generalized Base Model
3. **Refactor BaseSQM for multi-domain**
   - `tracer_framework/core/base_model.py` - Domain-agnostic abstract base class
   - Methods: `predict()`, `get_metadata()`, `validate_input()`, `get_domain_config()`
   - Pluggable feature and prediction pipelines

4. **Customer feedback infrastructure**
   - `tracer_framework/feedback/collector.py` - Customer interaction tracking
   - `tracer_framework/feedback/adapter.py` - Model improvement pipeline
   - `tracer_framework/config/customer_config.py` - Customer-specific adaptations

### Phase 3: Domain Marketplace Foundation
5. **Feature abstraction and marketplace**
   - `tracer_framework/features/base.py` - Domain-agnostic feature interface
   - `tracer_framework/features/registry.py` - Feature marketplace
   - `tracer_framework/domains/esports/features/` - Move esports features

### Phase 4: Multi-Domain API Layer
6. **Domain-aware API endpoints**
   - `api/domain_router.py` - Auto-generate APIs per domain
   - `api/prediction_endpoints.py` - Universal prediction interface
   - `api/feedback_endpoints.py` - Customer feedback collection

### Phase 5: Domain Template and Testing
7. **Complete esports domain as template**
   - `models/match_winner/model.py` - Refactor to use new domain system
   - Comprehensive testing across domain abstraction layer
   - Documentation for adding new domains

### Benefits of This Approach:
- Each file is <100 lines initially
- Can test each component in isolation
- Clear dependencies between components
- Easy to debug when something breaks
- Natural git commit boundaries

## Productization Strategy

### MVP Launch: Models 1-3 as Domain Templates

**Model 1**: eSports Match Winner (flagship - fully production ready)  
**Model 2**: Financial Market Signals (crypto/stock predictions)  
**Model 3**: Supply Chain Predictions (delivery delays, demand forecasting)

### Subscription Tiers

#### **Tracer Starter** ($99/month)
- Access to all public Models 1-3 via API
- 10K predictions/month
- Standard interpretability reports
- Community support
- Agent-ready API access

#### **Tracer Professional** ($499/month)
- Everything in Starter + 100K predictions/month
- Custom model training on customer data
- Advanced SHAP interpretability
- Customer feedback loop integration
- Priority support + white-label APIs

#### **Tracer Enterprise** ($2,499/month + usage)
- Everything in Professional + unlimited predictions
- Private model hosting + custom domains
- Dedicated customer success + SLA guarantees
- On-premise deployment options

### Customer Journey

**Phase 1: Consumption** → API access to pre-trained Models 1-3  
**Phase 2: Customization** → Upload data, auto-train custom models  
**Phase 3: Domain Creation** → Commission new domains for marketplace

### Multi-Tenant Architecture

```
tracer-cloud/
├── api-gateway/           # Route requests to appropriate models
├── model-registry/        # Manage public and private models  
├── training-service/      # Auto-train customer models
├── prediction-service/    # Serve predictions at scale
├── feedback-service/      # Collect customer feedback
└── domain-marketplace/    # Discover and deploy new domains
```

## Important Notes

- Model 1 (eSports) is the template for all future domain implementations - build it as the foundation of the multi-domain Tracer Framework ecosystem
- We're pioneering a General Purpose SQM Platform to fit into the broader landscape of agentic applications across industries
- Focus on domain abstraction, interpretability, and agent-friendly outputs from the start
- Customer-in-the-loop feedback is the key differentiator - prioritize this over advanced ML features
- Keep infrastructure lightweight but extensible across domains
- Each new domain should require minimal code changes to the core framework