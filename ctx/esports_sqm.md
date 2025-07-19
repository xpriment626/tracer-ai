<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Product Briefing Document: eSports Betting Signals Platform - SQM Phase

## Executive Summary

**Purpose**: Engineering handoff briefing for Phase 1 development of a small quantitative models (SQMs) based eSports betting signals platform.

**Scope**: MVP focused exclusively on building lightweight, specialized prediction models for eSports match outcomes and betting signal generation - laying the foundation for future AI agent integration.

**Technical Objective**: Develop and deploy 3-5 specialized small quantitative models targeting specific eSports betting scenarios, achieving >65% prediction accuracy while maintaining sub-second inference times.

## Technical Architecture Overview

### Core SQM Framework

**Small Quantitative Models Definition**: Lightweight, domain-specific ML models (typically <100MB) optimized for narrow prediction tasks with fast inference and high interpretability[^1][^2].

**Target Model Types**:

- **Binary Classification Models**: Match winner prediction (Team A vs Team B)
- **Regression Models**: Score differential prediction
- **Time-Series Models**: In-match momentum shift detection
- **Ensemble Models**: Combined signals from multiple specialized models


### Key Technical Requirements

**Model Performance Specifications**:

- Inference time: <500ms per prediction
- Memory footprint: <50MB per model
- Prediction accuracy: >65% (industry benchmark for eSports ML models[^3][^4])
- Model interpretability: SHAP values for all predictions

**Infrastructure Requirements**:

- Real-time data ingestion pipeline supporting 100+ concurrent matches
- Model serving infrastructure with auto-scaling capabilities
- A/B testing framework for model comparison
- Feature store for consistent data preprocessing


## Data Pipeline Architecture

### Primary Data Sources

**eSports Odds APIs** (Previously validated availability):

- **PandaScore**: 132K+ historical matches, 308 unique markets, real-time odds[^5]
- **OddsMatrix**: Live odds updates, cross-verified accuracy, fraud monitoring[^6][^7]
- **Abios**: 15K+ monthly events, 60+ market types, official tournament data[^8][^9]
- **Bayes Esports**: Multiple odds providers through unified API[^10]

**Match Data Requirements**:

- Historical match results (minimum 10K matches per game)
- Player/team statistics and performance metrics
- Real-time match state data (kills, objectives, economic advantage)
- Betting odds time series data


### Feature Engineering Pipeline

**Core Feature Categories**:

1. **Team Strength Metrics**: ELO ratings, recent form, head-to-head records
2. **Market Inefficiency Signals**: Odds discrepancies across bookmakers
3. **Momentum Indicators**: In-game objective control, gold differential trends
4. **Meta-Game Features**: Champion/map selection patterns, ban priorities

**Feature Store Architecture**:

- Real-time feature computation and caching
- Historical feature backfill for model training
- Feature versioning for model reproducibility
- Data quality monitoring and alerting


## SQM Model Specifications

### Model 1: Match Winner Prediction

**Architecture**: Logistic Regression with regularization
**Input Features**: Team ratings, recent form, map pool advantage (15-20 features)
**Target Accuracy**: >70%
**Use Case**: Pre-match betting signals

### Model 2: Score Differential Prediction

**Architecture**: Random Forest Regression
**Input Features**: Team strength metrics, historical scoring patterns (25-30 features)
**Target Accuracy**: RMSE <2.5 points
**Use Case**: Spread betting optimization

### Model 3: Live Momentum Detection

**Architecture**: LightGBM with time-series features
**Input Features**: Real-time game state, objective control, economic metrics (35-40 features)
**Target Accuracy**: >65% for momentum shift prediction
**Use Case**: Live betting opportunities

### Model 4: Map-Specific Predictor

**Architecture**: Support Vector Machine with RBF kernel
**Input Features**: Team map win rates, pick/ban patterns, player comfort levels (20-25 features)
**Target Accuracy**: >68%
**Use Case**: Map-specific betting markets

### Model 5: Ensemble Meta-Model

**Architecture**: Weighted averaging with confidence scoring
**Input Features**: Predictions from Models 1-4 plus confidence intervals
**Target Accuracy**: >72% (expected 3-5% lift over individual models)
**Use Case**: Primary signal generation for production

## Development Methodology

### MVP Development Timeline (12 weeks)

**Weeks 1-3: Data Infrastructure**

- API integrations and data pipeline setup
- Feature store implementation
- Historical data backfill (targeting 50K+ matches minimum[^11])

**Weeks 4-8: Model Development**

- Individual SQM training and validation
- Hyperparameter optimization using grid search
- Cross-validation with time-series splits[^12]

**Weeks 9-10: Ensemble \& Integration**

- Meta-model development and ensemble optimization
- API endpoint development for model serving
- Performance monitoring dashboard

**Weeks 11-12: Testing \& Validation**

- Backtesting on out-of-sample data
- A/B testing framework setup
- Production deployment preparation


### Technical Success Metrics

**Model Performance KPIs**:

- Prediction accuracy benchmarked against closing odds
- Sharpe ratio of generated signals >1.0
- Model stability across different time periods
- Feature importance consistency

**Engineering KPIs**:

- API response time <200ms (95th percentile)
- System uptime >99.5%
- Data pipeline latency <30 seconds
- Model serving throughput >1000 requests/minute


## Risk Mitigation \& Technical Challenges

### Primary Technical Risks

**Data Quality \& Consistency**:

- **Risk**: eSports data can have inconsistencies across sources[^13]
- **Mitigation**: Multi-source validation, automated data quality checks

**Model Overfitting**:

- **Risk**: Small datasets leading to overfitting[^14]
- **Mitigation**: Cross-validation, regularization, ensemble methods

**Real-time Performance**:

- **Risk**: Latency in live betting scenarios
- **Mitigation**: Model pre-computation, caching strategies, edge deployment


### Technical Debt Considerations

**Code Architecture**: Modular design allowing easy model swapping and version control
**Documentation**: Comprehensive model documentation for future AI agent integration
**Testing**: Unit tests for all feature engineering and model prediction logic
**Monitoring**: Comprehensive model drift detection and alerting systems

## Future-Proofing for AI Agent Integration

### Design Principles for Phase 2 Compatibility

**Model Interpretability**: All SQMs designed with SHAP-compatible architectures for AI agent reasoning
**API Standardization**: RESTful APIs with OpenAPI specifications for easy agent integration
**Signal Standardization**: Consistent confidence scoring and uncertainty quantification
**Metadata Tracking**: Complete model lineage and feature attribution for agent decision-making

### Handoff Deliverables to Engineering

1. **Technical Specification Document**: Detailed model architectures and requirements
2. **Data Schema Documentation**: Complete feature definitions and data pipeline specs
3. **API Specification**: OpenAPI definitions for all model endpoints
4. **Testing Framework**: Unit and integration test suites
5. **Deployment Guide**: Container specifications and infrastructure requirements
6. **Performance Benchmarks**: Success criteria and monitoring dashboard specs

## Budget \& Resource Requirements

**Development Team**: 3-4 engineers (ML Engineer, Backend Engineer, Data Engineer, DevOps)
**Infrastructure Costs**: ~\$5K/month for cloud resources during development
**Data Costs**: ~\$2K/month for API subscriptions and data feeds
**Timeline**: 12 weeks to production-ready MVP

This briefing provides the engineering team with a complete technical roadmap for building the SQM foundation that will support future vertical AI agent development while delivering immediate value through specialized betting signal generation.

<div style="text-align: center">⁂</div>

[^1]: https://www.underdogchance.com/quant-ai-sports-betting/

[^2]: https://arxiv.org/abs/2410.21484

[^3]: https://www.reddit.com/r/leagueoflegends/comments/19e91me/training_machine_learning_models_to_predict/

[^4]: https://github.com/SamuelAitamaa/lolesports-predictor

[^5]: https://www.pandascore.co

[^6]: https://oddsmatrix.com/esports/

[^7]: https://oddsmatrix.com/odds-service/esports/

[^8]: https://abiosgaming.com/esports-odds

[^9]: https://abiosgaming.com/esports-data-api

[^10]: https://www.bayesesports.com/product/live-odds

[^11]: https://thesis.eur.nl/pub/59277/Final-Thesis-Dion-van-Wijk-477793-.pdf

[^12]: https://journals.sagepub.com/doi/10.3233/JSA-200463

[^13]: https://intellias.com/machine-learning-for-sports-betting/

[^14]: https://www.youtube.com/watch?v=Log7X5Mj_vc

[^15]: https://arxiv.org/html/2506.04602v2

[^16]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9070895/

[^17]: https://www.reddit.com/r/SoccerBetting/comments/135fu06/prediction_model_for_value_betting_machine/

[^18]: https://www.reddit.com/r/leagueoflegends/comments/1ckt0m1/exmaster_player_turned_data_scientist_how/

[^19]: https://www.dataart.com/blog/5-use-cases-for-machine-learning-in-sports-betting

[^20]: https://escharts.com

[^21]: https://www.reddit.com/r/quant/comments/1da7f73/sports_betting_strategies/

[^22]: https://www.sloansportsconference.com/event/enhancing-valorant-performance-computer-vision-and-data-science-in-esports

[^23]: https://journals.sagepub.com/doi/10.3233/JIFS-232932?int.sj-full-text.similar-articles.6

[^24]: https://www.sciencedirect.com/science/article/pii/S266682702400015X

[^25]: https://esportslane.com/data-analytics-esports/

[^26]: https://journals.library.brocku.ca/index.php/jess/article/download/4544/3280/15589

[^27]: https://www.sarjournal.com/content/73/SARJournalSeptember2024_184_189.pdf

[^28]: https://www.sciencedirect.com/science/article/pii/S074756322400219X

[^29]: https://ideausher.com/blog/aimodels-sports-prediction-betting-apps/

[^30]: https://www.datascience-pm.com/data-science-mvp/

[^31]: https://esportsinsider.com/betting-odds

[^32]: https://core.ac.uk/download/pdf/210608830.pdf

[^33]: https://oddin.gg/official-esports-data

[^34]: https://www.diva-portal.org/smash/get/diva2:1931818/FULLTEXT01.pdf

[^35]: https://oddin.gg/esports-odds-feed

[^36]: https://the-odds-api.com

[^37]: https://everymatrix.com/oddsmatrix/esports-services/

[^38]: https://paperswithcode.com/paper/real-time-esports-match-result-prediction/review/

[^39]: https://esport-api.com

[^40]: https://beter.co/odds/

[^41]: https://www.kaggle.com/code/businesstech/predictive-analytics-in-esports-tournaments

[^42]: https://oddin.gg

[^43]: https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting

[^44]: https://digitalcommons.molloy.edu/cgi/viewcontent.cgi?article=1068\&context=bus_fac

[^45]: https://www.nature.com/articles/s41598-025-87794-y

[^46]: https://arxiv.org/html/2410.21484v1

[^47]: https://kth.diva-portal.org/smash/get/diva2:1909097/FULLTEXT01.pdf

[^48]: https://www.sciencedirect.com/science/article/pii/S2772662223001364

[^49]: https://vtechworks.lib.vt.edu/bitstreams/cc2b9977-e31c-4a3a-babf-f9138c163464/download

[^50]: https://drpress.org/ojs/index.php/HSET/article/view/21842/21368

[^51]: https://journals.sagepub.com/doi/pdf/10.3233/JSA-200463

[^52]: https://www.reddit.com/r/Python/comments/dyu01a/sports_betting_and_data_mining/

[^53]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9199346/

[^54]: https://www.youtube.com/watch?v=vKiRZIiXRkA

[^55]: https://journals.sagepub.com/doi/abs/10.3233/JSA-200463

[^56]: https://owlab.group/case-studies/gaming-stars

[^57]: https://pictureperfectportfolios.com/how-to-implement-a-quantitative-small-cap-investing-strategy/

[^58]: https://www.cs.utexas.edu/~cart/trips/publications/micro06_trips.pdf

[^59]: https://crustlab.com/case-study/dreampicks-online-sports-betting-platform-mvp-design/

[^60]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7176071/

[^61]: https://ruc.udc.es/dspace/bitstream/handle/2183/25671/F.Laport_2020_A_Prototype_of_EEG_System_for_IoT.pdf

[^62]: https://businessplan-templates.com/blogs/start-business/online-esports-betting-platform

[^63]: https://www.studysmarter.co.uk/explanations/math/applied-mathematics/quantitative-modeling/

[^64]: http://kth.diva-portal.org/smash/get/diva2:526960/FULLTEXT01.pdf

[^65]: https://www.suffescom.com/product/white-label-esports-betting-software

[^66]: https://www.coursera.org/learn/wharton-quantitative-modeling

[^67]: https://www.ijcai.org/proceedings/2024/0464.pdf

[^68]: https://prometteursolutions.com/blog/why-invest-in-sports-betting-app-development-in-2025/

[^69]: https://wjarr.com/sites/default/files/WJARR-2024-0465.pdf

[^70]: https://dl.acm.org/doi/pdf/10.5555/947185.947188

[^71]: https://esportsinsider.com/us/gambling/esports-betting-sites

[^72]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6949132/

[^73]: https://citeseerx.ist.psu.edu/document?repid=rep1\&type=pdf\&doi=34b5bbd21f4687c66904add09c5e3c6e60625ff2

[^74]: https://www.brsoftech.com/blog/esports-betting-software-development/

[^75]: https://unfccc.int/sites/default/files/hp_unfccc_may17.pdf

