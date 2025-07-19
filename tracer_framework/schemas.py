"""
Pydantic schemas for the Tracer Framework.

This module defines data validation and serialization schemas used for
API requests/responses and internal data structures.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator

from .types import (
    ModelID, ModelVersion, ModelType, MatchID, TeamID,
    Probability, Confidence, FeatureDict, PredictionID
)


class ExplanationFactor(BaseModel):
    """Represents a single factor contributing to a prediction."""
    factor: str = Field(..., description="Name of the contributing factor")
    impact: float = Field(..., ge=0.0, le=1.0, description="Impact score (0-1)")
    description: str = Field(..., description="Human-readable explanation")


class PredictionExplanation(BaseModel):
    """Agent-friendly explanation of a prediction."""
    top_factors: List[ExplanationFactor] = Field(
        ..., 
        description="Top factors influencing the prediction"
    )
    confidence_reasoning: Optional[str] = Field(
        None,
        description="Explanation of confidence score"
    )


class ModelMetadata(BaseModel):
    """Metadata about a model version."""
    model_id: ModelID
    model_version: ModelVersion
    model_type: ModelType
    training_date: datetime
    feature_version: str
    performance_metrics: Optional[Dict[str, float]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MatchPrediction(BaseModel):
    """Prediction for a single match."""
    prediction_id: PredictionID
    match_id: MatchID
    team_a_id: TeamID
    team_b_id: TeamID
    team_a_win_probability: Probability
    confidence: Confidence
    explanation: PredictionExplanation
    metadata: ModelMetadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('team_a_win_probability', 'confidence')
    def validate_probability(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Probability must be between 0 and 1')
        return v


class MatchInput(BaseModel):
    """Input data for making a match prediction."""
    match_id: MatchID
    team_a_id: TeamID
    team_b_id: TeamID
    features: Optional[FeatureDict] = Field(
        None,
        description="Optional pre-calculated features"
    )
    tournament_id: Optional[str] = None
    scheduled_time: Optional[datetime] = None


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    matches: List[MatchInput]
    include_explanations: bool = Field(
        True,
        description="Whether to include detailed explanations"
    )


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[MatchPrediction]
    batch_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the batch processing"
    )