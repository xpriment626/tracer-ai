"""
Core type definitions for the Tracer Framework.

This module defines custom types used throughout the framework to ensure
type safety and clarity in function signatures.
"""

from typing import NewType, Dict, Any, List, Optional, Union
from datetime import datetime
from decimal import Decimal

# Model-related types
ModelID = NewType('ModelID', str)
ModelVersion = NewType('ModelVersion', str)
ModelType = NewType('ModelType', str)

# Match and team-related types
MatchID = NewType('MatchID', str)
TeamID = NewType('TeamID', str)
PlayerID = NewType('PlayerID', str)
TournamentID = NewType('TournamentID', str)

# Prediction-related types
Probability = NewType('Probability', float)  # Value between 0.0 and 1.0
Confidence = NewType('Confidence', float)    # Value between 0.0 and 1.0
Score = NewType('Score', int)

# Feature-related types
FeatureName = NewType('FeatureName', str)
FeatureValue = Union[float, int, str, bool]
FeatureDict = Dict[FeatureName, FeatureValue]

# Data types
Timestamp = NewType('Timestamp', datetime)
ELORating = NewType('ELORating', float)

# Agent-specific types
ExplanationFactor = Dict[str, Any]  # Will be refined in schemas.py
MetadataDict = Dict[str, Any]      # Will be refined in schemas.py

# API response types
PredictionID = NewType('PredictionID', str)
BatchID = NewType('BatchID', str)