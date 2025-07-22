"""
Data Preprocessing Service for Tracer Framework

Advanced data preprocessing pipeline with:
- Automated feature engineering
- Data cleaning and transformation
- Encoding categorical variables
- Handling missing values
- Feature scaling and normalization
- Outlier detection and treatment
- Blueprint-specific preprocessing
"""

import asyncio
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
import structlog

from .models import (
    ProcessingResult,
    ProcessingStatus, 
    FeatureEngineering,
    DataQualityIssue,
    DataQualityLevel
)
from .schemas import schema_registry

logger = structlog.get_logger(__name__)

# Suppress sklearn warnings for cleaner logs
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class PreprocessingError(Exception):
    """Base exception for preprocessing errors"""
    pass


class InsufficientDataError(PreprocessingError):
    """Raised when insufficient data for preprocessing"""
    pass


class FeatureEngineeringEngine:
    """
    Automated feature engineering for ML blueprints
    
    Creates domain-specific features based on blueprint requirements
    """
    
    def __init__(self):
        self.feature_generators = {
            'customer_churn': self._generate_churn_features,
            'revenue_projection': self._generate_revenue_features,
            'price_optimization': self._generate_price_features
        }
    
    async def generate_features(
        self,
        df: pd.DataFrame,
        blueprint_name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, FeatureEngineering]:
        """
        Generate blueprint-specific features
        
        Args:
            df: Input dataframe
            blueprint_name: Name of the blueprint
            config: Optional configuration parameters
            
        Returns:
            Tuple of (enhanced_dataframe, feature_engineering_report)
        """
        config = config or {}
        generator = self.feature_generators.get(blueprint_name)
        
        if not generator:
            logger.warning(f"No feature generator found for blueprint: {blueprint_name}")
            return df.copy(), FeatureEngineering(dataset_id="unknown")
        
        start_time = datetime.utcnow()
        
        try:
            enhanced_df, feature_report = await generator(df, config)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                "Feature generation completed",
                blueprint_name=blueprint_name,
                features_created=len(feature_report.features_created),
                duration_seconds=duration
            )
            
            return enhanced_df, feature_report
            
        except Exception as e:
            logger.error(
                "Feature generation failed",
                blueprint_name=blueprint_name,
                error=str(e),
                exc_info=True
            )
            raise PreprocessingError(f"Feature generation failed: {str(e)}")
    
    async def _generate_churn_features(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, FeatureEngineering]:
        """Generate features for customer churn detection"""
        
        enhanced_df = df.copy()
        feature_report = FeatureEngineering(dataset_id=config.get('dataset_id', 'unknown'))
        
        # Current timestamp for calculations
        current_time = datetime.utcnow()
        
        # Feature 1: Days since last activity (recency)
        if 'last_activity_date' in enhanced_df.columns:
            enhanced_df['days_since_last_activity'] = (
                current_time - pd.to_datetime(enhanced_df['last_activity_date'])
            ).dt.days
            feature_report.add_feature(
                'days_since_last_activity',
                'Time difference between current date and last activity'
            )
        
        # Feature 2: Account tenure (days since account creation)
        if 'account_created_date' in enhanced_df.columns:
            enhanced_df['account_tenure_days'] = (
                current_time - pd.to_datetime(enhanced_df['account_created_date'])
            ).dt.days
            feature_report.add_feature(
                'account_tenure_days',
                'Number of days since account creation'
            )
        
        # Feature 3: Customer lifetime value per day
        if all(col in enhanced_df.columns for col in ['total_value', 'account_tenure_days']):
            enhanced_df['daily_value_rate'] = (
                enhanced_df['total_value'] / (enhanced_df['account_tenure_days'] + 1)
            ).replace([np.inf, -np.inf], 0)
            feature_report.add_feature(
                'daily_value_rate',
                'Average daily customer value (total_value / account_tenure_days)'
            )
        
        # Feature 4: Support ticket intensity
        if all(col in enhanced_df.columns for col in ['support_tickets', 'account_tenure_days']):
            enhanced_df['support_intensity'] = (
                enhanced_df['support_tickets'] / (enhanced_df['account_tenure_days'] / 30 + 1)
            ).replace([np.inf, -np.inf], 0)
            feature_report.add_feature(
                'support_intensity',
                'Support tickets per month (support_tickets / months_active)'
            )
        
        # Feature 5: Feature usage rate
        if all(col in enhanced_df.columns for col in ['feature_usage_count', 'account_tenure_days']):
            enhanced_df['feature_usage_rate'] = (
                enhanced_df['feature_usage_count'] / (enhanced_df['account_tenure_days'] + 1)
            ).replace([np.inf, -np.inf], 0)
            feature_report.add_feature(
                'feature_usage_rate',
                'Daily feature usage rate'
            )
        
        # Feature 6: Activity recency score (exponential decay)
        if 'days_since_last_activity' in enhanced_df.columns:
            enhanced_df['activity_recency_score'] = np.exp(
                -enhanced_df['days_since_last_activity'] / 30
            )
            feature_report.add_feature(
                'activity_recency_score',
                'Exponential decay score based on activity recency'
            )
        
        # Feature 7: Customer value tier
        if 'total_value' in enhanced_df.columns:
            enhanced_df['value_tier'] = pd.qcut(
                enhanced_df['total_value'].rank(method='first'),
                q=4,
                labels=['low', 'medium', 'high', 'premium'],
                duplicates='drop'
            ).astype(str)
            feature_report.add_feature(
                'value_tier',
                'Customer value quartile ranking'
            )
        
        # Feature 8: Plan type encoding (if exists)
        if 'plan_type' in enhanced_df.columns:
            plan_value_mapping = {
                'free': 0, 'trial': 1, 'basic': 2, 
                'premium': 3, 'enterprise': 4, 'unknown': 1
            }
            enhanced_df['plan_value_score'] = enhanced_df['plan_type'].map(
                plan_value_mapping
            ).fillna(1)
            feature_report.add_feature(
                'plan_value_score',
                'Numeric encoding of plan type by value hierarchy'
            )
        
        # Feature 9: Engagement score (composite metric)
        engagement_components = []
        if 'feature_usage_rate' in enhanced_df.columns:
            engagement_components.append('feature_usage_rate')
        if 'activity_recency_score' in enhanced_df.columns:
            engagement_components.append('activity_recency_score')
        
        if engagement_components:
            # Normalize components to 0-1 scale
            for component in engagement_components:
                col_max = enhanced_df[component].max()
                if col_max > 0:
                    enhanced_df[f'{component}_normalized'] = enhanced_df[component] / col_max
            
            # Calculate composite engagement score
            normalized_cols = [f'{comp}_normalized' for comp in engagement_components]
            enhanced_df['engagement_score'] = enhanced_df[normalized_cols].mean(axis=1)
            
            # Clean up temporary columns
            enhanced_df = enhanced_df.drop(columns=normalized_cols)
            
            feature_report.add_feature(
                'engagement_score',
                f'Composite engagement score from: {", ".join(engagement_components)}'
            )
        
        return enhanced_df, feature_report
    
    async def _generate_revenue_features(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, FeatureEngineering]:
        """Generate features for revenue projection"""
        
        enhanced_df = df.copy()
        feature_report = FeatureEngineering(dataset_id=config.get('dataset_id', 'unknown'))
        
        # Ensure date column is datetime
        if 'date' in enhanced_df.columns:
            enhanced_df['date'] = pd.to_datetime(enhanced_df['date'])
            enhanced_df = enhanced_df.sort_values('date').reset_index(drop=True)
        
        # Feature 1: Moving averages
        if 'revenue' in enhanced_df.columns:
            for window in [7, 30, 90]:
                enhanced_df[f'revenue_ma_{window}'] = (
                    enhanced_df['revenue'].rolling(window=window, min_periods=1).mean()
                )
                feature_report.add_feature(
                    f'revenue_ma_{window}',
                    f'{window}-day moving average of revenue'
                )
        
        # Feature 2: Revenue growth rates
        if 'revenue' in enhanced_df.columns:
            enhanced_df['revenue_growth_1d'] = enhanced_df['revenue'].pct_change(1).fillna(0)
            enhanced_df['revenue_growth_7d'] = enhanced_df['revenue'].pct_change(7).fillna(0)
            enhanced_df['revenue_growth_30d'] = enhanced_df['revenue'].pct_change(30).fillna(0)
            
            for period in ['1d', '7d', '30d']:
                feature_report.add_feature(
                    f'revenue_growth_{period}',
                    f'{period} revenue growth rate'
                )
        
        # Feature 3: Seasonal features
        if 'date' in enhanced_df.columns:
            enhanced_df['day_of_week'] = enhanced_df['date'].dt.dayofweek
            enhanced_df['day_of_month'] = enhanced_df['date'].dt.day
            enhanced_df['day_of_year'] = enhanced_df['date'].dt.dayofyear
            enhanced_df['month'] = enhanced_df['date'].dt.month
            enhanced_df['quarter'] = enhanced_df['date'].dt.quarter
            enhanced_df['is_weekend'] = (enhanced_df['day_of_week'] >= 5).astype(int)
            
            seasonal_features = [
                'day_of_week', 'day_of_month', 'day_of_year', 
                'month', 'quarter', 'is_weekend'
            ]
            for feature in seasonal_features:
                feature_report.add_feature(feature, f'Seasonal component: {feature}')
        
        # Feature 4: Revenue volatility
        if 'revenue' in enhanced_df.columns:
            for window in [7, 30]:
                enhanced_df[f'revenue_volatility_{window}'] = (
                    enhanced_df['revenue'].rolling(window=window, min_periods=1).std().fillna(0)
                )
                feature_report.add_feature(
                    f'revenue_volatility_{window}',
                    f'{window}-day revenue volatility (standard deviation)'
                )
        
        return enhanced_df, feature_report
    
    async def _generate_price_features(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, FeatureEngineering]:
        """Generate features for price optimization"""
        
        enhanced_df = df.copy()
        feature_report = FeatureEngineering(dataset_id=config.get('dataset_id', 'unknown'))
        
        # Feature 1: Revenue (price * quantity)
        if all(col in enhanced_df.columns for col in ['price', 'quantity_sold']):
            enhanced_df['revenue'] = enhanced_df['price'] * enhanced_df['quantity_sold']
            feature_report.add_feature('revenue', 'Total revenue (price * quantity_sold)')
        
        # Feature 2: Price elasticity indicators
        if 'product_id' in enhanced_df.columns:
            # Group by product for product-level features
            enhanced_df = enhanced_df.sort_values(['product_id', 'date']).reset_index(drop=True)
            
            # Price changes
            enhanced_df['price_change'] = enhanced_df.groupby('product_id')['price'].pct_change().fillna(0)
            enhanced_df['quantity_change'] = enhanced_df.groupby('product_id')['quantity_sold'].pct_change().fillna(0)
            
            feature_report.add_feature('price_change', 'Period-over-period price change rate')
            feature_report.add_feature('quantity_change', 'Period-over-period quantity change rate')
            
            # Demand trend
            enhanced_df['quantity_trend'] = enhanced_df.groupby('product_id')['quantity_sold'].rolling(
                window=7, min_periods=1
            ).mean().reset_index(level=0, drop=True)
            feature_report.add_feature('quantity_trend', '7-day moving average of quantity sold')
        
        # Feature 3: Price positioning
        if 'price' in enhanced_df.columns:
            # Overall price percentiles
            enhanced_df['price_percentile'] = enhanced_df['price'].rank(pct=True)
            feature_report.add_feature('price_percentile', 'Price percentile across all products')
            
            # Price tier
            enhanced_df['price_tier'] = pd.qcut(
                enhanced_df['price'].rank(method='first'),
                q=3,
                labels=['low', 'medium', 'high'],
                duplicates='drop'
            ).astype(str)
            feature_report.add_feature('price_tier', 'Price tier (low/medium/high)')
        
        return enhanced_df, feature_report


class DataPreprocessingService:
    """
    Comprehensive data preprocessing service for ML pipelines
    
    Features:
    - Missing value imputation
    - Outlier detection and treatment
    - Feature scaling and encoding
    - Automated feature engineering
    - Data type optimization
    - Memory efficiency optimizations
    """
    
    def __init__(self):
        self.feature_engine = FeatureEngineeringEngine()
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
        logger.info("DataPreprocessingService initialized")
    
    async def preprocess_dataset(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        blueprint_name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, ProcessingResult]:
        """
        Main preprocessing pipeline
        
        Args:
            df: Input dataframe
            dataset_id: Unique dataset identifier
            blueprint_name: Target blueprint name
            config: Processing configuration
            
        Returns:
            Tuple of (processed_dataframe, processing_result)
        """
        config = config or {}
        start_time = datetime.utcnow()
        
        result = ProcessingResult(
            dataset_id=dataset_id,
            operation="preprocessing",
            status=ProcessingStatus.IN_PROGRESS,
            start_time=start_time,
            rows_processed=len(df)
        )
        
        try:
            logger.info(
                "Starting dataset preprocessing",
                dataset_id=dataset_id,
                blueprint_name=blueprint_name,
                input_shape=df.shape
            )
            
            # Step 1: Data type optimization
            processed_df = await self._optimize_data_types(df)
            logger.debug("Data type optimization completed", dataset_id=dataset_id)
            
            # Step 2: Handle missing values
            processed_df, missing_report = await self._handle_missing_values(
                processed_df, config.get('missing_value_strategy', 'auto')
            )
            result.metadata['missing_values_handled'] = missing_report
            logger.debug("Missing value handling completed", dataset_id=dataset_id)
            
            # Step 3: Outlier detection and treatment
            processed_df, outlier_report = await self._handle_outliers(
                processed_df, config.get('outlier_method', 'iqr')
            )
            result.metadata['outliers_handled'] = outlier_report
            logger.debug("Outlier handling completed", dataset_id=dataset_id)
            
            # Step 4: Feature engineering
            processed_df, feature_report = await self.feature_engine.generate_features(
                processed_df, blueprint_name, {'dataset_id': dataset_id, **config}
            )
            result.metadata['features_engineered'] = {
                'features_created': feature_report.features_created,
                'transformations': feature_report.transformations_applied
            }
            logger.debug("Feature engineering completed", dataset_id=dataset_id)
            
            # Step 5: Encode categorical variables
            processed_df, encoding_report = await self._encode_categorical_variables(
                processed_df, config.get('encoding_strategy', 'auto')
            )
            result.metadata['categorical_encoding'] = encoding_report
            logger.debug("Categorical encoding completed", dataset_id=dataset_id)
            
            # Step 6: Scale numerical features
            processed_df, scaling_report = await self._scale_numerical_features(
                processed_df, config.get('scaling_method', 'robust')
            )
            result.metadata['feature_scaling'] = scaling_report
            logger.debug("Feature scaling completed", dataset_id=dataset_id)
            
            # Step 7: Final optimization
            processed_df = await self._final_optimization(processed_df)
            
            result.mark_completed(rows_output=len(processed_df))
            result.metadata['final_shape'] = list(processed_df.shape)
            result.metadata['memory_reduction_mb'] = (
                df.memory_usage(deep=True).sum() - processed_df.memory_usage(deep=True).sum()
            ) / 1024 / 1024
            
            logger.info(
                "Dataset preprocessing completed successfully",
                dataset_id=dataset_id,
                input_shape=df.shape,
                output_shape=processed_df.shape,
                duration_seconds=result.duration_seconds,
                features_created=len(feature_report.features_created)
            )
            
            return processed_df, result
            
        except Exception as e:
            result.mark_failed(f"Preprocessing failed: {str(e)}")
            
            logger.error(
                "Dataset preprocessing failed",
                dataset_id=dataset_id,
                error=str(e),
                duration_seconds=result.duration_seconds,
                exc_info=True
            )
            
            raise PreprocessingError(f"Preprocessing failed for {dataset_id}: {str(e)}")
    
    async def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency"""
        optimized_df = df.copy()
        
        # Optimize integer columns
        int_cols = optimized_df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            col_min = optimized_df[col].min()
            col_max = optimized_df[col].max()
            
            if col_min >= 0:  # Unsigned integers
                if col_max < 255:
                    optimized_df[col] = optimized_df[col].astype('uint8')
                elif col_max < 65535:
                    optimized_df[col] = optimized_df[col].astype('uint16')
                elif col_max < 4294967295:
                    optimized_df[col] = optimized_df[col].astype('uint32')
            else:  # Signed integers
                if col_min >= -128 and col_max <= 127:
                    optimized_df[col] = optimized_df[col].astype('int8')
                elif col_min >= -32768 and col_max <= 32767:
                    optimized_df[col] = optimized_df[col].astype('int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    optimized_df[col] = optimized_df[col].astype('int32')
        
        # Optimize float columns
        float_cols = optimized_df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            # Try float32 if precision is sufficient
            if (optimized_df[col].astype('float32') == optimized_df[col]).all():
                optimized_df[col] = optimized_df[col].astype('float32')
        
        # Convert object columns to categorical if beneficial
        obj_cols = optimized_df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            num_unique = optimized_df[col].nunique()
            if num_unique / len(optimized_df) < 0.5:  # Less than 50% unique values
                optimized_df[col] = optimized_df[col].astype('category')
        
        return optimized_df
    
    async def _handle_missing_values(
        self, 
        df: pd.DataFrame, 
        strategy: str = 'auto'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values using various strategies"""
        
        processed_df = df.copy()
        missing_report = {
            'columns_with_missing': [],
            'strategies_applied': {},
            'imputation_summary': {}
        }
        
        # Identify columns with missing values
        missing_cols = processed_df.columns[processed_df.isnull().any()].tolist()
        missing_report['columns_with_missing'] = missing_cols
        
        if not missing_cols:
            return processed_df, missing_report
        
        for col in missing_cols:
            missing_count = processed_df[col].isnull().sum()
            missing_ratio = missing_count / len(processed_df)
            
            # Determine strategy per column
            if strategy == 'auto':
                if missing_ratio > 0.5:
                    # Too many missing values - drop column
                    processed_df = processed_df.drop(columns=[col])
                    applied_strategy = 'drop_column'
                elif processed_df[col].dtype in ['object', 'category']:
                    # Categorical - use mode or 'unknown'
                    mode_value = processed_df[col].mode()
                    fill_value = mode_value[0] if len(mode_value) > 0 else 'unknown'
                    processed_df[col] = processed_df[col].fillna(fill_value)
                    applied_strategy = f'mode_fill_{fill_value}'
                else:
                    # Numerical - use median
                    median_value = processed_df[col].median()
                    processed_df[col] = processed_df[col].fillna(median_value)
                    applied_strategy = f'median_fill_{median_value}'
            
            else:
                # Apply specified strategy
                if strategy == 'drop':
                    processed_df = processed_df.dropna(subset=[col])
                    applied_strategy = 'drop_rows'
                elif strategy == 'mean':
                    if pd.api.types.is_numeric_dtype(processed_df[col]):
                        processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
                        applied_strategy = 'mean_fill'
                elif strategy == 'median':
                    if pd.api.types.is_numeric_dtype(processed_df[col]):
                        processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                        applied_strategy = 'median_fill'
                elif strategy == 'mode':
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
                    applied_strategy = 'mode_fill'
                else:
                    applied_strategy = 'no_action'
            
            missing_report['strategies_applied'][col] = applied_strategy
            missing_report['imputation_summary'][col] = {
                'original_missing_count': missing_count,
                'missing_ratio': missing_ratio,
                'strategy': applied_strategy
            }
        
        return processed_df, missing_report
    
    async def _handle_outliers(
        self, 
        df: pd.DataFrame, 
        method: str = 'iqr'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect and handle outliers in numerical columns"""
        
        processed_df = df.copy()
        outlier_report = {
            'method': method,
            'columns_processed': [],
            'outliers_detected': {},
            'outliers_treated': {}
        }
        
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if processed_df[col].nunique() < 2:  # Skip constants
                continue
            
            outlier_report['columns_processed'].append(col)
            
            if method == 'iqr':
                Q1 = processed_df[col].quantile(0.25)
                Q3 = processed_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((processed_df[col] - processed_df[col].mean()) / processed_df[col].std())
                outlier_mask = z_scores > 3
                
            else:
                continue  # Unknown method
            
            outlier_count = outlier_mask.sum()
            outlier_report['outliers_detected'][col] = int(outlier_count)
            
            if outlier_count > 0:
                # Cap outliers instead of removing them
                if method == 'iqr':
                    processed_df.loc[processed_df[col] < lower_bound, col] = lower_bound
                    processed_df.loc[processed_df[col] > upper_bound, col] = upper_bound
                    treatment = f'capped_to_bounds_{lower_bound:.2f}_{upper_bound:.2f}'
                else:
                    # Z-score method - cap to 3 standard deviations
                    mean_val = processed_df[col].mean()
                    std_val = processed_df[col].std()
                    lower_cap = mean_val - 3 * std_val
                    upper_cap = mean_val + 3 * std_val
                    processed_df.loc[processed_df[col] < lower_cap, col] = lower_cap
                    processed_df.loc[processed_df[col] > upper_cap, col] = upper_cap
                    treatment = f'capped_to_3_std_{lower_cap:.2f}_{upper_cap:.2f}'
                
                outlier_report['outliers_treated'][col] = treatment
        
        return processed_df, outlier_report
    
    async def _encode_categorical_variables(
        self, 
        df: pd.DataFrame, 
        strategy: str = 'auto'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Encode categorical variables"""
        
        processed_df = df.copy()
        encoding_report = {
            'strategy': strategy,
            'columns_encoded': {},
            'new_columns_created': []
        }
        
        categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            unique_count = processed_df[col].nunique()
            
            if strategy == 'auto':
                if unique_count <= 2:
                    # Binary encoding
                    le = LabelEncoder()
                    processed_df[col] = le.fit_transform(processed_df[col].fillna('missing'))
                    encoding_method = 'label_encoding'
                    
                elif unique_count <= 10:
                    # One-hot encoding for low cardinality
                    dummies = pd.get_dummies(processed_df[col], prefix=col, dummy_na=True)
                    processed_df = pd.concat([processed_df.drop(columns=[col]), dummies], axis=1)
                    encoding_method = 'one_hot_encoding'
                    encoding_report['new_columns_created'].extend(dummies.columns.tolist())
                    
                else:
                    # High cardinality - use target encoding or keep as is
                    le = LabelEncoder()
                    processed_df[col] = le.fit_transform(processed_df[col].fillna('missing'))
                    encoding_method = 'label_encoding_high_cardinality'
            
            encoding_report['columns_encoded'][col] = {
                'method': encoding_method,
                'unique_values_count': unique_count
            }
        
        return processed_df, encoding_report
    
    async def _scale_numerical_features(
        self, 
        df: pd.DataFrame, 
        method: str = 'robust'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Scale numerical features"""
        
        processed_df = df.copy()
        scaling_report = {
            'method': method,
            'columns_scaled': [],
            'scaling_parameters': {}
        }
        
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        
        # Skip binary columns (0/1)
        binary_cols = []
        for col in numeric_cols:
            unique_vals = processed_df[col].dropna().unique()
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                binary_cols.append(col)
        
        cols_to_scale = [col for col in numeric_cols if col not in binary_cols]
        
        if cols_to_scale:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                logger.warning(f"Unknown scaling method: {method}, using robust")
                scaler = RobustScaler()
            
            processed_df[cols_to_scale] = scaler.fit_transform(processed_df[cols_to_scale])
            
            scaling_report['columns_scaled'] = cols_to_scale
            scaling_report['scaling_parameters'] = {
                'scaler_type': type(scaler).__name__,
                'columns_count': len(cols_to_scale),
                'binary_columns_skipped': binary_cols
            }
        
        return processed_df, scaling_report
    
    async def _final_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final optimizations before returning processed data"""
        
        optimized_df = df.copy()
        
        # Remove any columns that are all NaN
        optimized_df = optimized_df.dropna(axis=1, how='all')
        
        # Remove duplicate columns
        optimized_df = optimized_df.loc[:, ~optimized_df.columns.duplicated()]
        
        # Reset index
        optimized_df = optimized_df.reset_index(drop=True)
        
        # Final data type optimization
        optimized_df = await self._optimize_data_types(optimized_df)
        
        return optimized_df