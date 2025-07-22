"""
Unit tests for machine learning model components.

Tests model training, prediction, evaluation, and model persistence
for customer churn detection pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Tuple

from tracer.core.ml_models import (
    ChurnPredictor,
    ModelTrainer,
    ModelEvaluator,
    FeaturePreprocessor,
    ModelPersistence,
    ModelMetrics
)

class TestFeaturePreprocessor:
    """Test cases for feature preprocessing."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create FeaturePreprocessor instance."""
        return FeaturePreprocessor()
    
    @pytest.fixture
    def sample_features_data(self) -> pd.DataFrame:
        """Sample feature data for testing."""
        return pd.DataFrame({
            'age': [25, 35, 45, 30, 28],
            'tenure': [12, 24, 36, 18, 15],
            'monthly_charges': [50.0, 75.5, 89.99, 65.25, 55.0],
            'total_charges': [600.0, 1812.0, 3239.64, 1174.5, 825.0],
            'contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year'],
            'payment_method': ['Credit card', 'Bank transfer', 'Electronic check', 'Mailed check', 'Credit card']
        })
    
    def test_numerical_feature_scaling(self, preprocessor: FeaturePreprocessor, sample_features_data: pd.DataFrame):
        """Test numerical feature scaling."""
        numerical_columns = ['age', 'tenure', 'monthly_charges', 'total_charges']
        
        scaled_data = preprocessor.scale_numerical_features(sample_features_data, numerical_columns)
        
        # Check that scaling was applied
        for col in numerical_columns:
            assert col in scaled_data.columns
            # Scaled features should have different values
            assert not np.array_equal(scaled_data[col].values, sample_features_data[col].values)
            # Check for reasonable scaling (mean close to 0, std close to 1 for standard scaling)
            if preprocessor.scaling_method == 'standard':
                assert abs(scaled_data[col].mean()) < 0.1
                assert abs(scaled_data[col].std() - 1.0) < 0.1
    
    def test_categorical_feature_encoding(self, preprocessor: FeaturePreprocessor, sample_features_data: pd.DataFrame):
        """Test categorical feature encoding."""
        categorical_columns = ['contract', 'payment_method']
        
        encoded_data = preprocessor.encode_categorical_features(sample_features_data, categorical_columns)
        
        # Check that original categorical columns are replaced with encoded versions
        for col in categorical_columns:
            # Original column should be removed or encoded columns should be added
            encoded_cols = [c for c in encoded_data.columns if c.startswith(f"{col}_")]
            assert len(encoded_cols) > 0 or col not in encoded_data.columns
    
    def test_feature_selection(self, preprocessor: FeaturePreprocessor):
        """Test feature selection functionality."""
        # Create data with some irrelevant features
        data = pd.DataFrame({
            'relevant_feature': np.random.rand(100),
            'target_correlated': np.random.rand(100),
            'noise_feature': np.random.rand(100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # Make target_correlated actually correlated with target
        data['target_correlated'] = data['target'] + np.random.rand(100) * 0.1
        
        selected_features = preprocessor.select_features(
            data.drop('target', axis=1), 
            data['target'], 
            method='correlation',
            k=2
        )
        
        assert len(selected_features.columns) <= 2
        assert 'target_correlated' in selected_features.columns
    
    def test_missing_value_handling(self, preprocessor: FeaturePreprocessor):
        """Test missing value handling."""
        data = pd.DataFrame({
            'numeric_col': [1, 2, np.nan, 4, 5],
            'categorical_col': ['A', 'B', None, 'A', 'B']
        })
        
        cleaned_data = preprocessor.handle_missing_values(data, strategy='median')
        
        assert cleaned_data['numeric_col'].isna().sum() == 0
        assert cleaned_data['categorical_col'].isna().sum() == 0
    
    def test_feature_engineering(self, preprocessor: FeaturePreprocessor, sample_features_data: pd.DataFrame):
        """Test feature engineering functionality."""
        engineered_data = preprocessor.engineer_features(sample_features_data)
        
        # Should create new features based on existing ones
        original_columns = set(sample_features_data.columns)
        new_columns = set(engineered_data.columns)
        
        # New features should be added
        assert len(new_columns) >= len(original_columns)
        
        # Check for common engineered features
        if 'charges_per_tenure' in engineered_data.columns:
            assert all(engineered_data['charges_per_tenure'] >= 0)


class TestModelTrainer:
    """Test cases for model training."""
    
    @pytest.fixture
    def trainer(self):
        """Create ModelTrainer instance."""
        return ModelTrainer()
    
    @pytest.fixture
    def training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate training data."""
        np.random.seed(42)
        n_samples = 1000
        
        X = pd.DataFrame({
            'age': np.random.randint(18, 80, n_samples),
            'tenure': np.random.randint(1, 72, n_samples),
            'monthly_charges': np.random.uniform(20, 120, n_samples),
            'total_charges': np.random.uniform(100, 8000, n_samples)
        })
        
        # Create target with some logic
        y = ((X['tenure'] < 12) & (X['monthly_charges'] > 80)).astype(int)
        
        return X, y
    
    def test_model_training(self, trainer: ModelTrainer, training_data: Tuple[pd.DataFrame, pd.Series]):
        """Test basic model training."""
        X, y = training_data
        
        model = trainer.train_model(X, y, model_type='xgboost')
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_cross_validation(self, trainer: ModelTrainer, training_data: Tuple[pd.DataFrame, pd.Series]):
        """Test cross-validation during training."""
        X, y = training_data
        
        cv_scores = trainer.cross_validate(X, y, model_type='xgboost', cv_folds=3)
        
        assert len(cv_scores) == 3
        assert all(0 <= score <= 1 for score in cv_scores)  # Assuming scores are between 0 and 1
    
    def test_hyperparameter_tuning(self, trainer: ModelTrainer, training_data: Tuple[pd.DataFrame, pd.Series]):
        """Test hyperparameter tuning."""
        X, y = training_data
        
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [50, 100]
        }
        
        best_model, best_params = trainer.tune_hyperparameters(
            X, y, 
            model_type='xgboost', 
            param_grid=param_grid,
            cv_folds=3
        )
        
        assert best_model is not None
        assert isinstance(best_params, dict)
        assert len(best_params) > 0
    
    def test_feature_importance(self, trainer: ModelTrainer, training_data: Tuple[pd.DataFrame, pd.Series]):
        """Test feature importance extraction."""
        X, y = training_data
        
        model = trainer.train_model(X, y, model_type='xgboost')
        importance = trainer.get_feature_importance(model, X.columns)
        
        assert isinstance(importance, dict)
        assert len(importance) == len(X.columns)
        assert all(col in importance for col in X.columns)
        assert all(isinstance(score, (int, float)) for score in importance.values())
    
    @pytest.mark.parametrize("model_type", ['xgboost', 'random_forest', 'logistic_regression'])
    def test_different_model_types(self, trainer: ModelTrainer, training_data: Tuple[pd.DataFrame, pd.Series], model_type: str):
        """Test training different model types."""
        X, y = training_data
        
        model = trainer.train_model(X, y, model_type=model_type)
        
        assert model is not None
        assert hasattr(model, 'predict')
        
        # Test prediction
        predictions = model.predict(X[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)


class TestChurnPredictor:
    """Test cases for churn prediction."""
    
    @pytest.fixture
    def predictor(self):
        """Create ChurnPredictor instance."""
        return ChurnPredictor()
    
    @pytest.fixture
    def mock_model(self):
        """Create mock trained model."""
        mock = Mock()
        mock.predict.return_value = np.array([0, 1, 0])
        mock.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        return mock
    
    def test_single_prediction(self, predictor: ChurnPredictor, mock_model: Mock):
        """Test single customer prediction."""
        predictor.model = mock_model
        
        customer_data = {
            'age': 35,
            'tenure': 24,
            'monthly_charges': 75.5,
            'total_charges': 1812.0
        }
        
        result = predictor.predict_single(customer_data)
        
        assert 'churn_probability' in result
        assert 'churn_prediction' in result
        assert 'risk_category' in result
        assert 0 <= result['churn_probability'] <= 1
        assert result['churn_prediction'] in [0, 1]
    
    def test_batch_prediction(self, predictor: ChurnPredictor, mock_model: Mock):
        """Test batch customer predictions."""
        predictor.model = mock_model
        
        customers_data = pd.DataFrame({
            'age': [25, 35, 45],
            'tenure': [12, 24, 36],
            'monthly_charges': [50.0, 75.5, 89.99],
            'total_charges': [600.0, 1812.0, 3239.64]
        })
        
        results = predictor.predict_batch(customers_data)
        
        assert len(results) == 3
        assert all('churn_probability' in result for result in results)
        assert all('churn_prediction' in result for result in results)
        assert all('risk_category' in result for result in results)
    
    def test_risk_categorization(self, predictor: ChurnPredictor):
        """Test risk category assignment."""
        test_cases = [
            (0.1, 'Low'),
            (0.3, 'Medium'),
            (0.7, 'High'),
            (0.9, 'Critical')
        ]
        
        for probability, expected_category in test_cases:
            category = predictor._categorize_risk(probability)
            assert category == expected_category
    
    def test_prediction_with_missing_model(self, predictor: ChurnPredictor):
        """Test prediction behavior when model is not loaded."""
        predictor.model = None
        
        customer_data = {'age': 35, 'tenure': 24}
        
        with pytest.raises(ValueError, match="Model not loaded"):
            predictor.predict_single(customer_data)
    
    def test_prediction_with_invalid_data(self, predictor: ChurnPredictor, mock_model: Mock):
        """Test prediction with invalid input data."""
        predictor.model = mock_model
        
        # Missing required features
        invalid_data = {'age': 35}  # Missing other required features
        
        with pytest.raises(ValueError):
            predictor.predict_single(invalid_data)


class TestModelEvaluator:
    """Test cases for model evaluation."""
    
    @pytest.fixture
    def evaluator(self):
        """Create ModelEvaluator instance."""
        return ModelEvaluator()
    
    @pytest.fixture
    def test_predictions(self):
        """Generate test predictions and actual values."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        y_proba = np.array([0.1, 0.6, 0.8, 0.9, 0.2, 0.4, 0.1, 0.7])
        
        return y_true, y_pred, y_proba
    
    def test_classification_metrics(self, evaluator: ModelEvaluator, test_predictions: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """Test classification metrics calculation."""
        y_true, y_pred, y_proba = test_predictions
        
        metrics = evaluator.calculate_classification_metrics(y_true, y_pred, y_proba)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'auc_roc' in metrics
        assert 'auc_pr' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['auc_roc'] <= 1
        assert 0 <= metrics['auc_pr'] <= 1
    
    def test_confusion_matrix(self, evaluator: ModelEvaluator, test_predictions: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """Test confusion matrix calculation."""
        y_true, y_pred, _ = test_predictions
        
        cm = evaluator.calculate_confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_true)
        assert all(cm.flatten() >= 0)  # All values should be non-negative
    
    def test_business_metrics(self, evaluator: ModelEvaluator, test_predictions: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """Test business-specific metrics calculation."""
        y_true, y_pred, y_proba = test_predictions
        
        # Mock business parameters
        cost_per_false_positive = 10  # Cost of incorrectly targeting non-churners
        cost_per_false_negative = 100  # Cost of missing actual churners
        revenue_per_customer = 1000
        
        business_metrics = evaluator.calculate_business_metrics(
            y_true, y_pred, 
            cost_false_positive=cost_per_false_positive,
            cost_false_negative=cost_per_false_negative,
            revenue_per_customer=revenue_per_customer
        )
        
        assert 'total_cost' in business_metrics
        assert 'potential_revenue_saved' in business_metrics
        assert 'roi' in business_metrics
    
    def test_model_fairness_evaluation(self, evaluator: ModelEvaluator):
        """Test model fairness evaluation across different groups."""
        # Create synthetic data with potential bias
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1] * 2)
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1] * 2)
        sensitive_attribute = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'] * 2)
        
        fairness_metrics = evaluator.evaluate_fairness(y_true, y_pred, sensitive_attribute)
        
        assert 'demographic_parity' in fairness_metrics
        assert 'equalized_odds' in fairness_metrics
        assert 'calibration' in fairness_metrics
    
    def test_feature_importance_analysis(self, evaluator: ModelEvaluator):
        """Test feature importance analysis."""
        feature_importance = {
            'tenure': 0.3,
            'monthly_charges': 0.25,
            'total_charges': 0.2,
            'age': 0.15,
            'contract': 0.1
        }
        
        analysis = evaluator.analyze_feature_importance(feature_importance)
        
        assert 'top_features' in analysis
        assert 'cumulative_importance' in analysis
        assert len(analysis['top_features']) > 0


class TestModelPersistence:
    """Test cases for model persistence."""
    
    @pytest.fixture
    def persistence(self, temp_dir: Path):
        """Create ModelPersistence instance."""
        return ModelPersistence(model_dir=temp_dir)
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for persistence testing."""
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Create minimal training data to fit the model
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        
        return model
    
    def test_model_saving(self, persistence: ModelPersistence, mock_model, temp_dir: Path):
        """Test model saving functionality."""
        model_name = "test_model"
        version = "1.0.0"
        
        model_path = persistence.save_model(mock_model, model_name, version)
        
        assert model_path.exists()
        assert model_name in str(model_path)
        assert version in str(model_path)
    
    def test_model_loading(self, persistence: ModelPersistence, mock_model, temp_dir: Path):
        """Test model loading functionality."""
        model_name = "test_model"
        version = "1.0.0"
        
        # Save model first
        persistence.save_model(mock_model, model_name, version)
        
        # Load model
        loaded_model = persistence.load_model(model_name, version)
        
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')
        
        # Test that loaded model produces same predictions
        test_data = np.array([[1, 2], [3, 4]])
        original_pred = mock_model.predict(test_data)
        loaded_pred = loaded_model.predict(test_data)
        
        assert np.array_equal(original_pred, loaded_pred)
    
    def test_model_metadata_saving(self, persistence: ModelPersistence, mock_model, temp_dir: Path):
        """Test saving model metadata."""
        model_name = "test_model"
        version = "1.0.0"
        metadata = {
            'training_date': '2024-01-01',
            'metrics': {'accuracy': 0.85, 'f1_score': 0.82},
            'features': ['feature1', 'feature2'],
            'model_type': 'RandomForest'
        }
        
        persistence.save_model(mock_model, model_name, version, metadata=metadata)
        loaded_metadata = persistence.load_model_metadata(model_name, version)
        
        assert loaded_metadata is not None
        assert loaded_metadata['training_date'] == '2024-01-01'
        assert loaded_metadata['metrics']['accuracy'] == 0.85
        assert 'feature1' in loaded_metadata['features']
    
    def test_model_versioning(self, persistence: ModelPersistence, mock_model, temp_dir: Path):
        """Test model versioning functionality."""
        model_name = "versioned_model"
        
        # Save multiple versions
        versions = ["1.0.0", "1.1.0", "2.0.0"]
        for version in versions:
            persistence.save_model(mock_model, model_name, version)
        
        # List versions
        available_versions = persistence.list_model_versions(model_name)
        
        assert len(available_versions) == 3
        assert all(v in available_versions for v in versions)
        
        # Get latest version
        latest_version = persistence.get_latest_version(model_name)
        assert latest_version == "2.0.0"  # Assuming semantic versioning sort
    
    def test_model_cleanup(self, persistence: ModelPersistence, mock_model, temp_dir: Path):
        """Test model cleanup functionality."""
        model_name = "cleanup_test_model"
        version = "1.0.0"
        
        # Save model
        model_path = persistence.save_model(mock_model, model_name, version)
        assert model_path.exists()
        
        # Delete model
        persistence.delete_model(model_name, version)
        assert not model_path.exists()


@pytest.mark.integration
class TestMLPipelineIntegration:
    """Integration tests for ML pipeline components."""
    
    def test_end_to_end_ml_pipeline(self, sample_csv_data: Dict[str, Any], temp_dir: Path):
        """Test complete ML pipeline from data to prediction."""
        # Create test data
        df = pd.DataFrame(sample_csv_data)
        X = df.drop('churn', axis=1)
        y = df['churn']
        
        # Initialize components
        preprocessor = FeaturePreprocessor()
        trainer = ModelTrainer()
        evaluator = ModelEvaluator()
        predictor = ChurnPredictor()
        persistence = ModelPersistence(model_dir=temp_dir)
        
        # Step 1: Preprocess features
        X_processed = preprocessor.scale_numerical_features(
            X.select_dtypes(include=[np.number]), 
            X.select_dtypes(include=[np.number]).columns
        )
        
        # Step 2: Train model
        model = trainer.train_model(X_processed, y, model_type='xgboost')
        
        # Step 3: Evaluate model
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed)[:, 1]
        metrics = evaluator.calculate_classification_metrics(y, predictions, probabilities)
        
        assert metrics['accuracy'] >= 0  # Should have some accuracy
        
        # Step 4: Save model
        model_path = persistence.save_model(model, "integration_test_model", "1.0.0")
        assert model_path.exists()
        
        # Step 5: Load model and make predictions
        loaded_model = persistence.load_model("integration_test_model", "1.0.0")
        predictor.model = loaded_model
        
        # Test single prediction
        customer_data = X_processed.iloc[0].to_dict()
        result = predictor.predict_single(customer_data)
        
        assert 'churn_probability' in result
        assert 'churn_prediction' in result
        assert 'risk_category' in result


@pytest.mark.performance
class TestMLPerformance:
    """Performance tests for ML components."""
    
    def test_training_performance(self, performance_test_data: Dict):
        """Test model training performance with larger datasets."""
        # Generate larger dataset
        np.random.seed(42)
        n_samples = 10000
        n_features = 20
        
        X = pd.DataFrame(np.random.rand(n_samples, n_features))
        y = np.random.choice([0, 1], n_samples)
        
        trainer = ModelTrainer()
        
        import time
        start_time = time.time()
        
        model = trainer.train_model(X, y, model_type='xgboost')
        
        end_time = time.time()
        training_time = end_time - start_time
        
        assert training_time < performance_test_data['response_time_threshold']
        assert model is not None
    
    def test_prediction_performance(self, performance_test_data: Dict):
        """Test prediction performance."""
        # Create mock predictor with trained model
        predictor = ChurnPredictor()
        
        # Create mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
        
        predictor.model = mock_model
        
        # Test prediction speed
        customer_data = {
            'age': 35, 'tenure': 24, 'monthly_charges': 75.5, 'total_charges': 1812.0
        }
        
        import time
        start_time = time.time()
        
        # Make multiple predictions
        for _ in range(1000):
            predictor.predict_single(customer_data)
        
        end_time = time.time()
        avg_prediction_time = (end_time - start_time) / 1000
        
        assert avg_prediction_time < 0.01  # Less than 10ms per prediction