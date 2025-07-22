"""
Data schemas and validation rules for Tracer Framework blueprints

Defines comprehensive data schemas for different ML blueprints using both
marshmallow and jsonschema for flexible validation approaches.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from marshmallow import Schema, fields, ValidationError, validate, post_load, pre_load
import jsonschema
from jsonschema import Draft7Validator
import structlog

from .models import ColumnSpec, DataType, DatasetMetadata

logger = structlog.get_logger(__name__)


class ChurnDetectionSchema(Schema):
    """
    Marshmallow schema for Customer Churn Detection blueprint
    
    Required fields align with blueprint specification:
    - customer_id: Unique customer identifier  
    - last_activity_date: Most recent interaction timestamp
    - account_created_date: Registration timestamp
    - total_value: Lifetime spend or usage metrics
    
    Optional enhancement fields:
    - support_tickets: Issue frequency data
    - feature_usage_count: Product engagement metrics  
    - plan_type: Subscription tier classification
    """
    
    # Required core fields
    customer_id = fields.Str(
        required=True, 
        validate=validate.Length(min=1, max=100),
        error_messages={'required': 'customer_id is required for churn detection'}
    )
    
    last_activity_date = fields.DateTime(
        required=True,
        error_messages={'required': 'last_activity_date is required to calculate recency'}
    )
    
    account_created_date = fields.DateTime(
        required=True,
        error_messages={'required': 'account_created_date is required to calculate tenure'}
    )
    
    total_value = fields.Float(
        required=True,
        validate=validate.Range(min=0),
        error_messages={
            'required': 'total_value is required to assess customer worth',
            'invalid': 'total_value must be a positive number'
        }
    )
    
    # Optional enhancement fields
    support_tickets = fields.Int(
        missing=0,
        validate=validate.Range(min=0),
        error_messages={'invalid': 'support_tickets must be a non-negative integer'}
    )
    
    feature_usage_count = fields.Int(
        missing=0,
        validate=validate.Range(min=0),
        error_messages={'invalid': 'feature_usage_count must be a non-negative integer'}
    )
    
    plan_type = fields.Str(
        missing="unknown",
        validate=validate.OneOf([
            "basic", "premium", "enterprise", "trial", "free", "unknown"
        ]),
        error_messages={'invalid': 'plan_type must be one of: basic, premium, enterprise, trial, free, unknown'}
    )
    
    # Optional additional enrichment fields
    login_frequency = fields.Float(
        missing=None,
        validate=validate.Range(min=0),
        allow_none=True
    )
    
    payment_failures = fields.Int(
        missing=0,
        validate=validate.Range(min=0)
    )
    
    referral_count = fields.Int(
        missing=0,
        validate=validate.Range(min=0)
    )
    
    @pre_load
    def preprocess_data(self, data, **kwargs):
        """Preprocess data before validation"""
        # Handle common date formats
        date_fields = ['last_activity_date', 'account_created_date']
        for field in date_fields:
            if field in data and isinstance(data[field], str):
                # Try to parse common date formats
                try:
                    # Handle ISO format
                    if 'T' in data[field]:
                        data[field] = datetime.fromisoformat(data[field].replace('Z', '+00:00'))
                    else:
                        # Handle date-only format
                        data[field] = datetime.strptime(data[field], '%Y-%m-%d')
                except ValueError:
                    # Let marshmallow handle the validation error
                    pass
        
        # Normalize string fields
        string_fields = ['customer_id', 'plan_type']
        for field in string_fields:
            if field in data and isinstance(data[field], str):
                data[field] = data[field].strip().lower() if field == 'plan_type' else data[field].strip()
        
        return data
    
    @post_load
    def validate_business_rules(self, data, **kwargs):
        """Apply business rule validations after schema validation"""
        
        # Business Rule 1: Account creation must precede last activity
        if data['account_created_date'] > data['last_activity_date']:
            raise ValidationError({
                'account_created_date': ['Account creation date cannot be after last activity date']
            })
        
        # Business Rule 2: Reasonable account age limits (not older than 20 years)
        account_age_days = (datetime.now() - data['account_created_date']).days
        if account_age_days > 20 * 365:
            raise ValidationError({
                'account_created_date': [f'Account age of {account_age_days} days seems unreasonable']
            })
        
        if account_age_days < 0:
            raise ValidationError({
                'account_created_date': ['Account creation date cannot be in the future']
            })
        
        # Business Rule 3: Recent activity limits (not in the future)
        activity_age_days = (datetime.now() - data['last_activity_date']).days
        if activity_age_days < 0:
            raise ValidationError({
                'last_activity_date': ['Last activity date cannot be in the future']
            })
        
        # Business Rule 4: Value consistency checks
        if data['total_value'] > 1000000:  # $1M limit for sanity
            logger.warning(
                "High customer value detected",
                customer_id=data.get('customer_id'),
                total_value=data['total_value']
            )
        
        # Business Rule 5: Logical consistency for engagement metrics
        if data.get('feature_usage_count', 0) > 0 and data.get('login_frequency', 0) == 0:
            logger.warning(
                "Feature usage without login frequency may indicate data quality issue",
                customer_id=data.get('customer_id')
            )
        
        return data


class RevenueProjectionSchema(Schema):
    """
    Schema for Revenue Projection blueprint
    
    Time series data for revenue forecasting
    """
    
    date = fields.Date(
        required=True,
        error_messages={'required': 'date is required for time series forecasting'}
    )
    
    revenue = fields.Float(
        required=True,
        validate=validate.Range(min=0),
        error_messages={
            'required': 'revenue is required for forecasting',
            'invalid': 'revenue must be a positive number'
        }
    )
    
    customer_count = fields.Int(
        missing=None,
        validate=validate.Range(min=0),
        allow_none=True
    )
    
    # Optional enhancement fields
    marketing_spend = fields.Float(missing=0, validate=validate.Range(min=0))
    product_launches = fields.Int(missing=0, validate=validate.Range(min=0))
    seasonal_factor = fields.Float(missing=1.0, validate=validate.Range(min=0))
    
    @post_load
    def validate_business_rules(self, data, **kwargs):
        """Business rule validation for revenue data"""
        
        # Ensure date is not too far in the future
        if data['date'] > datetime.now().date() + timedelta(days=1):
            raise ValidationError({
                'date': ['Revenue date cannot be in the future']
            })
        
        # Sanity check for revenue amounts
        if data['revenue'] > 10000000:  # $10M daily revenue limit
            logger.warning("Unusually high daily revenue detected", revenue=data['revenue'])
        
        return data


class PriceOptimizationSchema(Schema):
    """
    Schema for Price Optimization blueprint
    
    Historical pricing and sales data
    """
    
    product_id = fields.Str(
        required=True,
        validate=validate.Length(min=1, max=100)
    )
    
    price = fields.Float(
        required=True,
        validate=validate.Range(min=0.01)  # Must be positive
    )
    
    quantity_sold = fields.Int(
        required=True,
        validate=validate.Range(min=0)
    )
    
    date = fields.Date(required=True)
    
    # Optional fields
    cost_per_unit = fields.Float(missing=None, validate=validate.Range(min=0), allow_none=True)
    competitor_price = fields.Float(missing=None, validate=validate.Range(min=0), allow_none=True)
    promotion_active = fields.Bool(missing=False)
    
    @post_load
    def validate_business_rules(self, data, **kwargs):
        """Business rule validation for pricing data"""
        
        # Ensure cost doesn't exceed price (if provided)
        if data.get('cost_per_unit') and data['cost_per_unit'] > data['price']:
            raise ValidationError({
                'cost_per_unit': ['Cost per unit cannot exceed selling price']
            })
        
        # Reasonable price limits
        if data['price'] > 100000:  # $100k per unit limit
            logger.warning("Very high unit price detected", 
                         product_id=data['product_id'], 
                         price=data['price'])
        
        return data


# JSON Schema definitions for alternative validation approach
CHURN_DETECTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "customer_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": 100,
            "description": "Unique customer identifier"
        },
        "last_activity_date": {
            "type": "string",
            "format": "date-time",
            "description": "Most recent customer interaction timestamp"
        },
        "account_created_date": {
            "type": "string", 
            "format": "date-time",
            "description": "Customer account creation timestamp"
        },
        "total_value": {
            "type": "number",
            "minimum": 0,
            "description": "Lifetime customer value or usage metrics"
        },
        "support_tickets": {
            "type": "integer",
            "minimum": 0,
            "default": 0,
            "description": "Number of support tickets submitted"
        },
        "feature_usage_count": {
            "type": "integer",
            "minimum": 0,
            "default": 0,
            "description": "Count of product features used"
        },
        "plan_type": {
            "type": "string",
            "enum": ["basic", "premium", "enterprise", "trial", "free", "unknown"],
            "default": "unknown",
            "description": "Customer subscription plan type"
        }
    },
    "required": ["customer_id", "last_activity_date", "account_created_date", "total_value"],
    "additionalProperties": True
}

REVENUE_PROJECTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "date": {
            "type": "string",
            "format": "date",
            "description": "Date for revenue record"
        },
        "revenue": {
            "type": "number",
            "minimum": 0,
            "description": "Revenue amount for the date"
        },
        "customer_count": {
            "type": ["integer", "null"],
            "minimum": 0,
            "description": "Number of active customers"
        }
    },
    "required": ["date", "revenue"],
    "additionalProperties": True
}

PRICE_OPTIMIZATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "product_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": 100
        },
        "price": {
            "type": "number",
            "minimum": 0.01
        },
        "quantity_sold": {
            "type": "integer",
            "minimum": 0
        },
        "date": {
            "type": "string",
            "format": "date"
        }
    },
    "required": ["product_id", "price", "quantity_sold", "date"],
    "additionalProperties": True
}


class SchemaRegistry:
    """
    Central registry for all data schemas used in Tracer blueprints
    
    Provides both marshmallow and JSON schema validation options
    """
    
    def __init__(self):
        # Marshmallow schemas
        self.marshmallow_schemas = {
            'customer_churn': ChurnDetectionSchema(),
            'revenue_projection': RevenueProjectionSchema(), 
            'price_optimization': PriceOptimizationSchema()
        }
        
        # JSON schemas
        self.json_schemas = {
            'customer_churn': CHURN_DETECTION_JSON_SCHEMA,
            'revenue_projection': REVENUE_PROJECTION_JSON_SCHEMA,
            'price_optimization': PRICE_OPTIMIZATION_JSON_SCHEMA
        }
        
        # Blueprint column specifications
        self.column_specs = {
            'customer_churn': self._get_churn_column_specs(),
            'revenue_projection': self._get_revenue_column_specs(),
            'price_optimization': self._get_price_column_specs()
        }
        
        logger.info(
            "SchemaRegistry initialized",
            schemas_loaded=list(self.marshmallow_schemas.keys())
        )
    
    def _get_churn_column_specs(self) -> Dict[str, ColumnSpec]:
        """Get column specifications for churn detection"""
        return {
            'customer_id': ColumnSpec(
                name='customer_id',
                data_type=DataType.STRING,
                required=True,
                nullable=False,
                description='Unique customer identifier',
                constraints={'max_length': 100}
            ),
            'last_activity_date': ColumnSpec(
                name='last_activity_date', 
                data_type=DataType.DATETIME,
                required=True,
                nullable=False,
                description='Most recent customer interaction timestamp'
            ),
            'account_created_date': ColumnSpec(
                name='account_created_date',
                data_type=DataType.DATETIME, 
                required=True,
                nullable=False,
                description='Customer account creation timestamp'
            ),
            'total_value': ColumnSpec(
                name='total_value',
                data_type=DataType.FLOAT,
                required=True,
                nullable=False,
                description='Lifetime customer value',
                constraints={'min': 0}
            ),
            'support_tickets': ColumnSpec(
                name='support_tickets',
                data_type=DataType.INTEGER,
                required=False,
                nullable=True,
                description='Number of support tickets',
                constraints={'min': 0}
            ),
            'feature_usage_count': ColumnSpec(
                name='feature_usage_count',
                data_type=DataType.INTEGER,
                required=False,
                nullable=True,
                description='Count of features used',
                constraints={'min': 0}
            ),
            'plan_type': ColumnSpec(
                name='plan_type',
                data_type=DataType.CATEGORICAL,
                required=False,
                nullable=True,
                description='Subscription plan type',
                constraints={'allowed_values': ['basic', 'premium', 'enterprise', 'trial', 'free', 'unknown']}
            )
        }
    
    def _get_revenue_column_specs(self) -> Dict[str, ColumnSpec]:
        """Get column specifications for revenue projection"""
        return {
            'date': ColumnSpec(
                name='date',
                data_type=DataType.DATETIME,
                required=True,
                nullable=False,
                description='Date for revenue record'
            ),
            'revenue': ColumnSpec(
                name='revenue',
                data_type=DataType.FLOAT,
                required=True,
                nullable=False,
                description='Revenue amount',
                constraints={'min': 0}
            ),
            'customer_count': ColumnSpec(
                name='customer_count',
                data_type=DataType.INTEGER,
                required=False,
                nullable=True,
                description='Number of active customers',
                constraints={'min': 0}
            )
        }
    
    def _get_price_column_specs(self) -> Dict[str, ColumnSpec]:
        """Get column specifications for price optimization"""
        return {
            'product_id': ColumnSpec(
                name='product_id',
                data_type=DataType.STRING,
                required=True,
                nullable=False,
                description='Product identifier',
                constraints={'max_length': 100}
            ),
            'price': ColumnSpec(
                name='price',
                data_type=DataType.FLOAT,
                required=True,
                nullable=False,
                description='Product price',
                constraints={'min': 0.01}
            ),
            'quantity_sold': ColumnSpec(
                name='quantity_sold',
                data_type=DataType.INTEGER,
                required=True,
                nullable=False,
                description='Quantity sold',
                constraints={'min': 0}
            ),
            'date': ColumnSpec(
                name='date',
                data_type=DataType.DATETIME,
                required=True,
                nullable=False,
                description='Sale date'
            )
        }
    
    def get_schema(self, blueprint_name: str, schema_type: str = 'marshmallow'):
        """
        Get schema for a blueprint
        
        Args:
            blueprint_name: Name of the blueprint
            schema_type: Either 'marshmallow' or 'jsonschema'
            
        Returns:
            Schema object or None if not found
        """
        if schema_type == 'marshmallow':
            return self.marshmallow_schemas.get(blueprint_name)
        elif schema_type == 'jsonschema':
            return self.json_schemas.get(blueprint_name)
        else:
            raise ValueError(f"Unknown schema type: {schema_type}")
    
    def get_column_specs(self, blueprint_name: str) -> Optional[Dict[str, ColumnSpec]]:
        """Get column specifications for a blueprint"""
        return self.column_specs.get(blueprint_name)
    
    def validate_with_jsonschema(self, data: dict, blueprint_name: str) -> List[str]:
        """
        Validate data using JSON schema
        
        Args:
            data: Data to validate
            blueprint_name: Name of the blueprint
            
        Returns:
            List of validation error messages
        """
        schema = self.json_schemas.get(blueprint_name)
        if not schema:
            return [f"Unknown blueprint: {blueprint_name}"]
        
        try:
            validator = Draft7Validator(schema)
            errors = []
            for error in validator.iter_errors(data):
                errors.append(f"{'.'.join(str(p) for p in error.path)}: {error.message}")
            return errors
        except Exception as e:
            return [f"Validation error: {str(e)}"]
    
    def get_required_columns(self, blueprint_name: str) -> List[str]:
        """Get list of required columns for a blueprint"""
        column_specs = self.get_column_specs(blueprint_name)
        if not column_specs:
            return []
        
        return [name for name, spec in column_specs.items() if spec.required]
    
    def get_optional_columns(self, blueprint_name: str) -> List[str]:
        """Get list of optional columns for a blueprint"""
        column_specs = self.get_column_specs(blueprint_name)
        if not column_specs:
            return []
        
        return [name for name, spec in column_specs.items() if not spec.required]
    
    def register_custom_schema(
        self, 
        blueprint_name: str, 
        marshmallow_schema: Schema = None,
        json_schema: dict = None,
        column_specs: Dict[str, ColumnSpec] = None
    ):
        """
        Register a custom schema for a new blueprint
        
        Args:
            blueprint_name: Name of the blueprint
            marshmallow_schema: Marshmallow schema instance
            json_schema: JSON schema dictionary
            column_specs: Column specifications
        """
        if marshmallow_schema:
            self.marshmallow_schemas[blueprint_name] = marshmallow_schema
        
        if json_schema:
            self.json_schemas[blueprint_name] = json_schema
        
        if column_specs:
            self.column_specs[blueprint_name] = column_specs
        
        logger.info(
            "Custom schema registered",
            blueprint_name=blueprint_name,
            has_marshmallow=marshmallow_schema is not None,
            has_json_schema=json_schema is not None,
            has_column_specs=column_specs is not None
        )


# Global schema registry instance
schema_registry = SchemaRegistry()