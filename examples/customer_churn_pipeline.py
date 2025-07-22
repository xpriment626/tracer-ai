"""
Customer Churn Detection Pipeline Example

This example demonstrates the complete data pipeline for customer churn detection
using the Tracer Framework's data engineering components.

Requirements:
- CSV file with customer data containing: customer_id, last_activity_date, 
  account_created_date, total_value (required fields)
- Optional fields: support_tickets, feature_usage_count, plan_type
"""

import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Import Tracer core components
from src.tracer.core import (
    DataPipeline,
    schema_registry,
    ProcessingStatus
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Main example function"""
    
    # Initialize the data pipeline
    pipeline = DataPipeline(max_concurrent_pipelines=2)
    
    print("🚀 Tracer Framework - Customer Churn Detection Pipeline")
    print("=" * 60)
    
    # 1. Show available blueprints
    print("\n1. Available Blueprints:")
    blueprints = pipeline.get_available_blueprints()
    for blueprint in blueprints:
        print(f"   - {blueprint}")
    
    # 2. Get blueprint information
    print("\n2. Customer Churn Blueprint Information:")
    churn_info = pipeline.get_blueprint_info('customer_churn')
    print(f"   Required columns: {churn_info['required_columns']}")
    print(f"   Optional columns: {churn_info['optional_columns']}")
    
    # 3. Create sample data for demonstration
    print("\n3. Creating Sample Dataset...")
    sample_data = create_sample_churn_data()
    sample_file = "sample_customer_data.csv"
    sample_data.to_csv(sample_file, index=False)
    print(f"   Created: {sample_file} with {len(sample_data)} records")
    
    # 4. Check file compatibility before processing
    print("\n4. Checking File Compatibility...")
    compatibility = await pipeline.validate_file_compatibility(
        file_path=sample_file,
        blueprint_name='customer_churn'
    )
    print(f"   Compatible: {compatibility['compatible']}")
    print(f"   Compatibility Score: {compatibility['score']:.2f}")
    if compatibility.get('recommendations'):
        print("   Recommendations:")
        for rec in compatibility['recommendations']:
            print(f"     - {rec}")
    
    # 5. Process the dataset
    if compatibility['compatible']:
        print("\n5. Processing Dataset...")
        
        try:
            # Configure preprocessing options
            preprocessing_config = {
                'missing_value_strategy': 'auto',
                'outlier_method': 'iqr',
                'encoding_strategy': 'auto',
                'scaling_method': 'robust'
            }
            
            # Process the file
            processed_df, pipeline_state = await pipeline.process_file(
                file_path=sample_file,
                blueprint_name='customer_churn',
                preprocessing_config=preprocessing_config
            )
            
            print(f"   ✅ Processing completed successfully!")
            print(f"   Pipeline ID: {pipeline_state.pipeline_id}")
            print(f"   Input shape: {compatibility['file_metadata']['rows']} x {compatibility['file_metadata']['columns']}")
            print(f"   Output shape: {processed_df.shape[0]} x {processed_df.shape[1]}")
            
            # 6. Show processing results
            print("\n6. Processing Results:")
            show_processing_results(pipeline_state)
            
            # 7. Show feature engineering results
            if pipeline_state.feature_engineering:
                print("\n7. Feature Engineering Results:")
                show_feature_engineering_results(pipeline_state.feature_engineering)
            
            # 8. Show sample of processed data
            print("\n8. Sample of Processed Data:")
            print(processed_df.head().to_string())
            
            # 9. Show data quality report
            if pipeline_state.validation_report:
                print("\n9. Data Quality Report:")
                show_data_quality_report(pipeline_state.validation_report)
            
            # 10. Save processed data
            output_file = "processed_customer_data.csv"
            processed_df.to_csv(output_file, index=False)
            print(f"\n10. Processed data saved to: {output_file}")
            
        except Exception as e:
            print(f"   ❌ Processing failed: {str(e)}")
            
            # Show pipeline status for debugging
            if 'pipeline_state' in locals():
                print(f"   Pipeline Status: {pipeline_state.status.value}")
                print(f"   Current Stage: {pipeline_state.current_stage}")
                if pipeline_state.metadata.get('error'):
                    print(f"   Error: {pipeline_state.metadata['error']}")
    
    else:
        print("\n❌ Dataset is not compatible with customer_churn blueprint")
        print("   Please check the recommendations above and update your dataset")
    
    # 11. Show pipeline summary
    print("\n11. Pipeline Summary:")
    summary = pipeline.get_summary()
    for key, value in summary.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "=" * 60)
    print("🎉 Pipeline example completed!")


def create_sample_churn_data(num_records: int = 1000) -> pd.DataFrame:
    """Create sample customer churn data for demonstration"""
    
    import random
    import numpy as np
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate sample data
    data = []
    base_date = datetime(2023, 1, 1)
    
    for i in range(num_records):
        # Account created date (random date in the past 2 years)
        days_ago = random.randint(30, 730)
        account_created = base_date - timedelta(days=days_ago)
        
        # Last activity date (between account creation and now)
        activity_days_ago = random.randint(0, min(days_ago, 90))
        last_activity = base_date - timedelta(days=activity_days_ago)
        
        # Customer value (log-normal distribution)
        total_value = np.random.lognormal(mean=5, sigma=1.5)
        
        # Plan type (categorical)
        plan_types = ['basic', 'premium', 'enterprise', 'trial', 'free']
        plan_weights = [0.4, 0.3, 0.15, 0.1, 0.05]
        plan_type = np.random.choice(plan_types, p=plan_weights)
        
        # Support tickets (Poisson distribution)
        support_tickets = np.random.poisson(lam=2)
        
        # Feature usage count (related to plan type and value)
        base_usage = {'free': 5, 'trial': 10, 'basic': 20, 'premium': 50, 'enterprise': 100}
        feature_usage = int(np.random.normal(
            loc=base_usage[plan_type], 
            scale=base_usage[plan_type] * 0.3
        ))
        feature_usage = max(0, feature_usage)  # No negative usage
        
        record = {
            'customer_id': f'cust_{i+1:06d}',
            'last_activity_date': last_activity.strftime('%Y-%m-%d'),
            'account_created_date': account_created.strftime('%Y-%m-%d'),
            'total_value': round(total_value, 2),
            'support_tickets': support_tickets,
            'feature_usage_count': feature_usage,
            'plan_type': plan_type
        }
        
        data.append(record)
    
    return pd.DataFrame(data)


def show_processing_results(pipeline_state):
    """Display processing results from pipeline state"""
    
    for stage, result in pipeline_state.results.items():
        print(f"   {stage.replace('_', ' ').title()}:")
        print(f"     Status: {result.status.value}")
        print(f"     Duration: {result.duration_seconds:.2f}s")
        print(f"     Rows: {result.rows_processed} → {result.rows_output}")
        
        if result.errors:
            print(f"     Errors: {len(result.errors)}")
        
        # Show stage-specific metadata
        if stage == 'ingestion' and 'file_size_mb' in result.metadata:
            print(f"     File Size: {result.metadata['file_size_mb']:.2f} MB")
        elif stage == 'validation' and 'data_quality_score' in result.metadata:
            print(f"     Quality Score: {result.metadata['data_quality_score']}/100")
        elif stage == 'preprocessing' and 'memory_reduction_mb' in result.metadata:
            print(f"     Memory Saved: {result.metadata['memory_reduction_mb']:.2f} MB")


def show_feature_engineering_results(feature_engineering):
    """Display feature engineering results"""
    
    print(f"   Features Created: {len(feature_engineering.features_created)}")
    if feature_engineering.features_created:
        for feature in feature_engineering.features_created[:5]:  # Show first 5
            print(f"     - {feature}")
        if len(feature_engineering.features_created) > 5:
            print(f"     ... and {len(feature_engineering.features_created) - 5} more")
    
    print(f"   Features Dropped: {len(feature_engineering.features_dropped)}")
    if feature_engineering.features_dropped:
        for feature in feature_engineering.features_dropped:
            print(f"     - {feature}")


def show_data_quality_report(validation_report):
    """Display data quality report"""
    
    print(f"   Validation Passed: {validation_report.passed}")
    print(f"   Total Issues: {len(validation_report.issues)}")
    print(f"   Quality Score: {validation_report.summary.get('data_quality_score', 'N/A')}/100")
    
    if validation_report.issues:
        print("   Issues Found:")
        for issue in validation_report.issues[:3]:  # Show first 3 issues
            print(f"     - {issue.severity.value.upper()}: {issue.message}")
        if len(validation_report.issues) > 3:
            print(f"     ... and {len(validation_report.issues) - 3} more issues")


async def batch_processing_example():
    """Example of batch processing multiple files"""
    
    print("\n" + "=" * 60)
    print("📦 Batch Processing Example")
    print("=" * 60)
    
    pipeline = DataPipeline()
    
    # Create multiple sample datasets
    datasets = [
        ("customers_q1.csv", create_sample_churn_data(500)),
        ("customers_q2.csv", create_sample_churn_data(750)),
        ("customers_q3.csv", create_sample_churn_data(600))
    ]
    
    # Save sample files
    file_configs = []
    for filename, data in datasets:
        data.to_csv(filename, index=False)
        file_configs.append((filename, 'customer_churn'))
        print(f"Created: {filename} with {len(data)} records")
    
    # Process batch
    print(f"\nProcessing {len(file_configs)} files in batch...")
    results = await pipeline.process_batch(file_configs, max_concurrent=2)
    
    print(f"✅ Batch processing completed!")
    print(f"   Processed: {len(results)} files successfully")
    
    # Show results summary
    total_input_rows = sum(len(pd.read_csv(filename)) for filename, _ in file_configs)
    total_output_rows = sum(len(df) for df, _ in results)
    
    print(f"   Total input rows: {total_input_rows:,}")
    print(f"   Total output rows: {total_output_rows:,}")
    print(f"   Average processing rate: {total_output_rows / len(results):,.0f} rows per file")


if __name__ == "__main__":
    # Run main example
    asyncio.run(main())
    
    # Run batch processing example
    asyncio.run(batch_processing_example())