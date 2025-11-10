import pytest
import pandas as pd
import numpy as np
from scripts.inference_transformer import FeatureEnrichmentTransformer
from datetime import datetime

# Configure logger for tests (optional but good practice)
import logging
logging.basicConfig(level=logging.WARNING)

@pytest.fixture
def dummy_data() -> pd.DataFrame:
    """Fixture to create a minimal raw DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'date': ['2015-01-01', '2015-01-02', '2016-01-01', '2016-01-02', '2016-01-03'],
        'country': ['Australia', 'Australia', 'Australia', 'Australia', 'Canada'],
        'store': ['A', 'A', 'B', 'B', 'C'],
        'product': ['Sticker1', 'Sticker1', 'Sticker2', 'Sticker2', 'Sticker3'],
        'num_sold': [100, 110, 200, 210, 50], # Target column used for lag/rolling calculation
    })

def test_transformer_adds_time_features(dummy_data):
    """Verify time-based features (year, month, dayofweek) are added."""
    df_raw = dummy_data.copy()
    
    # Instantiate the transformer
    # Pass 'num_sold' as target_column so it knows which column to use for lag/rolling in fit
    transformer = FeatureEnrichmentTransformer(target_column='num_sold') 
    
    # Fit and transform
    df_transformed = transformer.fit_transform(df_raw)
    
    # Assert new columns exist and have correct dtypes
    assert 'year' in df_transformed.columns
    assert 'month' in df_transformed.columns
    assert 'dayofweek' in df_transformed.columns
    assert df_transformed['year'].dtype == np.int64
    
    # Assert the values are correct
    assert df_transformed.loc[df_transformed['id'] == 1, 'year'].iloc[0] == 2015
    assert df_transformed.loc[df_transformed['id'] == 3, 'month'].iloc[0] == 1
    
def test_transformer_calculates_lag_rolling_features(dummy_data):
    """Verify lag_1 and rolling_7 features are calculated correctly."""
    df_raw = dummy_data.copy()
    
    transformer = FeatureEnrichmentTransformer(target_column='num_sold')
    df_transformed = transformer.fit_transform(df_raw)
    
    # Assert new columns exist
    assert 'lag_1' in df_transformed.columns
    assert 'rolling_7' in df_transformed.columns
    
    # Test lag_1 (grouped by country, store, product)
    # The first day for a group should have a missing lag (filled by median)
    # For id=2 (date 2015-01-02), lag should be num_sold of id=1 (100)
    assert df_transformed.loc[df_transformed['id'] == 2, 'lag_1'].iloc[0] == 100.0
    
    # For id=4 (date 2016-01-02), lag should be num_sold of id=3 (200)
    assert df_transformed.loc[df_transformed['id'] == 4, 'lag_1'].iloc[0] == 200.0

def test_transformer_handles_inference_data(dummy_data):
    """Verify that when 'num_sold' is missing (inference time), the transformer uses fallback values."""
    # Simulate inference data by dropping 'num_sold'
    df_inference = dummy_data.copy().drop(columns=['num_sold'])
    
    # Fit the transformer on training data first to capture the medians
    transformer = FeatureEnrichmentTransformer(target_column='num_sold')
    transformer.fit(dummy_data) 
    
    # Transform the inference data
    df_transformed = transformer.transform(df_inference)
    
    # Check that the necessary columns for the model are still created
    assert 'year' in df_transformed.columns
    assert 'lag_1' in df_transformed.columns
    
    # Check that lag_1 is filled (it should be the median calculated during fit)
    # Since the median is calculated on the training data, we just assert it's not null.
    assert not df_transformed['lag_1'].isnull().any()
    assert df_transformed['lag_1'].iloc[0] == transformer.median_lag_1