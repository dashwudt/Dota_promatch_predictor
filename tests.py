import unittest
import pandas as pd
from data import remove_rows_with_least_frequent_values

def test_remove_least_frequent():
    # Create a sample DataFrame
    df = pd.DataFrame({
        'A': [1, 2, 2, 3, 3],
        'B': [3, 3, 4, 4, 5],
        'C': [5, 5, 6, 6, 6]
    })

    # Call the function with the sample DataFrame and number of least frequent values to remove
    result_df = remove_rows_with_least_frequent_values(df, 3)

    # Expected DataFrame after removing rows with the three least frequent values
    expected_df = pd.DataFrame({
        'A': [3],
        'B': [5],
        'C': [6]
    })

    # Check if the resulting DataFrame is equal to the expected DataFrame
    assert result_df.equals(expected_df), "The result does not match the expected DataFrame."