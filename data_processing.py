"""
data_processing.py

Contains functions for loading, cleaning, and transforming the raw dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Load dataset from a CSV file.
    
    Parameters:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    dtype_dict = {'ind02': str}
    df = pd.read_csv(filepath, dtype=dtype_dict, low_memory=False)
    return df

def clean_data(df, target_occupations=['0820', '0830', '0840']):
    """
    Clean and filter the data.
    
    Steps:
      1. Merge occupational codes by extracting the first 4 digits.
      2. Filter for target occupations.
      3. Handle missing values in key columns.
      4. Remove outliers based on weekly earnings, age, and working hours.
    
    Parameters:
        df (pd.DataFrame): Original dataframe.
        target_occupations (list): List of target occupational codes.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # Merge occupational codes (assuming occ2012 is a string)
    df['occ2012'] = df['occ2012'].astype(str).str.zfill(4)
    df['occ_code'] = df['occ2012'].str[:4]
    filtered_df = df[df['occ_code'].isin(target_occupations)].copy()

    # Replace missing 'ethnic' values with 0
    filtered_df['ethnic'] = filtered_df['ethnic'].fillna(0)

    # Drop rows with missing key columns
    required_cols = ['earnwke', 'uhours', 'age']
    filtered_df = filtered_df.dropna(subset=required_cols)

    # Filter out outliers:
    filtered_df = filtered_df[filtered_df['earnwke'] >= 290]  # Minimum weekly earnings
    filtered_df = filtered_df[(filtered_df['age'] >= 18) & (filtered_df['age'] <= 80)]  # Valid age range
    filtered_df = filtered_df[(filtered_df['uhours'] >= 0) & (filtered_df['uhours'] <= 168)]  # Valid work hours

    return filtered_df

def verify_clean_data(df):
    """
    Print verification details of the cleaned data.
    
    Parameters:
        df (pd.DataFrame): Cleaned dataframe.
    """
    print("Verification of cleaned data:")
    print(f"Total number of records: {len(df)}\n")
    print("Records per occupational code:")
    print(df['occ_code'].value_counts())
    print("\nMissing values count:")
    print(df.isnull().sum())
    print("\nFirst 5 rows sample:")
    print(df.head())

def split_and_transform(df):
    """
    Split the data into training and holdout sets and perform necessary transformations.
    
    Steps:
      1. Split data (80% training, 20% holdout).
      2. Create a log-transformed target variable.
      3. Group education and marital status into categories.
      4. Process industry information.
      5. One-hot encode categorical variables.
    
    Parameters:
        df (pd.DataFrame): Dataframe with earnings data.
        
    Returns:
        tuple: (train_df, holdout_df) after transformation.
    """
    # Split the dataset
    train_df, holdout_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training set size: {len(train_df)}, Holdout set size: {len(holdout_df)}")
    
    # Create log-transformed target variable
    train_df['earnings_per_hour'] = train_df['earnwke'] / train_df['uhours'].replace(0, np.nan)
    train_df = train_df.dropna(subset=['earnings_per_hour'])
    holdout_df['earnings_per_hour'] = holdout_df['earnwke'] / holdout_df['uhours'].replace(0, np.nan)
    holdout_df = holdout_df.dropna(subset=['earnings_per_hour'])
    
    train_df['log_earnings'] = np.log(train_df['earnings_per_hour'])
    holdout_df['log_earnings'] = np.log(holdout_df['earnings_per_hour'])
    
    # Group education levels based on grade92
    def map_education(grade):
        if grade <= 39:
            return "HighSchool_or_below"
        elif 40 <= grade <= 43:
            return "Bachelor_or_Associate"
        else:
            return "Master_or_above"
    train_df['edu_group'] = train_df['grade92'].apply(map_education)
    holdout_df['edu_group'] = holdout_df['grade92'].apply(map_education)
    
    # Group marital status
    marital_map = {
        1: 'Married', 2: 'Married', 3: 'Married',
        4: 'Widowed_Divorced', 5: 'Widowed_Divorced', 6: 'Widowed_Divorced',
        7: 'NeverMarried'
    }
    train_df['marital_group'] = train_df['marital'].map(marital_map)
    holdout_df['marital_group'] = holdout_df['marital'].map(marital_map)
    
    # Process industry information
    train_df['industry_main'] = train_df['ind02'].str.split('(').str[0].str.strip()
    holdout_df['industry_main'] = holdout_df['ind02'].str.split('(').str[0].str.strip()
    top_industries = train_df['industry_main'].value_counts().nlargest(5).index
    train_df['industry_group'] = np.where(train_df['industry_main'].isin(top_industries),
                                          train_df['industry_main'], 'Other')
    holdout_df['industry_group'] = np.where(holdout_df['industry_main'].isin(top_industries),
                                            holdout_df['industry_main'], 'Other')
    
    # One-hot encode categorical variables
    from sklearn.preprocessing import OneHotEncoder
    cat_cols = ['edu_group', 'marital_group', 'industry_group']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_train = encoder.fit_transform(train_df[cat_cols])
    encoded_cols = encoder.get_feature_names_out(cat_cols)
    train_df[encoded_cols] = encoded_train
    encoded_holdout = encoder.transform(holdout_df[cat_cols])
    holdout_df[encoded_cols] = encoded_holdout
    
    # Drop intermediate columns
    drop_cols = ['grade92', 'marital', 'ind02', 'industry_main']
    train_df = train_df.drop(columns=drop_cols)
    holdout_df = holdout_df.drop(columns=drop_cols)
    
    # Print summary of encoded features
    print("Encoded features (training set sample):")
    print(train_df[encoded_cols].head())
    print("\nEncoded feature dimensions:", train_df[encoded_cols].shape[1])
    
    return train_df, holdout_df
