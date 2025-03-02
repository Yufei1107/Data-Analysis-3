"""
visualization.py

Contains functions for generating descriptive statistics and visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def descriptive_stats_and_visualization(df):
    """
    Create an earnings per hour variable, generate descriptive statistics,
    and produce several visualizations.
    
    Parameters:
        df (pd.DataFrame): Cleaned dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with the 'earnings_per_hour' column added.
    """
    # Create earnings per hour variable (avoiding division by zero)
    df['earnings_per_hour'] = df['earnwke'] / df['uhours'].replace(0, float('nan'))
    df = df.dropna(subset=['earnings_per_hour'])

    # Print descriptive statistics
    desc_stats = df[['earnings_per_hour', 'age', 'uhours']].describe().T
    desc_stats['skewness'] = [
        df['earnings_per_hour'].skew(),
        df['age'].skew(),
        df['uhours'].skew()
    ]
    print("Descriptive Statistics:")
    print(desc_stats[['mean', 'std', 'min', '50%', 'max', 'skewness']])
    
    # Set visualization style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots(3, 2, figsize=(20, 20))
    
    # Distribution of hourly earnings
    sns.histplot(df['earnings_per_hour'], bins=50, kde=True, ax=ax[0,0])
    ax[0,0].set_title('Distribution of Hourly Earnings', fontweight='bold')
    ax[0,0].set_xlabel('USD per hour')
    ax[0,0].set_ylabel('Count')
    
    # Log-transformed hourly earnings
    sns.histplot(df['earnings_per_hour'].apply(lambda x: np.log(x+1)), bins=50, kde=True, ax=ax[0,1])
    ax[0,1].set_title('Log-Transformed Hourly Earnings', fontweight='bold')
    ax[0,1].set_xlabel('log(USD per hour + 1)')
    
    # Boxplot: Hourly earnings by gender
    sns.boxplot(x='sex', y='earnings_per_hour', data=df, ax=ax[1,0])
    ax[1,0].set_title('Hourly Earnings by Gender', fontweight='bold')
    ax[1,0].set_xticks([0, 1])
    ax[1,0].set_xticklabels(['Male', 'Female'])
    ax[1,0].set_ylabel('USD per hour')
    
    # Barplot: Median hourly earnings by education level (using grade92)
    sns.barplot(x='grade92', y='earnings_per_hour', data=df, estimator='median', ax=ax[1,1])
    ax[1,1].set_title('Hourly Earnings by Education Level', fontweight='bold')
    ax[1,1].set_xlabel('Education Code (grade92)')
    ax[1,1].set_ylabel('Median USD per hour')
    
    # Scatterplot: Age vs. hourly earnings
    sns.scatterplot(x='age', y='earnings_per_hour', data=df, alpha=0.4, ax=ax[2,0])
    ax[2,0].set_title('Age vs. Hourly Earnings', fontweight='bold')
    ax[2,0].set_ylabel('USD per hour')
    
    # Line plot: Age-Earnings profile (median)
    sns.lineplot(x='age', y='earnings_per_hour', data=df, 
                 estimator='median', errorbar=None, ax=ax[2,1])
    ax[2,1].set_title('Age-Earnings Profile (Median)', fontweight='bold')
    ax[2,1].set_ylabel('Median USD per hour')
    
    plt.tight_layout()
    plt.show()
    
    # Pie chart: Occupational distribution
    plt.figure(figsize=(8,8))
    df['occ_code'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Occupation Distribution', fontweight='bold')
    plt.ylabel('')
    plt.show()
    
    return df
