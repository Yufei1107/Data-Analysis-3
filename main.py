"""
main.py

Entry point of the project. Executes data processing, visualization, modeling, and evaluation.
"""

from data_processing import load_data, clean_data, verify_clean_data, split_and_transform
from visualization import descriptive_stats_and_visualization
from modeling import train_linear_models, evaluate_models

def main():
    # Load data
    df = load_data('morg-2014-emp.csv')
    
    # Clean and verify data
    df_clean = clean_data(df)
    verify_clean_data(df_clean)
    
    # Generate descriptive statistics and visualizations
    df_clean = descriptive_stats_and_visualization(df_clean)
    
    # Split and transform data into training and holdout sets
    train_df, holdout_df = split_and_transform(df_clean)
    
    # Train regression models with increasing complexity
    models = train_linear_models(train_df)
    
    # Evaluate models and visualize performance metrics
    evaluate_models(train_df, models)

if __name__ == '__main__':
    main()
