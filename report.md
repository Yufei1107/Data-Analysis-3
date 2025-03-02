
# Analysis of Hourly Earnings Data
### Yufei Shao
### 02.03.2025


**Abstract:**  
This study examines the determinants of hourly earnings using demographic, work-related, and socioeconomic predictors. We develop a series of linear regression models of increasing complexity to assess predictive performance and the trade-off between model fit and complexity.

**Introduction:**  
Hourly earnings are influenced by multiple factors including age, gender, work hours, education, marital status, and industry. This study investigates these relationships and identifies the optimal model specification based on cross-validated error and information criteria.

**Methods:**  
Data were cleaned by filtering outliers (e.g., earnings < $290, ages outside 18–80, and implausible work hours) and missing values. Key predictors included:
- *Demographics:* Age, sex  
- *Productivity:* Weekly work hours  
- *Human Capital:* Education level (categorized from grade92) and marital status  
- *Industry:* Grouping of primary industries  
Models were evaluated using RMSE and BIC.

**Results:**  
The analysis shows that while adding predictors continuously reduces training RMSE, cross-validation identifies Model 3 (demographics, work hours, education, and marital status) as optimal. Model 4, including industry effects, exhibits slight overfitting as indicated by higher BIC.  

**Conclusion:**  
For predictive accuracy, Model 3 is recommended due to its balanced performance. However, Model 4 provides additional business insights through industry segmentation. Future work could explore nonlinear models to capture more complex relationships.

*Keywords: hourly earnings, linear regression, model complexity, cross-validation, BIC*
