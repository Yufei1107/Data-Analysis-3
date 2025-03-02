
# Analysis of Hourly Earnings Data
### Yufei Shao
### 02.03.2025


**Abstract:**  
This study investigates hourly earnings using demographic, work-related, and socioeconomic predictors. We systematically select predictors based on theory and data availability, then develop a sequence of linear regression models with increasing complexity. The analysis focuses on how adding predictors influences both training fit and generalization performance, as assessed by RMSE and BIC.

**Introduction:**  
Hourly earnings are determined by a range of factors. We begin with basic demographic predictors (age and sex) and progressively incorporate work hours, education, marital status, and industry classification. The rationale behind predictor selection is twofold: (1) demographic and work hour variables capture fundamental individual characteristics and productivity; (2) education, marital status, and industry provide insights into human capital and sector-specific effects.

**Methods and Predictor Selection:**  
- **Basic Demographics:**  
  Age and sex are primary indicators of experience and inherent earning potential.  
- **Work Hours:**  
  Weekly work hours serve as a proxy for labor input and productivity, directly influencing earnings.
- **Human Capital Indicators:**  
  Education level (categorized from grade92) and marital status are included to represent accumulated skills and social stability.
- **Industry Classification:**  
  The industry variable groups firms into high-frequency sectors versus others, capturing contextual earning differences.  

Each step in the model sequence reflects a theoretical justification for including additional variables, aiming to balance simplicity with explanatory power.

**Results and Model Performance Comparison:**  
Four models were developed:
- **Model 1 (Demographics):** Baseline predictors (age and sex).  
- **Model 2 (+Work Hours):** Inclusion of weekly working hours improved the training fit by reflecting productivity.
- **Model 3 (+Education/Marital):** Adding education and marital status reduced cross-validated RMSE and achieved the lowest BIC, indicating optimal complexity.
- **Model 4 (+Industry):** Although further industry variables reduced the full-sample RMSE, they resulted in a slight increase in cross-validation error and BIC, suggesting mild overfitting.

**Conclusion:**  
The systematic selection of predictors—from core demographics to work input and human capital—reveals that while more variables may improve the in-sample fit, optimal generalization is achieved with a balanced model (Model 3). For predictive applications, Model 3 is recommended; however, Model 4 offers additional insights useful for industry-specific analyses. Future work may extend this approach to non-linear frameworks to further enhance predictive accuracy.

*Keywords: hourly earnings, predictor selection, linear regression, RMSE, BIC, model complexity*
