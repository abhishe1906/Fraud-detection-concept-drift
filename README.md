# Fraud Detection Under Concept Drift

This project studies **credit card fraud detection in a non-stationary environment**.
Instead of focusing on single-point accuracy, it evaluates **operational stability over time**
using temporal splits, rolling retraining, and threshold-based decision policies.

## Key Ideas
- Fraud patterns evolve over time (concept drift)
- Static models degrade after deployment
- Rolling retraining improves performance stability
- Threshold selection is a business decision

## Methodology
- Strict temporal evaluation (no random splits)
- Monthly rolling retraining with a 3-month window
- Logistic Regression with class balancing
- Evaluation using recall, precision, and metric variance

## Results Summary

| Model | Mean Recall | Std Recall |
|------|------------|------------|
| Static | ~0.42 | High |
| Rolling (3-month) | ~0.47 | Low |
| Rolling + Threshold | ~0.76 | Higher |

Rolling retraining significantly reduces recall volatility under concept drift.

## Kaggle Notebook
The full analysis is available on Kaggle:  
👉 **https://www.kaggle.com/code/abhishekagrawal1654/fraud-detection-under-concept-drift**

## Tools
- Python
- pandas, NumPy
- scikit-learn
- matplotlib

## Notes
This project emphasizes **production-aware evaluation** rather than leaderboard optimization.


<img width="989" height="490" alt="recall_over_time" src="https://github.com/user-attachments/assets/ac5df442-b177-487f-b20b-293ea64ce4a9" />
<img width="989" height="490" alt="precision_over_time" src="https://github.com/user-attachments/assets/1abffc49-eb5b-48e8-b050-23e4591cf165" />

