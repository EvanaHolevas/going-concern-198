import pandas as pd
import statsmodels.api as sm

data = pd.read_csv("data/1_american_dataset.csv")

grouped_data = data.groupby('company_name')
print(grouped_data.columns)

# Loop through each company and fit an autoregressive model
# for company, data in grouped_data:
#     # Set date column as index
#     data.set_index('fdate', inplace=True)
    
#     # Fit autoregressive model with lag 2
#     model = sm.tsa.AR(data['financial_feature']).fit(maxlag=2)
    
#     # Print summary of the model
#     print(f"Company: {company}")
#     print(model.summary())