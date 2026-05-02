import pandas as pd
from wddinference_test import ModelAPI

# 1) Initialize API (automatically loads model and scaler)
api = ModelAPI()

# 2) Read your inference data
#   ⚠️ This DataFrame must contain column names from INPUT_FEATURES and have at least SEQ_LEN rows
df = pd.read_excel("D:\code\学习\wdd\\25wdd.xlsx")

# 3) Single-step prediction (next step)
next_val = api.predict_next(df)
print("Next step prediction:", next_val)

