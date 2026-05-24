import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\ADMIN\Documents\ĐẠI HỌC\Nhập môn trí tuệ nhân tạo\dataset\diabetes.csv')
invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

df[invalid_cols] = df[invalid_cols].replace(0, np.nan)

for col in invalid_cols:
    df[col].fillna(df[col].median(), inplace=True)

print(df)
# df.to_csv(r"C:\Users\ADMIN\Documents\ĐẠI HỌC\Nhập môn trí tuệ nhân tạo\dataset\clean_diabetes.csv", index=False)
