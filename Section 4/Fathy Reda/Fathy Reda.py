import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("calories.csv")

df = df.dropna()
df = df[df['Calories'] > 0]

encoder = OneHotEncoder(drop="first", sparse_output=False)
activity_encoded = encoder.fit_transform(df[['Activity']])
df_encoded = pd.concat(
    [df.drop('Activity', axis=1), pd.DataFrame(activity_encoded, columns=encoder.get_feature_names_out())],
    axis=1
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded.drop('Calories', axis=1))
y = df_encoded['Calories']

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(mean_absolute_error(y_test, preds))
print(r2_score(y_test, preds))

sample = pd.DataFrame({
    "Age": [25],
    "Weight": [70],
    "Height": [175],
    "Duration": [60],
    "Heart_Rate": [130],
    "Activity": ["Brisk Walking"]
})

sample_encoded = encoder.transform(sample[['Activity']])
sample_final = pd.concat(
    [sample.drop('Activity', axis=1), pd.DataFrame(sample_encoded, columns=encoder.get_feature_names_out())],
    axis=1
)

sample_scaled = scaler.transform(sample_final)
prediction = model.predict(sample_scaled)
print("Predicted Calories:", prediction[0])