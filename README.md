# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required Python libraries such as Pandas and Scikit-learn.
2.Create a dataset containing multiple input features such as area, number of bedrooms, and age of the house.
3.Separate the input features from the target variables, namely house price and number of occupants.
4.Split the dataset into training and testing data.
5.Standardize the input features using StandardScaler.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Sathish v
RegisterNumber:  212225040391
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create the dataset
data = {
    "Area": [1000, 1200, 1500, 1800, 2000,
             2200, 2500, 1300, 1600, 1900,
             2100, 2400, 2800, 1100, 1700],

    "Bedrooms": [2, 2, 3, 3, 3,
                 4, 4, 2, 3, 3,
                 4, 4, 5, 2, 3],

    "Age": [20, 15, 10, 8, 5,
            4, 3, 18, 12, 7,
            6, 5, 2, 25, 9],

    "Price": [180000, 210000, 260000, 310000, 350000,
              390000, 450000, 220000, 280000, 330000,
              370000, 420000, 500000, 170000, 295000],

    "Occupants": [3, 3, 4, 5, 5,
                  6, 7, 3, 4, 5,
                  6, 6, 8, 2, 5]
}

df = pd.DataFrame(data)

# Display the dataset
print("Dataset:")
print(df)

# Input features
X = df[["Area", "Bedrooms", "Age"]]

# Target variables
y_price = df["Price"]
y_occupants = df["Occupants"]

# Split the dataset
X_train, X_test, y_price_train, y_price_test, \
y_occupants_train, y_occupants_test = train_test_split(
    X,
    y_price,
    y_occupants,
    test_size=0.2,
    random_state=42
)

# Standardize the input features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SGD Regressor for house price
price_model = SGDRegressor(
    max_iter=1000,
    learning_rate="constant",
    eta0=0.01,
    random_state=42
)

# Train the price model
price_model.fit(X_train_scaled, y_price_train)

# Predict house price
price_pred = price_model.predict(X_test_scaled)

# SGD Regressor for number of occupants
occupant_model = SGDRegressor(
    max_iter=1000,
    learning_rate="constant",
    eta0=0.01,
    random_state=42
)

# Train the occupant model
occupant_model.fit(X_train_scaled, y_occupants_train)

# Predict number of occupants
occupant_pred = occupant_model.predict(X_test_scaled)

# Calculate Mean Squared Error
price_mse = mean_squared_error(y_price_test, price_pred)
occupant_mse = mean_squared_error(
    y_occupants_test,
    occupant_pred
)

# Calculate R2 Score
price_r2 = r2_score(y_price_test, price_pred)
occupant_r2 = r2_score(
    y_occupants_test,
    occupant_pred
)

# Display results
print("\nActual House Prices:")
print(y_price_test.values)

print("\nPredicted House Prices:")
print(price_pred)

print("\nActual Number of Occupants:")
print(y_occupants_test.values)

print("\nPredicted Number of Occupants:")
print(occupant_pred)

print("\nHouse Price Mean Squared Error:")
print(price_mse)

print("\nHouse Price R2 Score:")
print(price_r2)

print("\nOccupants Mean Squared Error:")
print(occupant_mse)

print("\nOccupants R2 Score:")
print(occupant_r2)

# Predict for a new house
new_house = pd.DataFrame({
    "Area": [2000],
    "Bedrooms": [3],
    "Age": [10]
})

# Standardize new house data
new_house_scaled = scaler.transform(new_house)

# Make predictions
new_price = price_model.predict(new_house_scaled)
new_occupants = occupant_model.predict(new_house_scaled)

print("\nPrediction for New House:")
print("Area: 2000 sq.ft")
print("Bedrooms: 3")
print("Age: 10 years")

print("\nPredicted House Price:", new_price[0])
print("Predicted Number of Occupants:", new_occupants[0])
*/
```

## Output:
<img width="488" height="797" alt="640020411-ffa073b0-8b34-4faa-b506-572d9c0bcdfc" src="https://github.com/user-attachments/assets/23f129f8-bb85-478e-99ea-6b4e6bc4b6eb" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
