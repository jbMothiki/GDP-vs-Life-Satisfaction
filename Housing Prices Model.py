import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import xlrd


# Load the data
oecd_bli = pd.read_csv("oecd_bli.csv")
gdp = pd.read_excel("WEO_data.xlsX")

# Prepare the data
gdp = pd.DataFrame(gdp[["Country", 2015]])
# Drawing data of "Life Satisfaction" and "Total"
oecd_bli = oecd_bli.loc[(oecd_bli["Indicator"] == "Life satisfaction") & (oecd_bli["Inequality"] == "Total")]

oecd_bli = pd.DataFrame(oecd_bli[["Country", "Value"]])

# Prepare Country Stats
country_stats = pd.merge(gdp, oecd_bli, on="Country")
country_stats.columns = ["Country", "GDP", "BLI"]
country_stats = country_stats.round(2)

# Plot
plt.scatter(country_stats["GDP"], country_stats["BLI"])
plt.show()

# Prepare the data
X = np.c_[country_stats["GDP"]]
y = np.c_[country_stats["BLI"]]

# Select a linear model
model = sklearn.linear_model.LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus' GDP per capita
print(model.predict(X_new))  # output

