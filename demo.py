# Import python packages
import streamlit as st
#from snowflake.snowpark.context import get_active_session
from streamlit.hello.utils import show_code
import time
import joblib

# data science libs
import numpy as np
import pandas as pd
from datetime import date
#import holidays
import pandas as pd
from PIL import Image


# viz libs
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

# Streamlit config
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Plotting Demo", page_icon="📈")

st.title("Algorithim Avengers")

image1 = Image.open(r'purchased.png')

st.image(image1)






st.title("Distributed Random Forest")


image5 = Image.open(r'final_ye.png')
st.image(image5)

image7 = Image.open(r'yeeeet.png')
st.image(image7)

#image6 = Image.open(r'big_boi.png')
#st.image(image6)

#---------------------------------------------------------------------------------------------------------------
st.title("Data Discrepencies")
#---------------------------------------------------------------------------------------------------------------

image3 = Image.open(r'output.png')

st.image(image3)


image4 = Image.open(r'output2.png')

st.image(image4)


image2 = Image.open(r'grapht1.png')

st.image(image2)
 


st.write("Calculates the number of rows that had a delivery date prior to the create date")
st.code("""
        
original_count = len(koch)
weird_dates_df = koch[koch['DELIVERY_DATE'] > koch['CREATE_DATE']]
filtered_count = len(weird_dates_df)
# Calculate the number of dropped rows
dropped_count = original_count - filtered_count
print(f"Number of rows dropped: {dropped_count}")  
             
""")

st.write("Calculate the number of rows that had a delivery date that was early.")

st.code("""
original_count = len(koch)
# Get tomorrow's date for comparison
tomorrow = datetime.today().date() + timedelta(days=1)
# Filter rows where 'DELIVERY_DATE' is tomorrow or in the future
more_weird = koch[koch['DELIVERY_DATE'].dt.date <= tomorrow]
# Get the count of the filtered rows
more_weird_count = len(more_weird)
# Calculate the number of rows dropped
dropped_count_2 = original_count - more_weird_count
print(f"Number of rows dropped: {dropped_count_2}")
""")




st.code("""

# Snowpark for Python
from snowflake.snowpark import Session
from snowflake.snowpark.version import VERSION
from snowflake.snowpark.types import StructType, StructField, DoubleType, StringType
import snowflake.snowpark.functions as F

# data science libs
import numpy as np
import pandas as pd
import polars as pl
# misc
import json


""")



st.code("""

connection_parameters = json.load(open('connection.json'))
session = Session.builder.configs(connection_parameters).create()
session.sql_simplifier_enabled = True

snowflake_environment = session.sql('SELECT current_user(), current_version()').collect()
snowpark_version = VERSION

# Current Environment Details
print('\nConnection Established with the following parameters:')
print('User                        : {}'.format(snowflake_environment[0][0]))
print('Role                        : {}'.format(session.get_current_role()))
print('Database                    : {}'.format(session.get_current_database()))
print('Schema                      : {}'.format(session.get_current_schema()))
print('Warehouse                   : {}'.format(session.get_current_warehouse()))
print('Snowflake version           : {}'.format(snowflake_environment[0][1]))
print('Snowpark for Python version : {}.{}.{}'.format(snowpark_version[0],snowpark_version[1],snowpark_vers


""")





st.code("""
session.sql("DESCRIBE TABLE PROCUREMENT_ON_TIME_DELIVERY.PURCHASE_ORDER_HISTORY;").show()

df = session.read.options({"field_delimiter": ",",
                                    "field_optionally_enclosed_by": '"',
                                    "infer_schema": True,
                                    "parse_header": True}).table("PROCUREMENT_ON_TIME_DELIVERY.PURCHASE_ORDER_HISTORY")



# Select only three columns from the DataFrame
#selected_coke = coke.select("PURCHASE_DOCUMENT_ID", "CREATE_DATE", "COMPANY_CODE_ID")

# Show the selected columns

""")



st.code("""

date_columns = ['POR_DELIVERY_DATE', 'DELIVERY_DATE', 'REQUESTED_DELIVERY_DATE', 'FIRST_GR_POSTING_DATE']
for column in date_columns:
    temp_series = koch[column].fillna(0).astype(int).astype(str)
    # Replace '0' string back to NaN to avoid incorrect date conversion
    temp_series = temp_series.replace('0', np.nan)
    # Convert to datetime
    koch[column] = pd.to_datetime(temp_series, format='%Y%m%d', errors='coerce')
#Creating target column
koch['Time_Difference'] = koch['FIRST_GR_POSTING_DATE'] - koch['DELIVERY_DATE']
koch['Time_Difference'] = koch['Time_Difference'].dt.days
""")



st.code("""
coke = coke.with_columns(
    pl.col("CREATE_DATE").str.strptime(pl.Date, "%Y%m%d").alias("CREATE_DATE")
)

""")



st.code("""
coke = coke.with_columns([
    pl.col('CREATE_DATE').dt.weekday().alias('day_of_week'),  # Monday=0, Sunday=6
    pl.col('CREATE_DATE').dt.month().alias('month'),
    pl.col('CREATE_DATE').dt.quarter().alias('quarter')
])

""")



st.code("""
value_counts = coke.group_by('day_of_week').agg(pl.count('day_of_week').alias('frequency'))

value_counts_sorted = value_counts.sort('frequency')

""")



st.code("""

import plotly.express as px
df = value_counts_sorted.to_pandas()
# Map the numeric day of the week to actual day names for better readability
day_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 
             5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
df['day_name'] = df['day_of_week'].map(day_names)

# Sort the DataFrame by 'day_of_week' to ensure the days are in the correct order
df = df.sort_values('day_of_week')

# Create the bar chart using Plotly Express
fig = px.bar(df, x='day_name', y='frequency', 
             title='Frequency of Days of the Week When Purchased',
             labels={'day_name': 'Day of the Week', 'frequency': 'Frequency'},
             color='day_name')

# Show the plot
fig.show()
""")



st.code("""
coke = coke.with_columns(pl.col('DELIVERY_DATE').cast(pl.Utf8))
coke = coke.with_columns(
    pl.col("DELIVERY_DATE").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S.%f")
)

coke = coke.with_columns([
    pl.col('DELIVERY_DATE').dt.weekday().alias('D_day_of_week'),  # Monday=0, Sunday=6
])
value_counts = coke.group_by('D_day_of_week').agg(pl.count('D_day_of_week').alias('D_frequency'))

value_counts_sorted_D = value_counts.sort('D_frequency')
""")



st.code("""

ddf = value_counts_sorted_D 
ddf = ddf.to_pandas()
# Map the numeric day of the week to actual day names for better readability
day_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 
             5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
ddf['day_name'] = ddf['D_day_of_week'].map(day_names)

# Sort the DataFrame by 'D_day_of_week' to ensure the days are in the correct order
ddf = ddf.sort_values('D_day_of_week')

# Create the bar chart using Plotly Express
fig = px.bar(ddf, x='day_name', y='D_frequency', 
             title='Frequency of Days of the Week When Delivered',
             labels={'day_name': 'Day of the Week', 'D_frequency': 'Frequency'},
             color='day_name')

# Show the plot
fig.show()
""")



st.code("""
def determine_status(days):
    if days < 0:
        return 'Early'
    elif days == 0:
        return 'On Time'
    else:
        return 'Late'
# Give a status of it
koch['Arrival_Status'] = koch['Time_Difference'].apply(determine_status)

""")



st.code("""

# Count occurrences of each category in 'SUB_COMMODITY_DESC'
category_counts = koch['SUB_COMMODITY_DESC'].value_counts()

# Create a new column 'Category_total_counts' by mapping counts based on 'SUB_COMMODITY_DESC'
koch['Category_total_counts'] = koch['SUB_COMMODITY_DESC'].map(category_counts)

# Create a DataFrame with unique 'SUB_COMMODITY_DESC' and corresponding counts
selected_columns = koch[['SUB_COMMODITY_DESC', 'Category_total_counts']].drop_duplicates()

# Create a new column 'Ranking' based on the order of 'Category_total_counts'
selected_columns['Ranking'] = selected_columns['Category_total_counts'].rank(ascending=False, method='min')

# Sort the DataFrame by 'Category_total_counts' in descending order and then by 'Ranking' in ascending order
selected_columns = selected_columns.sort_values(by=['Ranking','Category_total_counts'], ascending=[False, True])

# Drop rows with missing values
selected_columns = selected_columns.dropna()

selected_columns = selected_columns[['Ranking', 'SUB_COMMODITY_DESC']]
""")



st.code("""
koch = koch.merge(selected_columns, on='SUB_COMMODITY_DESC', how='left')

""")



st.code("""

# Count occurrences of each category in 'MATERIAL_ID'
material_counts = koch['MATERIAL_ID'].value_counts()

# Create a new column 'Material_total_counts' by mapping counts based on 'MATERIAL_ID'
koch['Material_total_counts'] = koch['MATERIAL_ID'].map(material_counts)

# Create a DataFrame with unique 'MATERIAL_ID' and corresponding counts
selected_columns1 = koch[['MATERIAL_ID', 'Material_total_counts']].drop_duplicates()

# Create a new column 'Ranking' based on the order of 'Material_total_counts'
selected_columns1['Ranking_M'] = selected_columns1['Material_total_counts'].rank(ascending=False, method='min')

# Sort the DataFrame by 'Material_total_counts' in descending order and then by 'Ranking' in ascending order
selected_columns1 = selected_columns1.sort_values(by=['Material_total_counts', 'Ranking_M'], ascending=[False, True])
selected_columns1 = selected_columns1[['Ranking_M', 'Material_total_counts', 'MATERIAL_ID']].drop_duplicates()

# Drop rows with missing values
selected_columns1 = selected_columns1.dropna()
selected_columns1 = selected_columns1[['Ranking_M', 'MATERIAL_ID']]
""")





st.code("""

koch = pl.from_pandas(koch)
koch = koch.with_columns(pl.col('CREATE_DATE').cast(pl.Utf8))
koch = koch.with_columns(
    pl.col("CREATE_DATE").str.strptime(pl.Date, "%Y%m%d").alias("CREATE_DATE")
)
koch = koch.with_columns([
    pl.col('CREATE_DATE').dt.weekday().alias('day_of_week'),  # Monday=0, Sunday=6
    pl.col('CREATE_DATE').dt.month().alias('month'),
    pl.col('CREATE_DATE').dt.quarter().alias('quarter')
])
""")



st.code("""

koch = koch.with_columns(pl.col('DELIVERY_DATE').cast(pl.Utf8))
koch = koch.with_columns(
    pl.col("DELIVERY_DATE").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S.%f")
)

koch = koch.with_columns([
    pl.col('DELIVERY_DATE').dt.weekday().alias('D_day_of_week'),  # Monday=0, Sunday=6
])
""")





st.code("""

koch['PLANNED_DELIVERY_DAYS'] = pd.to_numeric(koch['PLANNED_DELIVERY_DAYS'], errors='coerce')

material_stats = koch.groupby('MATERIAL_ID')['PLANNED_DELIVERY_DAYS'].agg(['mean', 'median']).reset_index()

material_stats.columns = ['MATERIAL_ID', 'Mean_Delay_Material', 'Mean_Delay_Material']
""")



st.code("""

mean_delay_corr_material = koch['Mean_Delay_Material'].corr(koch['PLANNED_DELIVERY_DAYS'])
# Check correlation between Median_Delay and Time_Difference
median_delay_corr_mateial = koch['Mean_Delay_Material'].corr(koch['PLANNED_DELIVERY_DAYS'])


print(mean_delay_corr_material)
print(median_delay_corr_mateial)
""")




st.code("""

from datetime import date
import holidays
import pandas as pd

# Define the countries and years
countries = {
    'US': 'United States',
    'CA': 'Canada',
    'GB': 'United Kingdom',
    'CN': 'China',
    'JP': 'Japan',
    'NL': 'Netherlands',
    'KR': 'South Korea',
    'SG': 'Singapore',
    'DE': 'Germany',
    'FR': 'France'
}

years = [2018, 2019, 2020, 2021, 2022]

# Initialize a dictionary to store holidays for each country and year
federal_holidays = {}

# Loop through each country and each year to retrieve holidays
for country_code, country_name in countries.items():
    country_holidays = {}
    for year in years:
        # Retrieve holidays for the current country and year
        holidays_obj = holidays.CountryHoliday(country_code, years=year)
        country_holidays[year] = [date.strftime('%m-%d') for date in holidays_obj]
    # Store holidays for the current country in the dictionary
    federal_holidays[country_code] = country_holidays


# Function to check if a date is a holiday for a given country
def is_holiday(row):
    country = row['COUNTRY']
    create_date = row['CREATE_DATE']
    year = int(create_date[:4])  # Extract the year from the Formatted_Date
    return 1 if create_date[5:] in federal_holidays.get(country, {}).get(year, []) else 0
""")



st.code("""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier, DMatrix, train
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

df = koch
# Summoning the dataset

label_encoder = LabelEncoder()
categorical_columns = ['PLANNED_DELIVERY_DAYS', 'SUB_COMMODITY_DESC', 'COMPANY_CODE_ID']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Convert datetime columns to separate year, month, day columns
datetime_columns = ['REQUESTED_DELIVERY_DATE', 'CREATE_DATE']
for col in datetime_columns:
    df[col + '_year'] = df[col].dt.year
    df[col + '_month'] = df[col].dt.month
    df[col + '_day'] = df[col].dt.day

# Drop the original datetime columns
df.drop(columns=datetime_columns, inplace=True)

X = koch.drop('Time_Difference', axis=1)  # Replace 'target_column' with your target column name
y = koch['Time_Difference']

""")


#---------------------------------------------------------------------------------------------------------------
st.title("Subset Model")
#---------------------------------------------------------------------------------------------------------------

st.write("## Mean Absolute Error (MAE): 12.446554899726546")
st.write("## Mean Squared Error (MSE): 1178.0400238236136")
st.write("## R-squared Score: 0.295581301714428")



image = Image.open(r'model_test.png')

st.image(image)

st.code("""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Drop rows with missing values (or you can impute them as well)
df = koch.dropna()

# Assuming 'SHORT_TEXT', 'REQUESTED_DELIVERY_DATE', 'COMPANY_CODE_ID', 'CREATE_DATE', and 'SUB_COMMODITY_DESC' are categorical
categorical_features = ['SHORT_TEXT', 'REQUESTED_DELIVERY_DATE', 'COMPANY_CODE_ID', 'CREATE_DATE', 'SUB_COMMODITY_DESC']
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.drop(['Time_Difference']).tolist()

# Define preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split the dataset into features and target variable
X = df.drop(['Time_Difference'], axis=1)
y = df['Time_Difference']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
""")



st.code("""
import tensorflow as tf


def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=[input_shape]),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

""")



st.code("""
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

""")



st.code("""
# Import necessary libraries
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Drop rows with missing values (or you can impute them as well)
df = koch.dropna()

# Assuming 'SHORT_TEXT', 'REQUESTED_DELIVERY_DATE', 'COMPANY_CODE_ID', 'CREATE_DATE', and 'SUB_COMMODITY_DESC' are categorical
categorical_features = ['SHORT_TEXT', 'REQUESTED_DELIVERY_DATE', 'COMPANY_CODE_ID', 'CREATE_DATE', 'SUB_COMMODITY_DESC']
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.drop(['Time_Difference']).tolist()

# Define preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split the dataset into features and target variable
X = df.drop(['Time_Difference'], axis=1)
y = df['Time_Difference']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the input features (optional but recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the neural network model
model = Sequential()

# Add input layer with ReLU activation
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))

# Add hidden layers with ReLU activation
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))

# Add output layer with sigmoid activation (for binary classification)
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

""")



st.code("""
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

# Load your dataframe here
# Assuming 'df' is your DataFrame name and it has been loaded correctly
# df = pd.read_csv('your_dataframe.csv')  # Example for loading data

# Drop rows with missing values (or you can impute them as well)
df.dropna(inplace=True)

# Assuming 'SHORT_TEXT', 'REQUESTED_DELIVERY_DATE', 'COMPANY_CODE_ID', 'CREATE_DATE', and 'SUB_COMMODITY_DESC' are categorical
categorical_features = ['SHORT_TEXT', 'REQUESTED_DELIVERY_DATE', 'COMPANY_CODE_ID', 'CREATE_DATE', 'SUB_COMMODITY_DESC']
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.drop(['Time_Difference']).tolist()

# Define preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split the dataset into features and target variable
X = df.drop(['Time_Difference'], axis=1)
y = df['Time_Difference']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the input features
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Define the neural network model
model = Sequential()

# Add input layer with ReLU activation
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))

# Add hidden layers with ReLU activation
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))

# Modify the output layer for regression with no activation function
model.add(Dense(units=1, activation='linear'))

# Compile the model for a regression problem
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on test data
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

""")



st.code("""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Assuming your DataFrame is named `df`
# df = pd.read_csv('path_to_your_data.csv')  # Load your data here

# Drop rows with missing values (or you can impute them as well)
df = koch.dropna(inplace=True)

# Define categorical and numerical features
categorical_features = ['SHORT_TEXT', 'REQUESTED_DELIVERY_DATE', 'COMPANY_CODE_ID', 'CREATE_DATE', 'SUB_COMMODITY_DESC']
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.drop(['Time_Difference']).tolist()

# Define preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Split the dataset into features and target variable
X = df.drop(['Time_Difference'], axis=1)
y = df['Time_Difference']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing and convert sparse matrices to dense arrays
X_train = preprocessor.fit_transform(X_train).toarray()
X_test = preprocessor.transform(X_test).toarray()

# Neural network model definition
model = Sequential([
    Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(units=128, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=1, activation='linear')  # Linear activation for regression
])

# Compile the model for a regression problem
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on test data
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
""")






st.code("""



""")












