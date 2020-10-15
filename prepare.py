import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

from acquire import get_telco_data, new_telco_data

# Prep Telco Churn Data

# The function below will acquire the data with get_telco_data from acquire.py and prepare the features to use in data exploration.
def prep_telco():
    '''
    No arguments taken. Returns a df of the telco customer churn data in the Codeup SQL database.
    '''
    # acquire the data from csv (or sql if no csv exists) and assign it as a DataFrame
    df = get_telco_data()

    # set the index to be customer_id
    df = df.set_index('customer_id')

    # cut down on the complexity of data by combining the variables for add-on packages
    # into one variable, num_add_ons, that adds up the number of add-on services each customer has

    # map strings into 1s and 0s, 1 meaning that the customer has that add-on
    df.online_security = df.online_security.map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df.online_backup = df.online_backup.map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df.device_protection = df.device_protection.map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df.tech_support = df.tech_support.map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df.streaming_tv = df.streaming_tv.map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df.streaming_movies = df.streaming_movies.map({'Yes': 1, 'No': 0, 'No internet service': 0})

    # add and put into new column, num_add_ons (number of add-on services)
    df['num_add_ons'] = (df.online_security + df.online_backup + df.device_protection + df.tech_support + df.streaming_tv + df.streaming_movies)
    
    # drop the add-on columns we don't need anymore.
    df = df.drop(columns=['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies'])

    # encode phone_service and gender
    df.phone_service = df.phone_service.map({'Yes': 1, 'No': 0})
    # make new column is_male and encode
    df['is_male'] = df.gender.map({'Male': 1, 'Female': 0})
    # Now drop gender column
    df = df.drop(columns=['gender'])

    # encode partner, dependents, and churn
    df.partner = df.partner.map({'Yes': 1, 'No': 0})
    df.dependents = df.dependents.map({'Yes': 1, 'No': 0})
    df.churn = df.churn.map({'Yes': 1, 'No': 0})

    # Right now the contract types are map from 1 to 3, so I'm going to re-map it starting from 0.
    # 0 is Month-to-Month
    # 1 is Yearly
    # 2 is Two Year
    df.contract_type_id = df.contract_type_id.map({1: 0, 2: 1, 3: 2})

    # Going to re-map the internet_type_id column in the same way.
    # 0 is No Internet Service
    # 1 is DSL
    # 2 is Fiber Optic
    df.internet_service_type_id = df.internet_service_type_id.map({3: 0, 1: 1, 2: 2})

    # Now to add a column called tenure_yrs that represents tenure in years
    df['tenure_yrs'] = round((df.tenure / 12), 2)

    # Rename columns to shorter names for readability and ease of use:
    # contract_type_id --> contract_type
    # phone_service --> phone
    # internet_service_type_id --> internet_type
    # senior_citizen --> senior
    # dependents --> depend
    df = df.rename(columns={"contract_type_id": "contract_type", "phone_service": "phone",
                    "internet_service_type_id": "internet_type", "senior_citizen": "senior", "dependents": "depend"})
    
    # convert total_charges to a float and drop the rows with null total_charges
    df['total_charges'] = pd.to_numeric(df['total_charges'],errors='coerce')
    df = df[~df.total_charges.isnull()]

    # Data is now tidy and ready to split into train, validate, and test.

    return df

def telco_split(df):
    '''
    Splits data into train, validate, and test.
    '''
    # performs train, validate, test split on telco data, stratified by churn
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=666, 
                                        stratify=df.churn)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=666, 
                                   stratify=train_validate.churn)
    return train, validate, test

def telco_split_xy(df, target):
    '''
    Given the dataframe and the target (the target variable you want to predict on), returns X_train, X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test.
    '''
    # split data 
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=666, 
                                        stratify=df.churn)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=666, 
                                   stratify=train_validate.churn)

    # split again, into x and y
    # from Maggie's lesson notes: split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

def telco_X_scale(X_train, X_validate, X_test):
    '''
    Takes arguments X_train, X_validate, X_test. Run after using telco_split_xy function. Returns X_train_scaled, X_validate_scaled, and X_test_scaled.
    '''
    scaler = MinMaxScaler(copy=True).fit(X_train)

    X_train_scaled_array = scaler.transform(X_train)
    X_validate_scaled_array = scaler.transform(X_validate)
    X_test_scaled_array = scaler.transform(X_test)

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array).set_index([X_train.index.values])

    X_validate_scaled = pd.DataFrame(X_validate_scaled_array).set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array).set_index([X_test.index.values])

    X_train_scaled.columns = ['contract_type', 'phone', 'internet_type', 'senior', 'partner',
       'depend', 'tenure', 'monthly_charges', 'total_charges', 'num_add_ons',
       'is_male', 'tenure_yrs']

    X_validate_scaled.columns = ['contract_type', 'phone', 'internet_type', 'senior', 'partner',
    'depend', 'tenure', 'monthly_charges', 'total_charges', 'num_add_ons',
    'is_male', 'tenure_yrs']

    X_test_scaled.columns = ['contract_type', 'phone', 'internet_type', 'senior', 'partner',
    'depend', 'tenure', 'monthly_charges', 'total_charges', 'num_add_ons',
    'is_male', 'tenure_yrs']

    return X_train_scaled, X_validate_scaled, X_test_scaled

def prep_mall_data(df):
    '''
    Takes the acquired mall data, does data prep,
    and returns train, test, and validate data splits.
    '''
    df['is_female'] = (df.gender == 'Female').astype('int')
    train_and_validate, test = train_test_split(df, test_size=.15, random_state=666)
    train, validate = train_test_split(train_and_validate, test_size=.15, random_state=666)
    return train, test, validate

def wrangle_grades():
    '''
    Reads student_grades csv and returns a df.
    '''
    grades = pd.read_csv("student_grades.csv")
    grades.drop(columns="student_id", inplace=True)
    grades.replace(r"^\s*$", np.nan, regex=True, inplace=True)
    df = grades.dropna().astype("int")
    return df