import pandas as pd
import numpy as np
from pydataset import data
import matplotlib.pyplot as plt
import seaborn as sns
import os
from env import host, user, password

def get_connection(db, user=user, host=host, password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_mall_data():
    '''
    This function reads the mall customer data from the Codeup database into a df,
    writes it to a csv file, and returns the df.
    '''
    sql_query = 'SELECT * FROM customers'
    df = pd.read_sql(sql_query, get_connection('mall_customers'))
    df.to_csv('mall_customers_df.csv')
    return df

def get_mall_data(cached=False):
    '''
    This function reads in the mall customer from Codeup database if cached == False
    or if cached == True reads in mall customer df from a csv file, returns df
    '''
    if cached or os.path.isfile('mall_customers_df.csv') == False:
        df = new_mall_data()
    else:
        df = pd.read_csv('mall_customers_df.csv', index_col=0)
    return df

def get_telco_data():
    if os.path.isfile('telco.csv') == False:
        sql_query = """
                    SELECT customer_id,
                    contract_type_id,
                    phone_service,
                    internet_service_type_id,
                    gender,
                    senior_citizen,
                    partner,
                    dependents,
                    tenure,
                    online_security,
                    online_backup,
                    device_protection,
                    tech_support,
                    streaming_tv,
                    streaming_movies,
                    monthly_charges,
                    total_charges,
                    churn
                    FROM customers
                    """
        df = pd.read_sql(sql_query, get_connection('telco_churn'))
        df.to_csv('telco.csv')
        return df
    else:
        df = pd.read_csv('telco.csv', index_col=0)
        return df

def new_telco_data():
    sql_query = """
                SELECT customer_id,
                contract_type_id,
                phone_service,
                internet_service_type_id,
                gender,
                senior_citizen,
                partner,
                dependents,
                tenure,
                online_security,
                online_backup,
                device_protection,
                tech_support,
                streaming_tv,
                streaming_movies,
                monthly_charges,
                total_charges,
                churn
                FROM customers                    """
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    df.to_csv('telco.csv')
    return df