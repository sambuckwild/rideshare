import pandas as pd


def to_date(df):
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])

    return df


def drop_nan_ratings(df):
    df.dropna(subset=['avg_rating_by_driver', 'avg_rating_of_driver', 'phone'], inplace=True)
    return df


def create_churn_col(df, last_trip, date):
    '''Inputs: df - current dataframe
               last_trip - string of last_trip datetime column
               date - date string to sort at for active versus not active user
        Creates a boolean column for if the user churned or not based on last trip
        Output: df with new churn column boolean'''
    df['churn'] = df[last_trip] < date
    return df


def bool_to_int(df, col_lst):
    '''Inputs: df - current dataframe
               col_lst - list of boolean columns
        Changes boolean values to 1 and 0
        Output - df with new int columns'''
    for col in col_lst:
        df[col] = df[col].astype(int)
    return df


def hot_encode(df):
    df.replace(to_replace={"Android": 1, "iPhone": 0}, inplace=True)

    return df


<<<<<<< HEAD:.ipynb_checkpoints/clean-checkpoint.py
=======
def drop_cols(df):
    del df['city']
    del df['last_trip_date']
    del df['signup_date']

    return df


>>>>>>> origin/robert:clean.py
if __name__ == '__main__':
    # test_file = 'data/churn_train.csv'
    file = (input("Enter path of filename: "))
    df = pd.read_csv(file)
    df = to_date(df)
    df = create_churn_col(df, 'last_trip_date', '2014-06-01')
    df = bool_to_int(df, ['churn', 'luxury_car_user'])
    df = hot_encode(df)
    df = drop_nan_ratings(df)
    df.to_csv('data/churn_clean.csv')




