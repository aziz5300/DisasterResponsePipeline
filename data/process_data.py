#all imports
import sys
import pandas as pd
from sqlalchemy import create_engine

#this function is used to load data into dataframes and merge them on id column
def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


#this function is used to clean the data and return a concatenated dataframe
def clean_data(df):
    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames

    # only keep numeric values in the cells of the category columns and drop all text
    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])

        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: int(x))

    # drop the categories column from df since it is no longer needed
    df.drop('categories', axis=1, inplace=True)

    # concatenate df and categories data frames
    df = pd.concat([df, categories], axis=1)

    # remove duplicate rows
    df.drop_duplicates(inplace=True)

    return df




    # categories = pd.DataFrame(df.categories.str.split(';',-1).tolist())
    # row = categories.iloc[0]
    # category_colnames = []
    # for i in range(0, len(categories.iloc[0])):
    #     category_colnames.append(categories.iloc[0][i].split('-')[0])
    # print(category_colnames)
    # cols = categories.columns
    # for col in cols:
    #     categories[col] = categories[col].astype(str).str[-1:].astype(int)
    # df.drop(['categories'], axis=1, inplace=True)
    # frames = [df, categories]
    # df1 = pd.concat(frames, axis=1)
    # df1.drop_duplicates(inplace=True)
    # return df1


#this function is used to save the data into SQL table
def save_data(df, database_filename):
    Database_ = 'sqlite:///' + database_filename
    engine = create_engine(Database_)
    df.to_sql('Disaster_Response', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()