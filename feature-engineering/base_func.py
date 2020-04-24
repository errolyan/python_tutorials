# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： base_func.py
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2020/1/5  15:19
-------------------------------------------------
   Change Activity:
                  2020/1/5  15:19:
-------------------------------------------------
'''
__author__ = 'yanerrol'

def drop_multiple_col(col_names_list, df):
    '''
    AIM    -> Drop multiple columns based on their column names
    INPUT  -> List of column names, df
    OUTPUT -> updated df with dropped columns
    '''
    df.drop(col_names_list, axis=1, inplace=True)
    return df

def change_dtypes(col_int, col_float, df):
    '''
    AIM    -> Changing dtypes to save memory
    INPUT  -> List of column names(int, float), df
    OUTPUT -> updated df with smaller memory
    '''
    df[col_int] = df[col_int].astype(int32)
    df[col_float] = df[col_float].astype(float32)

def convert_cat2num(df):
    # Convert categorical variable to numerical variable
    num_encode = {col_1: {YES: 1, NO: 0},
                  col_2: {WON: 1, LOSE: 0, DRAW: 0}}
    df.replace(num_encode, inplace=True)

def check_missing_data(df):
    # check for any missing data in the df (display in descending order)
    return df.isnull().sum().sort_values(ascending=False)

def remove_col_str(df):
    # remove a portion of string in a dataframe column - col_1
    df[ col_1 ].replace(, , regex=True, inplace=True)

    # remove all the characters after &# (including &#) for column - col_1
    df[ col_1 ].replace(.* , , regex=True, inplace=True)

def remove_col_white_space (df):
    # remove white space at the beginning of string
    df[col] = df[col].str.lstrip()

# 字符串拼接两列
def concat_col_str_condition (df):
    # concat 2 columns with strings if the last 3 letters of the first column are  pil
    mask = df[ col_1 ].str.endswith( pil , na=False)
    col_new = df[mask][ col_1 ] + df[mask][ col_2 ]
    col_new.replace( pil ,    , regex=True, inplace=True)  # replace the  pil  with emtpy space

# 转换时间戳（从字符串到日期时间格式）
def convert_str_datetime(df):
    '''
    AIM    -> Convert  datetime(String）to datetime(format we  want)
    INPUT  -> df
    OUTPUT -> updated df with new datetime format
    '''
    df.insert(loc=2, column=timestamp,
              value=pd.to_datetime(df.transdate, format= % Y - % m - % d % H: % M: % S. % f ))