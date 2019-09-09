# ------------------------------------------------------------------------------------------------------
# Licensed under MIT License
# Written by sayooj_bk
# -------------------------------------------------------------------------------------------------------

import os
import re
import shutil
import warnings
import pandas as pd
from pandas import DataFrame

# The below data are which have been uploaded by the user while filling out form preference.
from ..configs.constants import (
    DEFAULT_USER_COL,
    DEFAULT_USER_BUDGET,
    DEFAULT_PURCHASED_COUNT,
    DEFAULT_NATIONALITY ,
    DEFAULT_ITEM_COL,
    DEFAULT_ITEM_CATEGORY,
    DEFAULT_RETAIL_PRICE,
    MERCHANT_PRODUCT_PRICE,
    DISCOUNTED_PRICE,
    MERCHANT_CATEGORY_ID,
    PRICE_LEVEL
)


import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")

# Load the fcgl database which has to be loaded and
"""
FCGL-DATABASE :
COLLECTIONS : USER_DATA
              PRODUCT_DATA
              MERCHANT_DATA
              PRODUCT_CATEGORY_DATA
              MERCHANT_CATEGORY_DATA
              MERCHANT_PRODUCT_DATA
              USER_PURCHASES_DATA.


""" 
mydb = myclient["fcgl-database"]          
training_collection = mydb["training_data"]   
# training_collection.insert(dict)   -- this is where we will add the data to be trained to the colleciton.

ERROR_HEADER = "Header error. At least user and item id should be provided."
WARNING_FCGL_HEADER = "The dataset has more than the required data so only few columns will be used #TODO select those few."
"""
try:
    from pyspark.sql.types import (
        StructType,
        StructField,
        IntegerType,
        FloatType,
        DoubleType,
        LongType,
        StringType,
    )
    from pyspark.sql.functions import concat_ws, col
except ImportError:
    pass  # so the environment without spark doesn't break
"""

# The data points on which the recommendation system will be trained.
DEFAULT_HEADER = (
    DEFAULT_USER_COL,
    DEFAULT_USER_BUDGET,
    DEFAULT_PURCHASED_COUNT,
    DEFAULT_NATIONALITY,
    DEFAULT_ITEM_COL,                       # This is the data point to be evaluated or in a sense this has to be the result of the model.
    DEFAULT_ITEM_CATEGORY,
    DEFAULT_RETAIL_PRICE,
    # MERCHANT_PRODUCT_PRICE,               # These three are the closest we have to find for the product which has to be recommended.
    # DISCOUNTED_PRICE,                     # based on lowest cost and closest distance.
    # PRICE_LEVEL
)


def load_pandas_df(
    header=None):
    """
    Loads the Mongodb  dataset as pd.DataFrame.

    Args: 
        mydb -- The database which hosts  the collection of datasets which are to be transformed to pandas dataframe for training the recommendation system.
        
    Returns:
        pd.DataFrame: The User - product relation dataset.
        

    **Examples**

    .. code-block:: python
    
        # To load just user-id, item-id, and ratings from MovieLens-1M dataset,
        df = load_pandas_df('1m', ('UserId', 'ItemId', 'Rating'))

        # To load rating's timestamp together,
        df = load_pandas_df('1m', ('UserId', 'ItemId', 'Rating', 'Timestamp'))

        # To load movie's title, genres, and released year info along with the ratings data,
        df = load_pandas_df('1m', ('UserId', 'ItemId', 'Rating', 'Timestamp'),
            title_col='Title',
            genres_col='Genres',
            year_col='Year'
        )
    """
    if header is None:
        header = DEFAULT_HEADER
    elif len(header) < 2:
        raise ValueError(ERROR_HEADER)
    elif len(header) > 7:
        warnings.warn(WARNING_FCGL_HEADER)

    
    # collection_name : Collection name that is used to train the vowpal wabbit model.
    # collection_name : Since this is what helps in recommnding we will push in the intial user data we get from the user while he/she
    # signups for the application.

    # data provided by the user.
    user_data = mydb["user_data"]
    user_purchase_data = mydb["user_purchase_data"]
    product_data = mydb["product_data"]
    product_category_data = mydb["product_category_data"]
    
    for user in user_data :
        new_data = user
        # This new data = {"_id":mongodb id provided,  user_id : "" , nationality:"" , average_spent: "" }
        try: 
            del new_data['_id']
        except KeyError: 
            pass
        
        # This get the purhcase query data which has item id and purchase count
        purchase_query = dict()
        purchase_query['user_id'] = new_data['user_id']
       
        user_purchase = user_purchase_data.find(purchase_query)
        
        for key in ['_id' , 'added_on' , 'last_updated' , 'user_id'] :
            try: 
                del user_purchase[key]
            except KeyError: 
                pass
        
        # user_purchase = { product_id , purchase_count}

        product_query = dict()
        product_query['product_id'] = user_purchase['product_id']

        product_category = product_category_data.find(product_query)
        product_price    = product_data.find(product_query)

        for key in ['_id' , 'added_on' , 'last_updated', 'product_id'] :
            try: 
                del product_category[key]
            except KeyError: 
                pass

        for key in ['_id' , 'added_on' , 'last_updated' , 'name' , 'categories', 'product_id'] :
            try: 
                del product_price[key]
            except KeyError: 
                pass
        
        # prodcut-cat = {category}
        # product price ={price}
    

        new_data.update(user_purchase)
        new_data.update(product_category)
        new_data.update(product_price)

        # new_data = {"user_id" , "nationality" , "item_id" , "purchase_count" , "budget" , "item_cat" , "item_retail_price."}
        training_collection.insert_one(new_data) 


    df = DataFrame( list(mydb.training_data.find({})) )
    
    return df

def load_spark_df(
    spark):
    raise NotImplementedError


