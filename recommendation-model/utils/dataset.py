# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

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
COLLECTIONS : USER DATA
              PRODUCT DATA
              MERCHANT DATA


""" 
mydb = myclient["fcgl-database"]          
training_collection = mydb["training_data"]
# training_collection.insert(dict)   -- this will add the data to the colleciton.

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
    DEFAULT_ITEM_COL,
    DEFAULT_ITEM_CATEGORY,
    DEFAULT_RETAIL_PRICE,
    MERCHANT_PRODUCT_PRICE,
    DISCOUNTED_PRICE,
    MERCHANT_CATEGORY_ID,
    PRICE_LEVEL
)



def load_pandas_df(
    header=None):
    """
    Loads the Mongodb  dataset as pd.DataFrame.
    To load customer information only, you can use load_item_df function.

    Args:
        size (str): Size of the data to load. One of ("100k", "1m", "10m", "20m").
        header (list or tuple or None): Rating dataset header.
        local_cache_path (str): Path (directory or a zip file) to cache the downloaded zip file.
            If None, all the intermediate files will be stored in a temporary directory and removed after use.
        title_col (str): Movie title column name. If None, the column will not be loaded.
        genres_col (str): Genres column name. Genres are '|' separated string.
            If None, the column will not be loaded.
        year_col (str): Movie release year column name. If None, the column will not be loaded.

    Returns:
        pd.DataFrame: Movie rating dataset.
        

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
    elif len(header) > 4:
        warnings.warn(WARNING_FCGL_HEADER)
        # header = header[:4]

    fcgl_column = header[1]
    # collection_name : Collection name that is used to train the vowpal wabbit model.
    # collection_name : Since this is what helps in recommnding we will push in the intial user data we get from the user while he/she
    # signups for the application.


    df = DataFrame( list(mydb.training_data.find({})) )
    
    return df


def load_item_df(
    movie_col=DEFAULT_ITEM_COL,
    title_col=None,
    genres_col=None,
    year_col=None):
    """Loads Movie info.

    Args:
        movie_col (str): Movie id column name.
        title_col (str): Movie title column name. If None, the column will not be loaded.
        genres_col (str): Genres column name. Genres are '|' separated string.
            If None, the column will not be loaded.
        year_col (str): Movie release year column name. If None, the column will not be loaded.

    Returns:
        pd.DataFrame: Movie information data, such as title, genres, and release year.
    """
    size = size.lower()
    if size not in DATA_FORMAT:
        raise ValueError(ERROR_MOVIE_LENS_SIZE)

    with download_path(local_cache_path) as path:
        filepath = os.path.join(path, "ml-{}.zip".format(size)) 
        _, item_datapath = _maybe_download_and_extract(size, filepath)
        item_df = _load_item_df(
            size, item_datapath, movie_col, title_col, genres_col, year_col
        )

    return item_df


def _load_item_df(size, item_datapath, movie_col, title_col, genres_col, year_col):
    """Loads Movie info"""
    if title_col is None and genres_col is None and year_col is None:
        return None

    item_header = [movie_col]
    usecols = [0]

    # Year is parsed from title
    if title_col is not None or year_col is not None:
        item_header.append("title_year")
        usecols.append(1)

    genres_header_100k = None
    if genres_col is not None:
        # 100k data's movie genres are encoded as a binary array (the last 19 fields)
        # For details, see http://files.grouplens.org/datasets/movielens/ml-100k-README.txt
        if size == "100k":
            genres_header_100k = [*(str(i) for i in range(19))]
            item_header.extend(genres_header_100k)
            usecols.extend([*range(5, 24)])  # genres columns
        else:
            item_header.append(genres_col)
            usecols.append(2)  # genres column

    item_df = pd.read_csv(
        item_datapath,
        sep=DATA_FORMAT[size].item_separator,
        engine="python",
        names=item_header,
        usecols=usecols,
        header=0 if DATA_FORMAT[size].item_has_header else None,
        encoding="ISO-8859-1",
    )

    # Convert 100k data's format: '0|0|1|...' to 'Action|Romance|..."
    if genres_header_100k is not None:
        item_df[genres_col] = item_df[genres_header_100k].values.tolist()
        item_df[genres_col] = item_df[genres_col].map(
            lambda l: "|".join([GENRES[i] for i, v in enumerate(l) if v == 1])
        )

        item_df.drop(genres_header_100k, axis=1, inplace=True)

    # Parse year from movie title. Note, MovieLens title format is "title (year)"
    # Note, there are very few records that are missing the year info.
    if year_col is not None:

        def parse_year(t):
            parsed = re.split("[()]", t)
            if len(parsed) > 2 and parsed[-2].isdecimal():
                return parsed[-2]
            else:
                return None

        item_df[year_col] = item_df["title_year"].map(parse_year)
        if title_col is None:
            item_df.drop("title_year", axis=1, inplace=True)

    if title_col is not None:
        item_df.rename(columns={"title_year": title_col}, inplace=True)

    return item_df


def load_spark_df(
    spark):
    raise NotImplementedError


