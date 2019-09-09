
import sys
sys.path.append('../..')

import os
from subprocess import run

import pandas as pd
import papermill as pm


from ..utils.evaluation import (rmse, mae, exp_var, rsquared, get_top_k_items,
                                                     map_at_k, ndcg_at_k, precision_at_k, recall_at_k)

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))


def to_vw(df, output, logistic=False):
    """Convert Pandas DataFrame to vw input format
    Args:
        df (pd.DataFrame): input DataFrame
        output (str): path to output file
        logistic (bool): flag to convert label to logistic value
    """
    with open(output, 'w') as f:
        tmp = df.reset_index()

        # we need to reset the rating type to an integer to simplify the vw formatting
        tmp['purchase_count'] = tmp['purchase_count'].astype('int64')
        
        # convert rating to binary value
        if logistic:
            tmp['purchase_count'] = tmp['puchase_count'].apply(lambda x: 1 if x >= 1 else -1)
        
        # convert each row to VW input format (https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format)
        # [label] [tag]|[user namespace] [user id feature] |[item namespace] [movie id feature]
        # label is the true rating, tag is a unique id for the example just used to link predictions to truth
        # user and item namespaces separate the features to support interaction features through command line options
        for _, row in tmp.iterrows():
            f.write('{purchase_count:d} {index:d}|user {userID:d} |item {productID:d}\n'.format_map(row))


def run_vw(train_params, test_params, test_data, prediction_path, logistic=False):
    """Convenience function to train, test, and show metrics of interest
    Args:
        train_params (str): vw training parameters
        test_params (str): vw testing parameters
        test_data (pd.dataFrame): test data
        prediction_path (str): path to vw prediction output
        logistic (bool): flag to convert label to logistic value
    Returns:
        (dict): metrics and timing information
    """

    # train model
    train_start = process_time()
    run(train_params.split(' '), check=True)
    train_stop = process_time()
    
    # test model
    test_start = process_time()
    run(test_params.split(' '), check=True)
    test_stop = process_time()
    
    # read in predictions
    pred_df = pd.read_csv(prediction_path, delim_whitespace=True, names=['prediction'], index_col=1).join(test_data)
    pred_df.drop("purchase_count", axis=1, inplace=True)

    test_df = test_data.copy()
    if logistic:
        # make the true label binary so that the metrics are captured correctly
        # test_df['purchase_count'] = test['purchase_count'].apply(lambda x: 1 if x >= 3 else -1)  # not sure if this was an error in the intial implementation need to get it checked with our dataset.
        test_df['purchase_count'] = test_df['purchase_count'].apply(lambda x: 1 if x >= 1 else -1)
    else:
        # ensure results are integers in correct range  
        # This 5 is for the movie ratings range not sure what to keep for our recommendation.  ???????????????????
        pred_df['prediction'] = pred_df['prediction'].apply(lambda x: int(max(1, min(5, round(x)))))

    # calculate metrics
    result = dict()
    result['RMSE'] = rmse(test_df, pred_df)
    result['MAE'] = mae(test_df, pred_df)
    result['R2'] = rsquared(test_df, pred_df)
    result['Explained Variance'] = exp_var(test_df, pred_df)
    result['Train Time (ms)'] = (train_stop - train_start) * 1000
    result['Test Time (ms)'] = (test_stop - test_start) * 1000

    return result



