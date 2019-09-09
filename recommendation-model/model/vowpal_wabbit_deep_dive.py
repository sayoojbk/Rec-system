
import sys
sys.path.append('../..')

import os
from subprocess import run
from tempfile import TemporaryDirectory
from time import process_time

import pandas as pd
import papermill as pm


from ..utils.dataset import load_pandas_df
from ..utils.python_splitters import python_random_split
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
        test_df['purchase_count'] = test['purchase_count'].apply(lambda x: 1 if x >= 3 else -1)
    else:
        # ensure results are integers in correct range  
        # This 5 is for the movie ratings range not sure what to keep for our recommendation. 
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

# create temp directory to maintain data files
tmpdir = TemporaryDirectory()

model_path = os.path.join(tmpdir.name, 'vw.model')
saved_model_path = os.path.join(tmpdir.name, 'vw_saved.model')
train_path = os.path.join(tmpdir.name, 'train.dat')
test_path = os.path.join(tmpdir.name, 'test.dat')
train_logistic_path = os.path.join(tmpdir.name, 'train_logistic.dat')
test_logistic_path = os.path.join(tmpdir.name, 'test_logistic.dat')
prediction_path = os.path.join(tmpdir.name, 'prediction.dat')
all_test_path = os.path.join(tmpdir.name, 'new_test.dat')
all_prediction_path = os.path.join(tmpdir.name, 'new_prediction.dat')


# # 1. TOP Recommendations to be displayed.
TOP_K = 5

# load movielens data 
df = load_pandas_df()

# split data to train and test sets, default values take 75% of each users ratings as train, and 25% as test
train, test = python_random_split(df, 0.75)

# save train and test data in vw format
to_vw(df=train, output=train_path)
to_vw(df=test, output=test_path)

# save data for logistic regression (requires adjusting the label)
to_vw(df=train, output=train_logistic_path, logistic=True)
to_vw(df=test, output=test_logistic_path, logistic=True)


# # 2. Regression Based Recommendations
# 
# When considering different approaches for solving a problem with machine learning it is helpful to generate a baseline approach to understand how more complex solutions perform across dimensions of performance, time, and resource (memory or cpu) usage.
# 
# Regression based approaches are some of the simplest and fastest baselines to consider for many ML problems.

# ## 2.1 Linear Regression
# 
# As the data provides a numerical rating between 1-5, fitting those values with a linear regression model is easy approach. This model is trained on examples of ratings as the target variable and corresponding user ids and movie ids as independent features.
# 
# By passing each user-item rating in as an example the model will begin to learn weights based on average ratings for each user as well as average ratings per item.
# 
# This however can generate predicted ratings which are no longer integers, so some additional adjustments should be made at prediction time to convert them back to the integer scale of 1 through 5 if necessary. Here, this is done in the evaluate function.

"""
Quick description of command line parameters used
  Other optional parameters can be found here: https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Command-Line-Arguments
  VW uses linear regression by default, so no extra command line options
  -f <model_path>: indicates where the final model file will reside after training
  -d <data_path>: indicates which data file to use for training or testing
  --quiet: this runs vw in quiet mode silencing stdout (for debugging it's helpful to not use quiet mode)
  -i <model_path>: indicates where to load the previously model file created during training
  -t: this executes inference only (no learned updates to the model)
  -p <prediction_path>: indicates where to store prediction output
"""
train_params = 'vw -f {model} -d {data} --quiet'.format(model=model_path, data=train_path)
# save these results for later use during top-k analysis
test_params = 'vw -i {model} -d {data} -t -p {pred} --quiet'.format(model=model_path, data=test_path, pred=prediction_path)

result = run_vw(train_params=train_params, 
                test_params=test_params, 
                test_data=test, 
                prediction_path=prediction_path)

comparison = pd.DataFrame(result, index=['Linear Regression'])
comparison


# ## 2.2 Linear Regression with Interaction Features
# 
# Previously we treated the user features and item features independently, but taking into account interactions between features can provide a mechanism to learn more fine grained preferences of the users.
# 
# To generate interaction features use the quadratic command line argument and specify the namespaces that should be combined: '-q ui' combines the user and item namespaces based on the first letter of each.
# 
# Currently the userIDs and itemIDs used are integers which means the feature ID is used directly, for instance when user ID 123 rates movie 456, the training example puts a 1 in the values for features 123 and 456. However when interaction is specified (or if a feature is a string) the resulting interaction feature is hashed into the available feature space. Feature hashing is a way to take a very sparse high dimensional feature space and reduce it into a lower dimensional space. This allows for reduced memory while retaining fast computation of feature and model weights.
# 
# The caveat with feature hashing, is that it can lead to hash collisions, where separate features are mapped to the same location. In this case it can be beneficial to increase the size of the space to support interactions between features of high cardinality. The available feature space is dictated by the --bit_precision (-b) <N> argument. Where the total available space for all features in the model is 2<sup>N</sup>. 
# 

"""
Quick description of command line parameters used
  -b <N>: sets the memory size to 2<sup>N</sup> entries
  -q <ab>: create quadratic feature interactions between features in namespaces starting with 'a' and 'b' 
"""
train_params = 'vw -b 26 -q ui -f {model} -d {data} --quiet'.format(model=saved_model_path, data=train_path)
test_params = 'vw -i {model} -d {data} -t -p {pred} --quiet'.format(model=saved_model_path, data=test_path, pred=prediction_path)

result = run_vw(train_params=train_params,
                test_params=test_params,
                test_data=test,
                prediction_path=prediction_path)
saved_result = result

comparison = comparison.append(pd.DataFrame(result, index=['Linear Regression w/ Interaction']))
comparison


# ## 2.3 Multinomial Logistic Regression
# 
# An alternative to linear regression is to leverage multinomial logistic regression, or multiclass classification, which treats each rating value as a distinct class. 
# 
# This avoids any non integer results, but also reduces the training data for each class which could lead to poorer performance if the counts of different rating levels are skewed.
# 
# Basic multiclass logistic regression can be accomplished using the One Against All approach specified by the '--oaa N' option, where N is the number of classes and proving the logistic option for the loss function to be used.

"""
Quick description of command line parameters used
  --loss_function logistic: sets the model loss function for logistic regression
  --oaa <N>: trains N separate models using One-Against-All approach (all models are captured in the single model file)
             This expects the labels to be contiguous integers starting at 1
  --link logistic: converts the predicted output from logit to probability
The predicted output is the model (label) with the largest likelihood
"""
train_params = 'vw --loss_function logistic --oaa 5 -f {model} -d {data} --quiet'.format(model=model_path, data=train_path)
test_params = 'vw --link logistic -i {model} -d {data} -t -p {pred} --quiet'.format(model=model_path, data=test_path, pred=prediction_path)

result = run_vw(train_params=train_params,
                test_params=test_params,
                test_data=test,
                prediction_path=prediction_path)

comparison = comparison.append(pd.DataFrame(result, index=['Multinomial Regression']))
comparison


# ## 2.4 Logistic Regression
# 
# Additionally, one might simply be interested in whether the user likes or dislikes an item and we can adjust the input data to represent a binary outcome, where ratings in (1,3] are dislikes (negative results) and (3,5] are likes (positive results).
# 
# This framing allows for a simple logistic regression model to be applied. To perform logistic regression the loss_function parameter is changed to 'logistic' and the target label is switched to [0, 1]. Also, be sure to set '--link logistic' during prediction to convert the logit output back to a probability value.


train_params = 'vw --loss_function logistic -f {model} -d {data} --quiet'.format(model=model_path, data=train_logistic_path)
test_params = 'vw --link logistic -i {model} -d {data} -t -p {pred} --quiet'.format(model=model_path, data=test_logistic_path, pred=prediction_path)

result = run_vw(train_params=train_params,
                test_params=test_params,
                test_data=test,
                prediction_path=prediction_path,
                logistic=True)

comparison = comparison.append(pd.DataFrame(result, index=['Logistic Regression']))
comparison


# # 3. Matrix Factorization Based Recommendations
# 
# All of the above approaches train a regression model, but VW also supports matrix factorization with two different approaches.
# 
# As opposed to learning direct weights for specific users, items and interactions when training a regression model, matrix factorization attempts to learn latent factors that determine how a user rates an item. An example of how this might work is if you could represent user preference and item categorization by genre. Given a smaller set of genres we can associate how much each item belongs to each genre class, and we can set weights for a user's preference for each genre. Both sets of weights could be represented as a vectors where the inner product would be the user-item rating. Matrix factorization approaches learn low rank matrices for latent features of users and items such that those matrices can be combined to approximate the original user item matrix.
# 
# ## 3.1. Singular Value Decomposition Based Matrix Factorization
# 
# The first approach performs matrix factorization based on Singular Value Decomposition (SVD) to learn a low rank approximation for the user-item rating matix. It is is called using the '--rank' command line argument.
# 
# See the [Matrix Factorization Example](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Matrix-factorization-example) for more detail.

"""
Quick description of command line parameters used
  --rank <N>: sets the number of latent factors in the reduced matrix
"""
train_params = 'vw --rank 5 -q ui -f {model} -d {data} --quiet'.format(model=model_path, data=train_path)
test_params = 'vw -i {model} -d {data} -t -p {pred} --quiet'.format(model=model_path, data=test_path, pred=prediction_path)

result = run_vw(train_params=train_params,
                test_params=test_params,
                test_data=test,
                prediction_path=prediction_path)

comparison = comparison.append(pd.DataFrame(result, index=['Matrix Factorization (Rank)']))
comparison


# ## 3.2. Factorization Machine Based Matrix Factorization
# 
# An alternative approach based on [Rendel's factorization machines](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/Rendle2010FM.pdf) is called using '--lrq' (low rank quadratic). More LRQ details in this [demo](https://github.com/VowpalWabbit/vowpal_wabbit/tree/master/demo/movielens).
# 
# This learns two lower rank matrices which are multiplied to generate an approximation of the user-item rating matrix. Compressing the matrix in this way leads to learning generalizable factors which avoids some of the limitations of using regression models with extremely sparse interaction features. This can lead to better convergence and smaller on-disk models.
# 
# An additional term to improve performance is --lrqdropout which will dropout columns during training. This however tends to increase the optimal rank size. Other parameters such as L2 regularization can help avoid overfitting.
"""
Quick description of command line parameters used
  --lrq <abN>: learns approximations of rank N for the quadratic interaction between namespaces starting with 'a' and 'b'
  --lrqdroupout: performs dropout during training to improve generalization
"""
train_params = 'vw --lrq ui7 -f {model} -d {data} --quiet'.format(model=model_path, data=train_path)
test_params = 'vw -i {model} -d {data} -t -p {pred} --quiet'.format(model=model_path, data=test_path, pred=prediction_path)

result = run_vw(train_params=train_params,
                test_params=test_params,
                test_data=test,
                prediction_path=prediction_path)

comparison = comparison.append(pd.DataFrame(result, index=['Matrix Factorization (LRQ)']))
comparison


# # 4. Conclusion
# 
# The table above shows a few of the approaches in the VW library that can be used for recommendation prediction. The relative performance can change when applied to different datasets and properly tuned, but it is useful to note the rapid speed at which all approaches are able to train (75,000 examples) and test (25,000 examples).

# # 5. Scoring

# After training a model with any of the above approaches, the model can be used to score potential user-pairs in offline batch mode, or in a real-time scoring mode. The example below shows how to leverage the utilities in the reco_utils directory to  generate Top-K recommendations from offline scored output.

# First construct a test set of all items (except those seen during training) for each user
users = df[['userID']].drop_duplicates()
users['key'] = 1

items = df[['itemID']].drop_duplicates()
items['key'] = 1

all_pairs = pd.merge(users, items, on='key').drop(columns=['key'])

# now combine with training data and keep only entries that were note in training
merged = pd.merge(train[['userID', 'itemID', 'purchase_count']], all_pairs, on=["userID", "itemID"], how="outer")
all_user_items = merged[merged['purchase_count'].isnull()].fillna(0).astype('int64')

# save in vw format (this can take a while)
to_vw(df=all_user_items, output=all_test_path)


# run the saved model (linear regression with interactions) on the new dataset
test_start = process_time()
test_params = 'vw -i {model} -d {data} -t -p {pred} --quiet'.format(model=saved_model_path, data=all_test_path, pred=prediction_path)
run(test_params.split(' '), check=True)
test_stop = process_time()
test_time = test_stop - test_start

# load predictions and get top-k from previous saved results
pred_data = pd.read_csv(prediction_path, delim_whitespace=True, names=['prediction'], index_col=1).join(all_user_items)
top_k = get_top_k_items(pred_data, col_rating='prediction', k=TOP_K)[['prediction', 'userID', 'itemID']]
top_k.head()



# get ranking metrics
args = [test, top_k]
kwargs = dict(col_user='userID', col_item='itemID', col_rating='purchase_count', col_prediction='prediction',
              relevancy_method='top_k', k=TOP_K)

rank_metrics = {'MAP': map_at_k(*args, **kwargs), 
                'NDCG': ndcg_at_k(*args, **kwargs),
                'Precision': precision_at_k(*args, **kwargs),
                'Recall': recall_at_k(*args, **kwargs)}


# final results
all_results = ['{k}: {v}'.format(k=k, v=v) for k, v in saved_result.items()]
all_results += ['{k}: {v}'.format(k=k, v=v) for k, v in rank_metrics.items()]
print('\n'.join(all_results))


# # 6. Cleanup

tmpdir.cleanup()


