import os 
from tempfile import TemporaryDirectory
import pandas as pd 
from time import process_time

from ..utils.dataset import load_pandas_df
from ..utils.python_splitters import python_random_split
from ..utils.evaluation import (rmse, mae, exp_var, rsquared, get_top_k_items,
                                                     map_at_k, ndcg_at_k, precision_at_k, recall_at_k)

from ..model.vowpal_wabbit import to_vw
from ..model.vowpal_wabbit import run_vw

# create temp directory to maintain data files
tmpdir = TemporaryDirectory()

model_path = os.path.join(tmpdir.name, 'vw.model')
saved_model_path = os.path.join(tmpdir.name, 'vw_saved.model')
train_path = os.path.join(tmpdir.name, 'train.dat')
test_path = os.path.join(tmpdir.name, 'test.dat')   
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


# #  Regression Based Recommendations
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

comparison = pd.DataFrame(result, index=['Linear Regrimport os 
from tempfile import TemporaryDirectory
import pandas as pd 

from ..utils.dataset import load_pandas_df
from ..utils.python_splitters import python_random_split
from ..utils.evaluation import (rmse, mae, exp_var, rsquared, get_top_k_items,
                                                     map_at_k, ndcg_at_k, precision_at_k, recall_at_k)

from ..model.vowpal_wabbit import to_vw
from ..model.vowpal_wabbit import run_vw

# create temp directory to maintain data files
tmpdir = TemporaryDirectory()

model_path = os.path.join(tmpdir.name, 'vw.model')
saved_model_path = os.path.join(tmpdir.name, 'vw_saved.model')
train_path = os.path.join(tmpdir.name, 'train.dat')
test_path = os.path.join(tmpdir.name, 'test.dat')   
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
ession'])
comparison



# ## Linear Regression with Interaction Features
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



# # 5. Scoring
# After training a model with any of the above approaches, the model can be used to score potential user-pairs in offline batch mode, 
# or in a real-time scoring mode. The example below shows how to leverage the utilities in the reco_utils directory to  generate 
# Top-K recommendations from offline scored output.
# First construct a test set of all items (except those seen during training) for each user
users = df[['user_id']].drop_duplicates()
users['key'] = 1
items = df[['product_id']].drop_duplicates()
items['key'] = 1

all_pairs = pd.merge(users, items, on='key').drop(columns=['key'])

# now combine with training data and keep only entries that were note in training
merged = pd.merge(train[['user_id', 'product_id', 'purchase_count']], all_pairs, on=["user_id", "product_id"], how="outer")
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
print( top_k.head() )

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

