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



# #  Matrix Factorization Based Recommendations
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

matrix_comparison = pd.DataFrame(result, index=['Matrix Factorization'])



# ## Factorization Machine Based Matrix Factorization
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

matrix_comparison = matrix_comparison.append(pd.DataFrame(result, index=['Matrix Factorization (LRQ)']))


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
