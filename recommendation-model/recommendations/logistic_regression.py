

import os 
from tempfile import TemporaryDirectory
import pandas as pd 
from time import process_time

from ..utils.dataset import load_pandas_df
from ..utils.python_splitters import python_random_split


from ..model.vowpal_wabbit import to_vw
from ..model.vowpal_wabbit import run_vw

# create temp directory to maintain data files
tmpdir = TemporaryDirectory()

model_path = os.path.join(tmpdir.name, 'vw.model')
saved_model_path = os.path.join(tmpdir.name, 'vw_saved.model')
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

# save data for logistic regression (requires adjusting the label)
to_vw(df=train, output=train_logistic_path, logistic=True)
to_vw(df=test, output=test_logistic_path, logistic=True)


# ##  Multinomial Logistic Regression
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
train_params = 'vw --loss_function logistic --oaa 5 -f {model} -d {data} --quiet'.format(model=model_path, data=train_logistic_path)
test_params = 'vw --link logistic -i {model} -d {data} -t -p {pred} --quiet'.format(model=model_path, data=test_logistic_path, pred=prediction_path)

result = run_vw(train_params=train_params,
                test_params=test_params,
                test_data=test,
                prediction_path=prediction_path)

Logistic_comparison = pd.DataFrame(result, index=['Linear Regression'])



# ## Logistic Regression
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

Logistic_comparison = Logistic_comparison.append(pd.DataFrame(result, index=['Logistic Regression']))



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
