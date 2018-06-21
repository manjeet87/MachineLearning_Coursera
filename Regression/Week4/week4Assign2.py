import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
desired_width = 360
pd.set_option('display.width', desired_width)

from IPython.display import display, HTML
import datetime
plt.rcParams['figure.figsize'] = [16,4]

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int,
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str,
              'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

trainDt = pd.read_csv('wk3_kc_house_train_data.csv')
validDt = pd.read_csv('wk3_kc_house_valid_data.csv')
testDt = pd.read_csv('wk3_kc_house_test_data.csv')

def get_numpy_data(data, features, output):
    featureData = pd.DataFrame()
    featureData[features] = data[features]
    featureData['constant'] = 1
    outData = data[output]
    #print(featureData.info())
    return featureData, outData

def predict_output(feature_matrix, weights):
    predictions = np.matmul(feature_matrix, weights)
    return predictions


def feature_derivative_ridge(errors, features, weight, l2_penalty, feature_is_constant):
    errors = np.array(errors)
    weight = np.array([weight])
    features = np.array(features)
    # If feature_is_constant is True, derivative is twice the dot product of errors and feature
    print("*** ", sum(errors), (weight * weight))
    # cost = sum(errors) + l2_penalty * (sum(weight*weight))
    print((weight))
    i = 0
    for feature in features:

        if i == 0:
            print("feature: ", errors, feature)
            if (len(features.shape) == 1):
                derivative = 2 * sum(features * errors)
            else:
                derivative = 2 * sum(np.matmul(feature, errors))

            weight[i] += derivative
        else:
            if (len(np.array([feature])) == 1):
                derivative = (1 - 2 * l2_penalty) * feature + (2 * sum(feature * errors))
            else:
                derivative = (1 - 2 * l2_penalty) * feature + 2 * sum(np.matmul(feature, errors))

            weight[i] += derivative
        i += 1

    # Otherwise, derivative is twice the dot product plus 2*l2_penalty*weight
    return derivative


(example_features, example_output) = get_numpy_data(trainDt, ['sqft_living'], 'price')
my_weights = np.array([10., 1.])
#print (example_features)
print(np.matmul(example_features, my_weights))
test_predictions = predict_output(example_features, my_weights)
errors = test_predictions - example_output # prediction errors

# next two lines should print the same values
print (feature_derivative_ridge(errors, example_features.iloc[:,1], my_weights[1], 1, False))
print (np.sum(errors*example_features[:,1])*2+20.)
print ('')

# next two lines should print the same values
print (feature_derivative_ridge(errors, example_features[:,0], my_weights[0], 1, True))
print (np.sum(errors)*2.)