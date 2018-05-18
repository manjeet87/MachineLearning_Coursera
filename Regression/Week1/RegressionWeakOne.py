import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

houseData = pd.read_csv('kc_house_data.csv')
x_train, x_test, y_train, y_test = train_test_split(houseData, test_size= 0.2, random_state= 25)

