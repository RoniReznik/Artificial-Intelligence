#!/usr/bin/env python
# coding: utf-8

#### 1. Setup

import numpy as np
import pandas as pd
from scipy import special
from sklearn.utils import (as_float_array, check_array, check_X_y, safe_sqr,safe_mask)
import sklearn.datasets as dataset
get_ipython().run_line_magic('matplotlib', 'inline')


#### 2. Load data

def load_data():
    # load iris data
    iris = dataset.load_iris()
    # Save attribute in dataframe
    X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    # Save target in dataframe
    y = pd.DataFrame(iris.target)
    # Concat X and y
    df = pd.concat([X, y], axis=1)
    # Rename y to target
    df = df.rename(columns={0:'target'})    
    return df


#### 3. Boxplot

def show_boxplot(df):
    for i in range(4):
        attribute = df.columns.to_list()[i]
        df.boxplot(attribute, by='target', figsize=(12,8))

#### 4. Anova

def Anova(df):
    # Create mask of the dataframe
    unique_class = pd.unique(df.target.values)
    args = [df[df.columns[:-1].tolist()][df['target'] == arg] for arg in unique_class]
    # Number of classes
    n_classes = len(args)
    # Array of array
    args = [as_float_array(a) for a in args]
    # Number Samplres per class
    number_samples_pre_class = np.array([a.shape[0] for a in args])
    # Overall samples in the data
    number_samples = np.sum(number_samples_pre_class)
    # Square each sample and sum all of them
    ss_all_data = sum(safe_sqr(a).sum(axis=0) for a in args)
    # Sum samples of each class 
    sums_args = [np.asarray(a.sum(axis=0)) for a in args]
    # Sum the sum_args and square it
    square_of_sums_all_data = sum(sums_args) ** 2
    # Square each of the sum_args
    square_of_sums_args = [s ** 2 for s in sums_args]
    #  Sum of Squares Total
    ss_tot = ss_all_data - square_of_sums_all_data / float(number_samples)
    #Sum of Squares Between
    ss_between = 0.
    for k, _ in enumerate(args):
        ss_between += square_of_sums_args[k] / n_samples_pre_class[k]
    ss_between -= square_of_sums_all_data / float(number_samples)             
    # Sum of Squares Within
    ss_within = ss_tot - ss_between
    #Degree of Freedom between
    df_between = n_classes - 1
    # Degree of Freedom within
    df_within = number_samples - n_classes
    #  Mean Square Between
    ms_between = ss_between / float(df_between)
    # Mean Square Within
    ms_within = ss_within / float(df_within)
    # F-value 
    f = ms_between / ms_within
    # flatten matrix to vector in sparse case
    f = np.asarray(f).ravel()
    #  p-value
    prob = special.fdtrc(df_between, df_within, f)
    return prob

#### 5. Run

Anova(df)

