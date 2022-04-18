import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
import seaborn as sns

from lifelines import KaplanMeierFitter

class KaplanMeierUtils():
    
    
    """
    Module of reusable function and utilities for
    survival analysis.
    
    """
    
    def __init__(self):
        pass
    
    def PlotKaplanMeierEstimatesForCategoricalVariables(self, data = None, categorical_columns=[]):
    
        '''
        Purpose: 
            Plots the Kaplan Meier Estimates For Categorical Variables in the data.

            **NOTE: The Kaplan-Meier estimator is used to estimate the survival function. 
            The visual representation of this function is usually called the Kaplan-Meier 
            curve, and it shows what the probability of an event (for example, survival) 
            is at a certain time interval. If the sample size is large enough, the curve 
            should approach the true survival function for the population under 
            investigation.

        Parameters:
            1. data: the dataset.
            2. categorical_columns: all the categorical data features as a list.

        Return Value: 
            NONE.

        Reference:       https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html#lifelines.fitters.kaplan_meier_fitter.KaplanMeierFitter
        '''

        categoricalData = data.loc(axis=1)[categorical_columns]

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18,10))
        plt.tight_layout(pad=5.0)

        def km_fits(data, curr_Feature = None):

            '''
            Purpose: 
                Generates the Kaplan Meier fits for fitting the Kaplan-Meier 
                estimate for the survival function.

            Parameters:
                1. data: the dataset.
                2. curr_Feature: the current feature under consideration.

            Return Value: 
                kmfits: the Kaplan-Meier estimates.
            '''

            curr_feature_range = np.unique(data[curr_Feature])

            X = [data[data[curr_Feature]==x]['time'] for x in curr_feature_range]
            Y = [data[data[curr_Feature]==y]['DEATH_EVENT'] for y in curr_feature_range]
            fit_label = [str(curr_Feature + ': ' + str(feature_range_i)) for feature_range_i in curr_feature_range]

            kmfits = [KaplanMeierFitter().fit(durations = x_i, 
                                              event_observed = y_i, 
                                              label = fit_label[i]) for i,(x_i, y_i) in enumerate(zip(X,Y))]

            return kmfits

        for idx, feature in enumerate(categorical_columns):
            cat_fits = km_fits(data = data, 
                               curr_Feature = feature)

            [x.plot(title=feature, ylabel="Survival Probability", xlabel="Days",
                    ylim=(0,1.1), xlim=(0,290),
                    ci_alpha=0.1, ax=ax.flatten()[idx]) for x in cat_fits]

        ax.flatten()[-1].set_visible(False)
        fig.suptitle("Kaplan Meier Estimates for Categorical Variables ", fontsize=16.0)
        plt.show()
        
        return None
    
    def PlotKaplanMeierEstimatesForContinuousVariables(self, data = None, continuous_columns=[]):
    
        '''
        Purpose: 
            Plots the Kaplan Meier Estimates For Continuous Variables in the data.

            **NOTE: The Kaplan-Meier estimator is used to estimate the survival function. 
            The visual representation of this function is usually called the Kaplan-Meier 
            curve, and it shows what the probability of an event (for example, survival) 
            is at a certain time interval. If the sample size is large enough, the curve 
            should approach the true survival function for the population under 
            investigation.

        Parameters:
            1. data: the dataset.
            2. continuous_columns: all the continuous data features as a list.

        Return Value: 
            NONE.

        Reference:
            https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html#lifelines.fitters.kaplan_meier_fitter.KaplanMeierFitter
        '''

        continuousData = data.loc(axis=1)[continuous_columns]

        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(18,15))
        plt.tight_layout(pad=5.0)

        def km_fits(data, curr_Feature = None, split_points = None):

            '''
            Purpose: 
                Generates the Kaplan Meier fits for fitting the Kaplan-Meier 
                estimate for the survival function.

            Parameters:
                1. data: the dataset.
                2. curr_Feature: the current feature under consideration.
                3. split_points: the data split points to cut the data.

            Return Value: 
                kmfits: the Kaplan-Meier estimates.
            '''

            bins = pd.cut(x=data[curr_Feature],bins=split_points)
            curr_feature_range = np.unique(bins)
            curr_feature_group = str(curr_Feature) + "_group"
            data[curr_feature_group] = pd.cut(x=data[curr_Feature], bins=split_points)

            X = [data[data[curr_feature_group] == bin_range]['time'] for bin_range in curr_feature_range]      
            Y = [data[data[curr_feature_group] == bin_range]['DEATH_EVENT'] for bin_range in curr_feature_range]        
            fit_label = [str(str(feature_range_i).replace(',',' -').replace(']',')')) for feature_range_i in curr_feature_range]        
            data.drop(curr_feature_group, axis=1, inplace=True)

            kmfits = [KaplanMeierFitter().fit(durations = x_i, 
                                              event_observed = y_i, 
                                              label=fit_label[i]) for i,(x_i, y_i) in enumerate(zip(X,Y))]

            return kmfits

        data_split_points = [[39.0,60.0,80.0,100.0],3,[0,30.0,45.0,100.0],3,3,3,3]

        for idx, feature in enumerate(continuous_columns):

            cont_fits = km_fits(data = data, 
                                curr_Feature = feature,
                                split_points = data_split_points[idx])

            [x.plot(title=feature, ylabel="Survival Probability", xlabel="Days",
                    ylim=(0,1.1), xlim=(0,290), ci_alpha=0.1, 
                    ax=ax.flatten()[idx]) for x in cont_fits]

        ax.flatten()[-1].set_visible(False)
        ax.flatten()[-2].set_visible(False)

        fig.suptitle("Kaplan Meier Estimates for Continuous Variables ", fontsize=16.0, y=1.0)

        plt.show()