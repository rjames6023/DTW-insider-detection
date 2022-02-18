# -*- coding: utf-8 -*-
"""
@author: Robert James
"""
import numpy as np 
import pandas as pd

import utils

class Trader():
    """
    """
    
    def __init__(self, sequence_length, reference_sequence_data, warping_percentage, features, percentage_obs_to_change, lower_quantile, upper_quantile):
        """

        Parameters
        ----------
        sequence_length : integer
            Length of subsequences in the "reference_data"
        reference_sequence_data : Pandas DataFrame
            DataFrame of reference subsequences.
        warping_percentage : float
            The ammount of warping (between 0 and 1) to add to the simulated sequence
        features : list
            List of features to modify in the random simulated sequences.
        percentage_obs_to_change : float
            Float between (between 0 and 1) describing the proportion of the random sequence to modify
        lower_quantile : list
            List of lower quantile limits for modifying the random sequence.
            Should have the same number of elements as the variable _features.
        upper_quantile : list
            List of upper quantile limits for modifying the random sequence.
            Should have the same number of elements as the variable _features.
        
        Returns
        -------
        None.

        """
        
        self.reference_sequence_data = reference_sequence_data
        self.sequence_length = sequence_length
        self.warping_percentage = warping_percentage
        self.percentage_obs_to_change = percentage_obs_to_change
        self.features = features
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    
    def simulate(self, seed):
        """
        Simulate a random sequence of trading activity from a high volume insider trader.

        Parameters
        ----------
        seed : int
            Random seed.

        Returns
        -------
        warped_modified_subsequence : Numpy array
            A randomly selected, modified and (possibly) warped time series subsequence 
        replication_X_train_subsequence : Pandas DataFrame
            A copy of the __reference_sequence_data__ with the randomly chosen subsequence removed

        """
        #Extract a random subsequence to modify
        np.random.seed(seed)
        random_sequence_id = np.random.choice(np.unique(self.reference_sequence_data['subsequence_id'].values))
        random_subsequence = self.reference_sequence_data[self.reference_sequence_data['subsequence_id'] == random_sequence_id]
        
        #Remove the modified sequence from the reference database. 
        replication_X_train_subsequence = self.reference_sequence_data[~(self.reference_sequence_data['subsequence_id'] == random_sequence_id)]
    
        #Adjust feature values
        modified_subsequence = random_subsequence.copy(deep = True)
        modified_subsequence.reset_index(drop = True, inplace = True)
        n_to_modify = int(np.ceil(self.percentage_obs_to_change*self.sequence_length))
        idx_to_modify = np.random.choice([x for x in range(self.sequence_length)], 
                                          size = n_to_modify, 
                                          replace = False)
    
        for _feature_ in self.features:
            random_sample_CDF = np.random.uniform(self.lower_quantile, self.upper_quantile, size = n_to_modify)
            simulated_feature = np.percentile(self.reference_sequence_data[_feature_], random_sample_CDF*100)
            modified_subsequence.loc[idx_to_modify, _feature_] = simulated_feature
        
        if self.warping_percentage > 0:
            #Randomly warp the modified subsequence
            warped_modified_subsequence_features = utils.add_random_warping(sequence = modified_subsequence.values[:,3:], 
                                                                   warping_percentage = self.warping_percentage)
            #Add back the subsequence id, date id and account id to the warped feature data
            warped_modified_subsequence = pd.DataFrame(np.concatenate([modified_subsequence.values[:,:3], warped_modified_subsequence_features], axis = 1), 
                                                       columns = modified_subsequence.columns)
            return warped_modified_subsequence, replication_X_train_subsequence
        else:
            return modified_subsequence.values, replication_X_train_subsequence
