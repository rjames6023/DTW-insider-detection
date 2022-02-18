# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 13:37:01 2021

@author: Robert James
"""
import numpy as np 
import pandas as pd
import multiprocessing
import pickle
import json
import os

import traders
from classification import NN_DTW_classifier, OCSVM_classifier, EnsembleGaussianMixtureModel, IsolationForest_classifier  #explicitly import the classifiers since it gets pickled and unpickled
 
def run_job_high_volume_insider(high_volume_insider, NN_DTW_classifier_model, OCSVM_classifier_model, EGMM_classifier_model, iForest_classifier_model, seed):
    """
    Function to allow parallel processing of a high volume insider trader sequence 
    classification simulation. 

    Parameters
    ----------
    high_volume_insider : Class
        Class describing the high volume insider trader to simulate sequences from.
    NN_DTW_classifier_model : Class
        Trained 1NN DTW classifier model.
    OCSVM_classifier_model : Class
        Trained OCSVM model
    EGMM_classifier_model : Class
        Trained EGMM model
    iForest_classifier_model : Class
        Trained iForest model
    seed : int
        Seed for the random number generator.

    Returns
    -------
    class_label : Dict
        Dictionary of Class labels for each model 
        -1 for anomolous and +1 for normal.

    """
    
    classification_results = {}
    #Simulate a modified trading sequence
    modified_subsequence, replication_X_train_sequence = high_volume_insider.simulate(seed = seed)    
        
    #Predict class label 
        #DTW
    DTW_prediction = NN_DTW_classifier_model.predict(X_test = modified_subsequence, 
                                                     X_train = replication_X_train_sequence)
    DTW_class_labels = [DTW_prediction['class_label'].values[0] for x in range(len(modified_subsequence))]
    classification_results['DTW'] = DTW_class_labels
        #OCSVM
    OCSVM_prediction = OCSVM_classifier_model.predict(X_test = modified_subsequence, 
                                                      X_train = None)
    OCSCM_class_labels = OCSVM_prediction['class_label']
    classification_results['OCSVM'] = OCSCM_class_labels.values

        #EGMM
    EGMM_prediction = EGMM_classifier_model.predict(X_test = modified_subsequence, 
                                                    X_train = None)
    EGMM_class_labels = EGMM_prediction['class_label']
    classification_results['EGMM'] = EGMM_class_labels.values

        #iForest
    iForest_prediction = iForest_classifier_model.predict(X_test = modified_subsequence, 
                                                    X_train = None)
    iForest_class_labels = iForest_prediction['class_label']
    classification_results['iForest'] = iForest_class_labels.values
    
    return classification_results

def run_job_normal_trader(normal_trader, NN_DTW_classifier_model, OCSVM_classifier_model, EGMM_classifier_model, iForest_classifier_model, features, seed):
    """
    Function to allow parallel processing of a normal trader sequence classification simulation. 

    Parameters
    ----------
    normal_trader : Class
        Class describing thenormal to simulate sequences from.
    NN_DTW_classifier_model : Class
        Trained 1NN DTW classifier model.
    OCSVM_classifier_model : Class
        Trained OCSVM model
    EGMM_classifier_model : Class
        Trained EGMM model
    iForest_classifier_model : Class
        Trained iForest model
    features : Numpy array
        Array of feature column naames
    seed : int
        Seed for the random number generator.

    Returns
    -------
    class_label : Dict
        Dictionary of Class labels for each model 
        -1 for anomolous and +1 for normal.

    """
    
    np.random.seed(seed)
    n_features_to_modify = np.random.choice([x for x in range(1,len(features))]) #randomly select a number of features to modify
    
    #modify the None type feature set with the randomly selected feature set 
    normal_trader.features = features[np.random.choice([x for x in range(len(features))], n_features_to_modify, replace = False)]
    
    classification_results = {}
    #Simulate a modified trading sequence
    modified_subsequence, replication_X_train_sequence = normal_trader.simulate(seed = seed)
    
    #Predict class label 
        #DTW
    DTW_prediction = NN_DTW_classifier_model.predict(X_test = modified_subsequence, 
                                                     X_train = None)
    DTW_class_labels = [DTW_prediction['class_label'].values[0] for x in range(len(modified_subsequence))]
    classification_results['DTW'] = DTW_class_labels
        #OCSVM
    OCSVM_prediction = OCSVM_classifier_model.predict(X_test = modified_subsequence, 
                                                      X_train = None)
    OCSCM_class_labels = OCSVM_prediction['class_label']
    classification_results['OCSVM'] = OCSCM_class_labels.values

        #EGMM
    EGMM_prediction = EGMM_classifier_model.predict(X_test = modified_subsequence, 
                                                    X_train = None)
    EGMM_class_labels = EGMM_prediction['class_label']
    classification_results['EGMM'] = EGMM_class_labels.values
    
        #iForest
    iForest_prediction = iForest_classifier_model.predict(X_test = modified_subsequence, 
                                                    X_train = None)
    iForest_class_labels = iForest_prediction['class_label']
    classification_results['iForest'] = iForest_class_labels.values

    return classification_results

def main(project_filepath):
    #Config
    with open(r'{}/config.json'.format(project_filepath), "r") as json_file: #use the working directory filepath
        config = json.loads(json_file.read())
    features = np.array(['standardised_residual_volume',
                'standardised_trade_frequency', 
                'standardised_volatility',
                'standardised_rolling_volume',
                'standardised_price_impact',
                'standardised_adv_spread_component',
                'standardised_participation_volume_rate',
                'standardised_short_sell_volume',
                'standardised_volume'])
    
    #Setup multiprocessing pool.
    num_processes = multiprocessing.cpu_count()
    Pool = multiprocessing.Pool(num_processes-2)
    
    #Import data
    ADQuantiles = pd.read_csv(r'{}/Data/ADQuantiles.csv'.format(project_filepath), dtype = np.float64)
    #Sequence data for the DTW classifier
    raw_reference_sequences = pd.read_csv(r'{}/Data/reference_subsequence_df_standardized_interpolated.csv'.format(project_filepath))
    
    raw_reference_sequences[features] = raw_reference_sequences[features].astype(float)
    reference_sequences = raw_reference_sequences[['subsequence_id', 'date_id', 'account_id'] + features.tolist()]

    ### 1NN-DTW classifier model ###
    try:
        with open('{}/trained_DTW_classifier.pickle'.format(project_filepath), 'rb') as handle:
            NN_DTW_classifier_model = pickle.load(handle)
    except FileNotFoundError:
        NN_DTW_classifier_model = NN_DTW_classifier(sequence_length = config['sequence_length'], 
                                                                   window_width = config['window_width'], 
                                                                   features = features, 
                                                                   threshold_alpha = config['alpha'], 
                                                                   n_jobs = 1)
        #Reference sequence anomaly scores
        NN_DTW_classifier_model.fit(X_train = reference_sequences)
        #Compute anomaly score threshold
        NN_DTW_classifier_model.fit_threshold(ADQuantiles = ADQuantiles)
        #Save the trained model
        with open('{}/trained_DTW_classifier.pickle'.format(project_filepath), 'wb') as handle:
            pickle.dump(NN_DTW_classifier_model, handle, protocol=pickle.HIGHEST_PROTOCOL)   
            
    ### OCSVM model ###
    try:
        with open('{}/trained_OCSVM_classifier.pickle'.format(project_filepath), 'rb') as handle:
            OCSVM_classifier_model = pickle.load(handle)
    except FileNotFoundError:
        OCSVM_classifier_model = OCSVM_classifier(features = features.tolist())
        OCSVM_classifier_model.fit(X_train = reference_sequences)
        #Save the trained model
        with open('{}/trained_OCSVM_classifier.pickle'.format(project_filepath), 'wb') as handle:
            pickle.dump(OCSVM_classifier_model, handle, protocol=pickle.HIGHEST_PROTOCOL)   
                                   
   ### EGMM model ###
    try:
        with open('{}/trained_EGMM_classifier.pickle'.format(project_filepath), 'rb') as handle:
            EGMM_classifier_model = pickle.load(handle)
    except FileNotFoundError:
        EGMM_classifier_model = EnsembleGaussianMixtureModel(features = features.tolist(), 
                                                             mixture_component_upper_limit = 50, 
                                                             threshold_alpha = 0.95)
        EGMM_classifier_model.fit(X_train = reference_sequences)
        #Save the trained model
        with open('{}/trained_EGMM_classifier.pickle'.format(project_filepath), 'wb') as handle:
            pickle.dump(EGMM_classifier_model, handle, protocol=pickle.HIGHEST_PROTOCOL)   
            
    ### Isolation Forest model ###
    try:
        with open('{}/trained_iForest_classifier.pickle'.format(project_filepath), 'rb') as handle:
            iForest_classifier_model = pickle.load(handle)
    except FileNotFoundError:
        iForest_classifier_model = IsolationForest_classifier(features = features.tolist())
        iForest_classifier_model.fit(X_train = reference_sequences)
        #Save the trained model
        with open('{}/trained_iForest_classifier.pickle'.format(project_filepath), 'wb') as handle:
            pickle.dump(iForest_classifier_model, handle, protocol=pickle.HIGHEST_PROTOCOL)   
    
    if not os.path.exists(r'{}/Results'.format(project_filepath)):
        os.makedirs(r'{}/Results'.format(project_filepath))

    # =============================================================================
    # Insider Trading Simulations   
    # =============================================================================
    all_insider_simulation_results = {warping_percentage:{model:pd.DataFrame(columns = ['TN', 'FP'], 
                                                                             index = [x for x in config['insider_percentage_obs_to_change_trials']]) 
                                                          for model in ['DTW', 'OCSVM', 'EGMM', 'iForest']}
                                      for warping_percentage in config['DTW_warping_percentage']}

    for warping_percentage in config['DTW_warping_percentage']:
        for percentage_obs_to_change in config['insider_percentage_obs_to_change_trials']:
            ### Simulation Case 1 - High volume insider ###
            high_volume_insider = traders.Trader(sequence_length = config['sequence_length'], 
                                                                    reference_sequence_data = reference_sequences,
                                                                    warping_percentage = warping_percentage,
                                                                    features = ['standardised_volume', 
                                                                                'standardised_participation_volume_rate',
                                                                                'standardised_residual_volume', 
                                                                                'standardised_rolling_volume'], 
                                                                    percentage_obs_to_change = percentage_obs_to_change,
                                                                    lower_quantile = 0.99, 
                                                                    upper_quantile = 1)
            high_volume_insider_classification_results = Pool.starmap(run_job_high_volume_insider, 
                                                                      [[high_volume_insider, 
                                                                        NN_DTW_classifier_model, 
                                                                        OCSVM_classifier_model, 
                                                                        EGMM_classifier_model, 
                                                                        iForest_classifier_model, 
                                                                        int(round(s*warping_percentage*percentage_obs_to_change*10))] for s in range(config['S'])])
                
            for model in ['DTW', 'OCSVM', 'EGMM', 'iForest']:
                model_class_labels = []
                for result in high_volume_insider_classification_results:
                    model_class_labels.append(result[model])
                model_class_labels = np.concatenate(model_class_labels)
                high_volume_insider_TN = np.sum(np.where(model_class_labels == -1, 1, 0))
                high_volume_insider_FP = np.sum(np.where(model_class_labels == 1, 1, 0))
                all_insider_simulation_results[warping_percentage][model].loc[percentage_obs_to_change, 'TN'] = high_volume_insider_TN
                all_insider_simulation_results[warping_percentage][model].loc[percentage_obs_to_change, 'FP'] = high_volume_insider_FP

        ### export results to csv files ###
        for model in ['DTW', 'OCSVM', 'EGMM', 'iForest']:
            all_insider_simulation_results[warping_percentage][model].to_csv(r'{}/Results/insider_trader_simulation_results_warping_percentage={}_model={}.csv'.format(project_filepath, 
                                                                                                                                                                       warping_percentage, 
                                                                                                                                                                       model))
    
    # =============================================================================
    # Normal Trading Simulations   
    # ============================================================================= 
    all_normal_simulation_results = {warping_percentage:{model:pd.DataFrame(columns = ['TP', 'FN'], 
                                                                            index = [x for x in config['insider_percentage_obs_to_change_trials']])
                                                         for model in ['DTW', 'OCSVM', 'EGMM', 'iForest']}
                                     for warping_percentage in config['DTW_warping_percentage']}

    for warping_percentage in config['DTW_warping_percentage']:
        for percentage_obs_to_change in config['insider_percentage_obs_to_change_trials']:
            ### Simulation Case 2 - Noisy normal trader ###
            normal_trader = traders.Trader(sequence_length = config['sequence_length'], 
                                                       reference_sequence_data = reference_sequences,
                                                       warping_percentage = warping_percentage,
                                                       features = None, 
                                                       percentage_obs_to_change = percentage_obs_to_change,
                                                       lower_quantile = 0, 
                                                       upper_quantile = 1)
            
            noise_normal_trader_classification_results = Pool.starmap(run_job_normal_trader, [[normal_trader, 
                                                                                               NN_DTW_classifier_model, 
                                                                                               OCSVM_classifier_model, 
                                                                                               EGMM_classifier_model, 
                                                                                               iForest_classifier_model, 
                                                                                               features, 
                                                                                               int(round(s*warping_percentage*percentage_obs_to_change*10))] for s in range(config['S'])])
            
            for model in ['DTW', 'OCSVM', 'EGMM', 'iForest']:
                model_class_labels = []
                for result in noise_normal_trader_classification_results:
                    model_class_labels.append(result[model])
                model_class_labels = np.concatenate(model_class_labels)
                noise_normal_trader_FN = np.sum(np.where(model_class_labels == -1, 1, 0))
                noise_normal_trader_TP = np.sum(np.where(model_class_labels == 1, 1, 0))
                all_normal_simulation_results[warping_percentage][model].loc[percentage_obs_to_change, 'FN'] = noise_normal_trader_FN
                all_normal_simulation_results[warping_percentage][model].loc[percentage_obs_to_change, 'TP'] = noise_normal_trader_TP
        
        ### export results to csv files ###
        for model in ['DTW', 'OCSVM', 'EGMM', 'iForest']:
            all_normal_simulation_results[warping_percentage][model].to_csv(r'{}/Results/normal_trader_simulation_results_warping_percentage={}_model={}.csv'.format(project_filepath, 
                                                                                                                                                                     warping_percentage, 
                                                                                                                                                                     model))

if __name__ == '__main__':  
    project_filepath = os.path.dirname(__file__)
    main(project_filepath = project_filepath)