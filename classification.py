# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 12:50:16 2021

@author: Robert James
"""
import numpy as np 
import pandas as pd

from scipy.optimize import minimize
from scipy import stats
from sklearn import svm, mixture #OCSVM and GMM benchmark
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from tqdm import tqdm

from numba import jit

def fit_GPD_forwardstop(n, distances, max_data_quantile, ADQuantiles):
    """
    Fit multiple generalized Pareto extreme value distributions using different intial
    threshold values to compute the forwars stop statistic. 
    
    See:
        Bader, B., Yan, J. and Zhang, X., 2018. 
        Automated threshold selection for extreme value analysis via ordered goodness-of-fit 
        tests with adjustment for false discovery rate. 
        The Annals of Applied Statistics, 12(1), pp.310-329.
        
        G'Sell, M.G., Wager, S., Chouldechova, A. and Tibshirani, R., 2016. 
        Sequential selection procedures and false discovery rate control. 
        Journal of the Royal Statistical Society: Series B: Statistical Methodology, pp.423-444.
    
    """
    threshold_information_df = pd.DataFrame(columns = ['threshold_quantile', 'threshold', 'sample_size', 'test_stat', 'p-value', 'shape', 'scale'])
    for index, threshold in enumerate([x for x in range(70, int(max_data_quantile*100))]): #Loop over possible candidate threshold choices
        threshold_percentage = 1 - threshold/100
        candidate_POT_sample = distances.head(int(np.floor(threshold_percentage*n)))
        candidate_threshold = candidate_POT_sample.min()-0.000001
        candidate_excedances = (candidate_POT_sample-candidate_threshold).sort_values()
        n_ = len(candidate_POT_sample)
    
        #Fit generalized Pareto distribution to the candidate excedances                 
        generalizedPareto_dist = generalizedPareto(excess_loss = candidate_excedances.values, initial_threshold = candidate_threshold, n = n_)
        candidate_shape, candidate_scale = generalizedPareto_dist.fit()
        
        #Obtain CDF estimates
        candidate_PITs = stats.genpareto.cdf(x = candidate_excedances, c = candidate_shape, loc = 0, scale = candidate_scale)
        candidate_PITs = np.sort(candidate_PITs)
        AD_sum = 0
        for i, candidate_PIT in enumerate(candidate_PITs):
            AD_sum += ((2*(i+1))-1)*(np.log(candidate_PIT) + np.log1p(-candidate_PITs[n_-i-1])) #Use n-i-1 since python indexing starts at 0
        A2_statistic = -n_-(1/n_)*AD_sum
    
        #Find p-value
        ADQuantiles_row_index = ADQuantiles[ADQuantiles['shape'] == round(candidate_shape,2)].index[0]
        if A2_statistic > ADQuantiles.at[ADQuantiles_row_index,'0.001']: #Check if we need to use log linear interpolation
            table_pvals = np.log([float(x) for x in ADQuantiles.columns.tolist()[950:999+1]])*-1
            x = ADQuantiles.iloc[ADQuantiles_row_index, 950:999+1]
            x = np.column_stack([np.ones(len(x)), x])
            LS_coef = np.dot(np.linalg.pinv(x),table_pvals)
            p_value = np.exp((np.dot([1,A2_statistic], LS_coef)*-1))
        else:
            ADQuantiles_row = ADQuantiles.loc[ADQuantiles_row_index, :].iloc[1:]
            bound = max(ADQuantiles_row[A2_statistic < ADQuantiles_row].index)            
            if bound == '0.999':
                p_value = 0.999
            else:
                lower = ADQuantiles.at[ADQuantiles_row_index, str(round(float(bound)+0.001,3))]
                upper = ADQuantiles.at[ADQuantiles_row_index, bound]
                diff = (upper - A2_statistic)/(upper - lower)
                val = (diff * (-np.log(float(bound))--np.log(float(bound) + 0.001))) + np.log(float(bound))
                p_value = np.exp(val)
    
        threshold_information_df.at[index, 'threshold_quantile'] = threshold_percentage
        threshold_information_df.at[index, 'threshold'] = candidate_threshold
        threshold_information_df.at[index, 'sample_size'] = len(candidate_POT_sample)
        threshold_information_df.at[index, 'test_stat'] = A2_statistic
        threshold_information_df.at[index, 'p-value'] = p_value
        threshold_information_df.at[index, 'shape'] = candidate_shape
        threshold_information_df.at[index, 'scale'] = candidate_scale
    
    #Compute forward stop scores (G'Sell et al. 2016 - Journal of the Royal Statistical Society Series B)
    raw_p_values = threshold_information_df['p-value'].values
    m = len(threshold_information_df)
    int_ = [x for x in range(0,m,1)]
    reversed_p_values = raw_p_values[::-1]
    pFDR = np.cumsum(-np.log(1-reversed_p_values[int_].astype(float)))/([x+1 for x in int_])
    reversed_pFDR = pFDR[::-1]
    threshold_information_df['forward_stop'] = reversed_pFDR
    
    return threshold_information_df
        
class generalizedPareto():    
    """
    Functions for fitting the generalized Pareto extreme value distribution to 
    data and computing tail functionals fopr the underlying data distribution 
    based upon the POT tail estimator
    """
    def __init__(self, excess_loss, initial_threshold, n):
        self.excess_loss = excess_loss
        self.sorted_excess_loss = np.sort(excess_loss)
        self.initial_threshold = initial_threshold
        self.n = n 
        self.n_u = len(excess_loss)
        
    def LMoments(self):
        """
        LMoments parameter estimates, see Hosking, J.R., 1990. 
        L‐moments: Analysis and estimation of distributions using linear combinations of order statistics. 
        Journal of the Royal Statistical Society: Series B (Methodological), 52(1), pp.105-124.

        Returns
        -------
        L1 : TYPE
            DESCRIPTION.
        L2 : TYPE
            DESCRIPTION.

        """
        
        try:
            n = len(self.excess_loss)
            #First L-Moment
            b_0 = np.mean(self.excess_loss)
            L1 = b_0            
            #Second L-Moment
            to_sum = []
            for j in range(1, n):
                to_sum.append(self.excess_loss[j]*((j+1-1)/(n+1-1)))
            beta_1 = np.mean(np.array(to_sum))
            L2 = 2*beta_1 - b_0
        except:
            L1 = np.nan
            L2 = np.nan
        return L1,L2

    @staticmethod
    @jit(nopython = True)   
    def loglikelihood(params, data): 
        sum_component = np.log(np.where(1 + params[0]*data/params[1] > 0, 1 + params[0]*data/params[1], 1e-6))
        ll = -(-len(data)*np.log(params[1]) - (1 + (1/params[0]))*np.sum(sum_component))
        return ll
        
    def fit(self):
        """
        Estimate distribution parameters
        """
        
        #Initial parameters for MLE
        L1,L2 = generalizedPareto.LMoments(self)
        tau = L2/L1
        initial_scale = L1 * (1/tau-1)
        initial_shape = -(1/tau-2)
                
        MLE_estimate_results = minimize(fun = generalizedPareto.loglikelihood, x0 = [initial_shape, initial_scale], 
                                        method = 'nelder-mead', args = (self.excess_loss), options = {'maxiter':10000})                
            
        self.shape = MLE_estimate_results['x'][0]
        self.scale = MLE_estimate_results['x'][1]
        return MLE_estimate_results['x']
        
    def POT_tail_estimator_quantile(self, alpha):
        """
        POT tail estimator quantile 
        """

        q = self.initial_threshold + (self.scale/self.shape)*(((self.n/self.n_u)*(1-alpha))**(-self.shape) - 1)
        self.q = q
        return q
        
    def POT_tail_estimator_CTE(self, alpha):
        """
        POT tail estimator Conditional_tail expectation based upon generalized Pareto estimated parameters
        """
        
        CTE = (self.q + self.scale - (self.shape * self.initial_threshold))/(1 - self.shape)
        self.CTE = CTE
        return CTE
        

class NN_DTW_classifier:
    """
    NN-DTW time series classification algorithm
    
    """
    
    def __init__(self, sequence_length, window_width, features, threshold_alpha, n_jobs = 1):
        self.sequence_length = sequence_length
        self.window_width = window_width
        self.features = features
        self.n_jobs = n_jobs 
        
        #Parameters for threshold estimation
        self.threshold_alpha = threshold_alpha
        self.max_data_quantile = threshold_alpha - 0.01  #largest percentile for the initial threshold
           

    ### Lower bound functions ###
    @staticmethod
    @jit(nopython = True)   
    def _lb_kim_UCR(s1, s2):
        """
        See Rakthanmanon, T., Campana, B., Mueen, A., Batista, G., Westover, B., Zhu, Q., Zakaria, J. and Keogh, E., 2013. Addressing big data time series: Mining trillions of time series subsequences under dynamic time warping. ACM Transactions on Knowledge Discovery from Data (TKDD), 7(3), pp.1-31.
        """
        start_distance = np.sum((s1[0,:] - s2[0,:])**2)
        end_distance = np.sum((s1[-1,:] - s2[-1,:])**2)
        lb_kim = start_distance + end_distance
        return lb_kim

    @staticmethod
    @jit(nopython = True)    
    def _lb_keogh_multivariate(s1, s2, window, create_band, upper_matrix, lower_matrix):
        """
        see Keogh, E. and Ratanamahatana, C.A., 2005. Exact indexing of dynamic time warping. Knowledge and information systems, 7(3), pp.358-386.
        """
        lb = 0
        lb_list = np.zeros((len(s1)))    
        projection_frame = s2.copy()
        for i in range(len(s1)):
            imin = int(max(0, i - max(0, len(s1) - len(s2)) - window))
            imax = int(min(len(s2), i + max(0, len(s2) - len(s1)) + window))
            for p in range(s1.shape[1]):
                if create_band == True:
                    u_ip = np.max(s1[imin:imax,p])
                    l_ip = np.min(s1[imin:imax, p])
                    upper_matrix[i,p] = u_ip
                    lower_matrix[i,p] = l_ip
                else:
                    u_ip = upper_matrix[i,p]
                    l_ip = lower_matrix[i,p]
                if s2[i,p] > u_ip:
                    lb += (s2[i,p] - u_ip)**2
                    projection_frame[i,p] = u_ip
                elif s2[i,p] < l_ip:
                    lb += (s2[i,p] - l_ip)**2
                    projection_frame[i,p]= l_ip
            lb_list[i] = lb
        return lb, np.sqrt(lb_list), projection_frame, upper_matrix, lower_matrix

    @staticmethod
    @jit(nopython = True)   
    def _lb_improved_multivariate(s1, projection_frame, window):
        """
        See Lemire, D., 2009. Faster retrieval with a two-pass dynamic-time-warping lower bound. Pattern recognition, 42(9), pp.2169-2180.
        """
        lb_prime = 0
        for i in range(len(s1)):
            imin = max(0, i - max(0, len(s1) - len(projection_frame)) - window)
            imax = min(len(projection_frame), i + max(0, len(projection_frame) - len(s1)) + window)
            for p in range(s1.shape[1]):
                u_ip_prime = np.max(projection_frame[imin:imax, p])
                l_ip_prime = np.min(projection_frame[imin:imax, p])
                if s1[i, p] > u_ip_prime:
                    lb_prime += (s1[i, p] - u_ip_prime)**2
                elif s1[i, p] < l_ip_prime:
                    lb_prime += (s1[i, p] - l_ip_prime)**2
        return lb_prime
    
    @staticmethod
    @jit(nopython = True) 
    def DTW_distance_multivariate(s1, s2, window, early_abandon_score, cumulative_lb_list):
        """
        Copmpute the DTW distance between s1 and s2 using a warping width window and early abandoning
        using the cumulative LB_keogh bound and BSF distance
        """
        
        D = np.full((len(s1), len(s2)), np.inf)
        D[0,0] = np.sum((s1[0,:] - s2[0,:])**2)
        D_raw = np.full((len(s1), len(s2)), np.inf)
        D_raw[0,0] = np.sum((s1[0,:] - s2[0,:])**2)
        window = max(window, abs(len(s1) - len(s2)))
        ED_UB = np.sum((s1-s2)**2)
        sc = 0
        ec = 0    
        for i in range(0,len(s1)):
            min_cost = np.inf
            beg = int(max(sc, i-window))
            end = int(min(len(s2), i+window))
            smaller_found = False
            ec_next = i
            for j in range(beg, end):
                cost = np.sum((s1[i,:] - s2[j,:])**2)
                D_raw[i,j] = cost
                
                if i ==0 and j ==0:
                    continue
                if i == 0:
                    D[i,j] = cost + D[i, j-1]
                elif j == 0:
                    D[i,j] = cost + D[i-1,j]
                else: 
                    D[i,j] = cost + min(D[i, j-1], D[i-1,j], D[i-1, j-1])
                
                    #Pruning Strategy of Silva & Bartista(2018)
                    if D[i,j] > ED_UB:
                        if smaller_found == False:
                            sc = j+1
                        if j >= ec:
                            break
                    else:
                        smaller_found = True
                        ec_next = j+1
                    
                #LB_Keogh Early Abandoning
                if D[i,j] < min_cost:
                    min_cost = D[i,j]  
                    
            if np.sqrt(min_cost) >= early_abandon_score and i !=0 and j !=0:
                return np.inf
                break
            if i <= len(s1)-2 and i !=0 and j !=0: #Check that the loop is not at the corner of the matrix
                if (np.sqrt(min_cost) + (cumulative_lb_list[i+1] - cumulative_lb_list[len(s1)-1])) >= early_abandon_score:
                    break
                    return np.inf
                    
            ec = ec_next
        d = np.sqrt(D[i,j])            
        return d  

    def _NN_DTW_search(self, query_subsequence, reference_subsequences):
        """
        Nearest Neighbour search for a given query sequence over a dataframe of reference sequences
        """
        
        #Setup the best-so-far distance, LB_keogh upper and lower band matrices
        BSF_distance = np.inf
        BSF_subsequence_ID = np.inf
        LB_keogh_first_pass_bool = True
        initial_u_ip_matrix = np.zeros(query_subsequence.shape, dtype = np.float64)
        initial_l_ip_matrix = np.zeros(query_subsequence.shape, dtype = np.float64)
        
        #Search over all reference subsequences for the NN under DTW distance
        for reference_subsequence_id in reference_subsequences['subsequence_id'].unique():
            reference_subsequence = reference_subsequences[reference_subsequences['subsequence_id'] == reference_subsequence_id][self.features]

            #Check if LB_kim lower bound is violated
            lb_kim = NN_DTW_classifier._lb_kim_UCR(s1 = query_subsequence.values, s2 = reference_subsequence.values)    
            if np.sqrt(lb_kim) < BSF_distance:
                #Check if LB_keogh lower bound is violated
                if LB_keogh_first_pass_bool == True:
                    lb_keogh, cumulative_lb_keogh, projection, fixed_u_ip_matrix, fixed_l_ip_matrix = NN_DTW_classifier._lb_keogh_multivariate(s1 = query_subsequence.values, s2 = reference_subsequence.values, window = self.window_width, 
                                                                                                                                               create_band = True, upper_matrix = initial_u_ip_matrix, lower_matrix = initial_l_ip_matrix)
                    LB_keogh_first_pass_bool = False
                else:
                    lb_keogh, cumulative_lb_keogh, projection, _, _ = NN_DTW_classifier._lb_keogh_multivariate(s1 = query_subsequence.values, s2 = reference_subsequence.values, window = self.window_width, 
                                                                                                               create_band = False, upper_matrix = fixed_u_ip_matrix, lower_matrix = fixed_l_ip_matrix)
                if np.sqrt(lb_keogh) < BSF_distance:
                    #Check if LB_imporved lower bound is violated      
                    lb_imprv = NN_DTW_classifier._lb_improved_multivariate(s1 = query_subsequence.values, projection_frame = projection, window = self.window_width)
                    if np.sqrt(lb_keogh + lb_imprv) < BSF_distance:
                        #Compute DTW alignment
                        distance_score = NN_DTW_classifier.DTW_distance_multivariate(s1 = query_subsequence.values, s2 = reference_subsequence.values, window = self.window_width, 
                                                                   early_abandon_score = BSF_distance, cumulative_lb_list = cumulative_lb_keogh)
                    else:
                        distance_score = np.inf
                else:
                    distance_score = np.inf
            else:
                distance_score = np.inf
                
            if distance_score < BSF_distance: #Check if the current DTW distance is the smallest so far
                BSF_distance = distance_score
                BSF_subsequence_ID = reference_subsequence_id
        return BSF_distance, BSF_subsequence_ID

    def fit(self, X_train):
        """
        Compute nearest neighbour distances for all subsequences in the X_train - a df of reference subsequences
        
        """
        self.X_train = X_train
        train_NN_df = pd.DataFrame(columns = ['subsequence_id', 'date_id', 'account_id', '1NN_DTW_distance', '1NN_subsequence_id'], index = range(len(X_train['subsequence_id'].unique())))

        if self.n_jobs == 1:
            ### data pre-processing ###
                #remove time series subsequences from the same account id and date id to avoid trade overlap from rolling window construction
            for i, subsequence_id in enumerate(X_train['subsequence_id'].unique()): #Loop over all reference subsequences
                reference_query_subsequence = X_train[X_train['subsequence_id'] == subsequence_id]
                reference_query_subsequence_account_id = reference_query_subsequence['account_id'].unique()[0]
                reference_query_subsequence_date_id = reference_query_subsequence['date_id'].unique()[0]
                
                #Identify sequences by the same account on the same day and remove these to create a query reference sequence specific training dataset
                overlap_sequence_ids = np.unique(X_train[(X_train['date_id'] == reference_query_subsequence_date_id) & (X_train['account_id'] == reference_query_subsequence_account_id)]['subsequence_id'])
                clean_reference_sequences = X_train[~np.isin(X_train['subsequence_id'], overlap_sequence_ids)] #remove any overlap sequences (including the query sequence) from the reference sequence dataset
                
                #NN Search
                query_subsequence = reference_query_subsequence[self.features]                
                BSF_distance, BSF_subsequence_ID = self.CTE_threshold_NN_DTW_search(sequery_subsequence = query_subsequence, reference_subsequences = clean_reference_sequences)

                train_NN_df.at[i, 'subsequence_id'] = subsequence_id
                train_NN_df.at[i, 'date_id'] =  reference_query_subsequence_date_id
                train_NN_df.at[i, 'account_id'] =  reference_query_subsequence_account_id
                train_NN_df.at[i, '1NN_subsequence_id'] = BSF_subsequence_ID
                train_NN_df.at[i, '1NN_DTW_distance'] = BSF_distance
                
        train_NN_df.sort_values(by = ['subsequence_id'], inplace = True)
        #Ensure data types are correct
        train_NN_df[['subsequence_id', 'date_id', 'account_id', '1NN_subsequence_id']] = train_NN_df[['subsequence_id', 'date_id', 'account_id', '1NN_subsequence_id']].astype(int)
        train_NN_df['1NN_DTW_distance'] = train_NN_df['1NN_DTW_distance'].astype(float)
        train_NN_df['class_label'] = 1

        self.train_NN_df = train_NN_df
        return train_NN_df
    
    def fit_threshold(self, ADQuantiles = None):
        """
        Estimate the anomaly score threshold using 1NN-DTW distance scores (X) using sequential Goodness of fit testing based on Anderson Darling Statistic

        """
        
        #Could have clusters of extremes because of the rollowing window sequence extraction
        #Retain only the largest anomaly score for each account_id on each day to satisfy I.I.D assumption of data -
        X = self.train_NN_df.copy(deep = True)
        X_tilde = []
        for name, group in X.groupby(['date_id', 'account_id']):
            max_idx = group['1NN_DTW_distance'].idxmax()
            X_tilde.append(group.loc[[max_idx], :])
        X_tilde = pd.concat(X_tilde, axis = 0)
        X_tilde.reset_index(inplace = True, drop = True)
        distances = X_tilde['1NN_DTW_distance'].copy(deep = True)
        
        distances.sort_values(inplace = True, ascending = False) 
        distances.reset_index(inplace = True, drop = True)
        n = len(distances)
        
        initial_POT_sample = distances.head(int(np.floor(self.max_data_quantile*n)))
        if len(initial_POT_sample) < 30: #find the maximum initial threshold quantile such that at least 30 data points are available for estimation
            q = self.max_data_quantile
            termination_condition = False
            while termination_condition == False:
                new_max_data_quantile = q - 0.01
                initial_POT_sample = distances.head(int(np.floor(new_max_data_quantile*n)))
                print(len(initial_POT_sample))
                if len(initial_POT_sample) >= 30:
                    termination_condition = True
                q = q - 0.01        
        else:
            new_max_data_quantile = self.max_data_quantile

        threshold_information_df = fit_GPD_forwardstop(n, distances, new_max_data_quantile, ADQuantiles)

        #Determine threshold choice
        threshold_information_df['stop_point'] = np.where(threshold_information_df['forward_stop'] <= 0.1, 1, 0)
        if np.all(threshold_information_df['stop_point']== 0):
            k_hat = np.max(threshold_information_df.index.tolist()) #If no rejection is made select the largest threshold
        else:
            k_hat = np.min(threshold_information_df[threshold_information_df['stop_point'] == 1].index.tolist()) #Take the first rejection point
        
        #All final tail sample information as selected by the automated goodness-of-fit testing
        final_threshold = threshold_information_df.loc[k_hat, 'threshold']
        final_POT_sample = distances[distances > final_threshold]
        final_n_u = len(final_POT_sample)
        final_shape = threshold_information_df.loc[k_hat, 'shape']
        final_scale = threshold_information_df.loc[k_hat, 'scale']
        
        self.GPD_initial_threshold = final_threshold
        self.GPD_shape = final_shape
        self.GPD_scale = final_scale
        
        #Compute the final anomaly score threshold using the POT tail estimator function analytical quantile and CTE expressions
        tail_quantile = final_threshold + (final_scale/final_shape)*(((n/final_n_u)*(1 - self.threshold_alpha))**(-final_shape) - 1)
        CTE_threshold = (tail_quantile + final_scale - (final_shape * final_threshold))/(1 - final_shape)
        self.CTE_threshold = CTE_threshold
        
        return CTE_threshold

    def predict(self, X_test, X_train = None):
        """
        Assign class label 0:normal, 1:anomolous to each subsequence in X_test based upon the 1NN-DTW distance and CTE anomaly score threshold
        """
        if X_train is None:
            reference_subsequences = self.X_train
        else:
            reference_subsequences = X_train
        test_NN_df = pd.DataFrame(columns = ['subsequence_id', 'date_id', 'account_id', '1NN_DTW_distance', '1NN_subsequence_id'], index = range(len(X_test['subsequence_id'].unique())))
        
        #Compute 1NN-DTW distance scores for all subsequences in X_test
        if self.n_jobs == 1:
            for i, subsequence_id in enumerate(X_test['subsequence_id'].unique()):
                query_subsequence = X_test[X_test['subsequence_id'] == subsequence_id]
                query_subsequence_account_id = query_subsequence['account_id'].unique()[0]
                query_subsequence_date_id = query_subsequence['date_id'].unique()[0]

                query_subsequence = query_subsequence[self.features]                
                BSF_distance, BSF_subsequence_ID = NN_DTW_classifier._NN_DTW_search(self, query_subsequence = query_subsequence, reference_subsequences = reference_subsequences)

                test_NN_df.at[i, 'subsequence_id'] = subsequence_id
                test_NN_df.at[i, 'date_id'] =  query_subsequence_date_id
                test_NN_df.at[i, 'account_id'] =  query_subsequence_account_id
                test_NN_df.at[i, '1NN_subsequence_id'] = BSF_subsequence_ID
                test_NN_df.at[i, '1NN_DTW_distance'] = BSF_distance
    
        #Assign class labels
        test_NN_df['class_label'] = np.where(test_NN_df['1NN_DTW_distance'] > self.CTE_threshold, -1, 1)
        
        return test_NN_df
    
class OCSVM_classifier:
    """
    One-class Support Vector Machine classifier
    """
    
    def __init__(self, features, verbose = True):
        self.features = features
        self.verbose = verbose
    
    @staticmethod
    def calculate_distance(num, row, reference_dataset):
        """
        See Xiao, Y., Wang, H., Zhang, L. and Xu, W., 2014. 
        Two methods of selecting Gaussian kernel parameters for one-class SVM 
        and their application to fault detection. 
        Knowledge-Based Systems, 59, pp.75-84.
        """
        
        #create mask to remove current row from reference data 
        wanted = np.full(reference_dataset.shape[0], True)
        wanted[num] = False
        reference_dataset = reference_dataset[wanted]
        n,p = reference_dataset.shape
        distances = np.sum(np.square((np.tile(row, (n, 1)) - reference_dataset)), axis = 1)
        nearest_neighbour_distance = np.min(distances)
        furthest_neighbour_distance = np.max(distances)
        return nearest_neighbour_distance, furthest_neighbour_distance
    
    @staticmethod
    @jit(nopython = True)
    def dfn_function(sigma, far, near):
        """
        See eq.(6) of Xiao, Y., Wang, H., Zhang, L. and Xu, W., 2014. 
        Two methods of selecting Gaussian kernel parameters for one-class SVM and 
        their application to fault detection. Knowledge-Based Systems, 59, pp.75-84.

        """
        
        return -(((2/len(far)) * np.sum(np.exp(-(near/(sigma**2))))) - ((2/len(far)) * np.sum(np.exp(-(far/(sigma**2))))))

    def fit(self, X_train):
        """
        fit OCSVM to transactions in X_train

        """
        
        X_train.reset_index(inplace = True, drop = True)
        X_train_array = X_train[self.features].values #copy for speed
        
        nearest_distances = np.zeros(len(X_train))
        furthest_distances = np.zeros(len(X_train))
        #Find the radial basis function kernel bandwidth (gamma in sklearn)
        if self.verbose == True: print('Finding Distances') 
        for i in (tqdm(range(len(X_train))) if self.verbose == True else range(len(X_train))):
            data_row = X_train_array[i, :]
            distances = OCSVM_classifier.calculate_distance(num = i, row = data_row, reference_dataset = X_train_array)
            nearest_distances[i]= distances[0]
            furthest_distances[i] = distances[1]
            
        if self.verbose == True: print('Tuning gamma') 
        optimal_bandwidth_results = minimize(fun = OCSVM_classifier.dfn_function, x0 = [1], args = (furthest_distances, nearest_distances), 
                                     method = 'Nelder-Mead', options = {'maxiter':10000})
        self.optimal_bandwidth = np.sqrt(optimal_bandwidth_results['x'][0])
        
        #Fit OCSVM model
        one_class_SVM_model = svm.OneClassSVM(kernel = 'rbf', nu = 0.03, gamma = self.optimal_bandwidth)
        one_class_SVM_model.fit(X_train[self.features].values)
        
        self.one_class_SVM_model = one_class_SVM_model
    
    def predict(self, X_test, X_train = None):
        """
        predict class labels and the decision function for ROC curve
        """
        
        if X_train is not None: #retrain a new model using a revised training set - for use with simulations
            self.fit(X_train = X_train)
            
        class_labels = self.one_class_SVM_model.predict(X_test[self.features].values)
        decision_function_scores = self.one_class_SVM_model.decision_function(X_test[self.features].values)
        
        test_df = pd.DataFrame(data = {'class_label':class_labels, 'decision_function':decision_function_scores})
        return test_df

class EnsembleGaussianMixtureModel():
    """
    Ensemble gaussian Mixture Model. See Emmott, A., Das, S., Dietterich, T., Fern, A. and Wong, W.K., 2015. A meta-analysis of the anomaly detection problem. arXiv preprint arXiv:1503.01158.
    """
    
    def __init__(self, features, mixture_component_upper_limit, threshold_alpha, verbose = True):
        self.features = features
        self.mixture_component_upper_limit = mixture_component_upper_limit #maximum number of base estimators in the ensemble
        self.threshold_alpha = threshold_alpha
        self.verbose = verbose
        
    def _fit_GMM_cross_validate(self, components):
        """
        Fit a Gaussian Mixture Model with a specified number of components
        """
        
        KFold_object = KFold(n_splits = 5, random_state = 10, shuffle = True)
        KFold_splits = KFold_object.split(X = self.X_train)
        fold_loglikelihood = []
        for idx, split in enumerate(KFold_splits):
            train_split = self.X_train[self.X_train.index.isin(split[0])]
            test_split = self.X_train[self.X_train.index.isin(split[1])]
            
            if idx > 0: warm_start_bool = True
            GMM_model = mixture.GaussianMixture(n_components = components, 
                                                covariance_type = 'full', 
                                                random_state = idx, 
                                                max_iter = 1000, 
                                                warm_start = warm_start_bool)
            GMM_model.fit(train_split[self.features].values)
    
            log_scores = np.sum(GMM_model.score_samples(test_split[self.features].values)) #Compute log-likelihood
            fold_loglikelihood.append(log_scores)
            
        average_K_component_likelihood = np.mean(fold_loglikelihood)
        return components, average_K_component_likelihood
    
    def fit(self, X_train):
        """
        fit GMM to transactions in X_train
        """
        self.X_train = X_train
        self.mixture_component_upper_limit = int(np.floor(min(0.2*len(X_train), self.mixture_component_upper_limit)))
        
        train_df = pd.DataFrame(index = self.X_train.index)
        component_log_likelihood_dict = {}
        if self.verbose == True: print('Finding Loglikelihood range')
        for K_components in (tqdm(range(0, self.mixture_component_upper_limit)) if self.verbose == True else range(0, self.mixture_component_upper_limit)):
            component_log_likelihood_dict[K_components] = self._fit_GMM_cross_validate(components = K_components + 1)
        
        max_likelihood = np.max([x for x in component_log_likelihood_dict.values()])
        if max_likelihood < 0:
            components = [x[0] for x in component_log_likelihood_dict.values() if x[1] > (max_likelihood + 0.15*max_likelihood)]
        else:
            components = [x[0] for x in component_log_likelihood_dict.values() if x[1] > (max_likelihood - 0.15*max_likelihood)]
        self.components = components
        
        if self.verbose == True: print('Fitting ensemble GMM models')
        ensemble_component_model_fits = []
        for idx, ensemble_iteration in (tqdm(enumerate(self.components)) if self.verbose == True else enumerate(self.components)):
            if idx > 0: warm_start_bool = True
            ensemble_GMM_model = mixture.GaussianMixture(n_components = ensemble_iteration, 
                                                         covariance_type = 'full', 
                                                         random_state = ensemble_iteration, 
                                                         max_iter = 1000, 
                                                         warm_start = warm_start_bool)
            ensemble_GMM_model.fit(self.X_train[self.features].values)     
            ensemble_component_model_fits.append(ensemble_GMM_model)
            train_ensemble_scores = ensemble_GMM_model.score_samples(self.X_train[self.features].values)
            train_df[ensemble_iteration] = train_ensemble_scores
        self.ensemble_component_model_fits = ensemble_component_model_fits

        final_reference_ensemble_score = train_df.mean(axis = 1)
        ensemble_threshold = final_reference_ensemble_score.quantile(1 - self.threshold_alpha)
        self.threshold = ensemble_threshold

    def predict(self, X_test, X_train = None):
        """
        predict class labels and the log-likelihood scores
        """
        
        if X_train is not None: #retrain a new model using a revised training set - for use with simulations
            self.fit(X_train)
        
        test_scores_df = pd.DataFrame(index = X_test.index, columns = [x for x in self.components])
        for idx, ensemble_iteration in enumerate(self.components):
            ensemble_scores = self.ensemble_component_model_fits[idx].score_samples(X_test[self.features].values)
            test_scores_df[ensemble_iteration] = ensemble_scores
            
        final_test_ensemble_score = test_scores_df.mean(axis =1)
        class_labels = np.where(final_test_ensemble_score < self.threshold, -1, 1)
        test_df = pd.DataFrame(data = {'class_label':class_labels, 'decision_function':final_test_ensemble_score})
        
        return test_df

class IsolationForest_classifier:
    """
    Isolation forest. See Liu, F.T., Ting, K.M. and Zhou, Z.H., 2008, December. 
    In 2008 eighth ieee international conference on data mining (pp. 413-422). IEEE.
    
    """
    
    def __init__(self, features):
        self.features = features
        
    def fit(self, X_train):
        """

        """
        
        iforest_model_object = IsolationForest(n_estimators = 100, 
                                         max_samples = 'auto', 
                                         contamination = 'auto', 
                                         n_jobs = 1, 
                                         random_state = 1234)
        iforest_model_object.fit(X_train[self.features].values)
        self.iforest_model_object = iforest_model_object
        
    def predict(self, X_test, X_train = None):
        """

        """

        if X_train is not None: #retrain a new model using a revised training set - for use with simulations
            self.fit(X_train = X_train)
            
        class_labels = self.iforest_model_object.predict(X_test[self.features])
        decision_function_scores = self.iforest_model_object.decision_function(X_test[self.features])
        test_df = pd.DataFrame(data = {'class_label':class_labels, 'decision_function':decision_function_scores})
        
        return test_df

    
    
    
    