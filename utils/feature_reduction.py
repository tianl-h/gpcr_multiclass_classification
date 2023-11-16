# -*- coding: utf-8 -*-
# @Author: tianl
# @Date:   2020-12-17 11:59:13
# @Last Modified by:   tianl
# @Last Modified time: 2020-12-17 22:41:53

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix

## select top 100 features
def feature_reduction(data_):
    # tsvd = TruncatedSVD(n_components=data_all_features.shape[1]-1, n_iter=7, random_state=42)
    # X_tsvd = tsvd.fit(data_all_features)

    # # List of explained variances
    # tsvd_var_ratios = tsvd.explained_variance_ratio_

    # # Run function and return
    # n_comp = select_n_components(tsvd_var_ratios, 0.95)

    tsvd_optm = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
    tsvd_optm.fit(data_)
    data_red = tsvd_optm.transform(data_)

    return data_red



# # Create a function
# def select_n_components(var_ratio, goal_var: float) -> int:
#     # Set initial variance explained so far
#     total_variance = 0.0
    
#     # Set initial number of features
#     n_components = 0
    
#     # For the explained variance of each feature:
#     for explained_variance in var_ratio:
        
#         # Add the explained variance to the total
#         total_variance += explained_variance
        
#         # Add one to the number of components
#         n_components += 1
        
#         # If we reach our goal level of explained variance
#         if total_variance >= goal_var:
#             # End the loop
#             break
            
#     # Return the number of components
#     return n_components







