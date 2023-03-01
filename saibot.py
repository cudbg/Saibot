import pandas as pd 
import numpy as np
import math
from math import e
import random
import sklearn.datasets

def get_pos_def_mean_cov(num_att):
    cov = sklearn.datasets.make_spd_matrix(num_att)
    # the mean value is a random number from [-1,1]
    mean = [2* (random.random() - 0.5) for _ in range(num_att)]
    return mean, cov

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

# clip rows above b in the dataframe
def clip_data(df, b):
    for i, row in df.iterrows():
        cur_norm = np.linalg.norm(row.to_numpy())
        if cur_norm > b:
            df.at[i] = row * b / cur_norm

def union_sensitivity(b,single_feature=False):
    if not single_feature:
        return math.sqrt(2*b**4 + 4*b**2)
    else:
        return math.sqrt(b**4 + 4*b**2)

def join_sensitivity(b, single_feature=False):
    if not single_feature:
        return max(math.sqrt(2*b**4 + 4*b**2), math.sqrt(2 + 2*b**2 + 2*b**4))
    else:
        return max(math.sqrt(b**4 + 4*b**2), math.sqrt(2 + 2*b**2 + 2*b**4))

def join_sensitivity_opt(b):
    return (2)**0.5, 2*b, (2)**0.5*b*b

# iterate each row of df and compute the largest l2 norm
def get_l2_bound(df):
    max_bound = 0
    for _, row in df.iterrows():
        cur_bound = 0
        for col in df.columns:
            cur_bound += row[col]**2
        max_bound = max(max_bound, cur_bound)
    return max_bound ** 0.5

def get_l2_distance(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        keys = a.keys()
        diff_squared_sum = sum((a[key] - b[key]) ** 2 for key in keys)
        return np.sqrt(diff_squared_sum)
    else:
        return np.linalg.norm(a - b)

def get_cos_sim(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

# This function generates a dataframe with num_rows number of rows where each row satisfies a given constraint.
# The constraint is that the norm of each row should be less than or equal to B.
# The dataframe is generated from a multivariate normal distribution with the specified mean and covariance matrix.
def generate_dataframe(mean, cov, num_rows, columns, B):
    # Initialize a dataframe with one row generated from mean and cov
    df = pd.DataFrame(np.random.multivariate_normal(mean, cov, 1), columns=columns)
    
    # Keep generating new rows and adding them to the dataframe until the number of rows reaches num_rows
    while df.shape[0] < num_rows:
        # Generate a new row from mean and cov
        new_row = np.random.multivariate_normal(mean, cov, 1)
        
        # Check if the norm of the new row is less than or equal to B
        if np.linalg.norm(new_row) <= B:
            # Add the new row to the dataframe
            df = pd.concat([df, pd.DataFrame(new_row, columns=columns)], ignore_index=True)
            
    # Return the final dataframe
    return df

def sanity_check(cov_matrix, features, alpha):
    a = np.empty([len(features) + 1, len(features) + 1])
    
    for i in range(len(features)):
        for j in range(len(features)):
            if 'cov:Q:' + features[i] + ","+ features[j] in cov_matrix:
                a[i][j] = cov_matrix['cov:Q:' + features[i] + ","+ features[j]]
            else:
                a[i][j] = cov_matrix['cov:Q:' + features[j] + ","+ features[i]]
        if i == j:
            a[i][i] += alpha
        a[i][-1] = cov_matrix['cov:s:' + features[i]]
        a[-1][i] = cov_matrix['cov:s:' + features[i]]
    
    a[len(features)][len(features)] = cov_matrix['cov:c']
    
    if cov_matrix['cov:c'] <= 0:
        return False
    
    if not is_pos_def(a):
        return False
    
    return True

# std of natural distirbution for guassian noises
def compute_std(eps, delta, sensitivity=1):
    return (2*math.log(1.25/delta))**0.5*sensitivity/eps

# std of natural distirbution for guassian noises
def compute_std_lap(eps, sensitivity=1):
    return sensitivity/eps

# Generalized Randomized Response
def random_response(cur, total, eps):
    if cur >= total:
        raise ValueError("cur must be less than total")

    p_cur = math.exp(eps) / (math.exp(eps) + total - 1)
    p_other = 1 / (math.exp(eps) + total - 1)
    r = random.random()

    if r <= p_cur:
        return cur
    else:
        other_values = [i for i in range(total) if i != cur]
        return random.choice(other_values)

# return the coefficients of features and a constant 
def ridge_linear_regression(cov_matrix, features, result, alpha):
    a = np.empty([len(features) + 1, len(features) + 1])
    b = np.empty(len(features) + 1)
    
    for i in range(len(features)):
        for j in range(len(features)):
            if 'cov:Q:' + features[i] + ","+ features[j] in cov_matrix:
                a[i][j] = cov_matrix['cov:Q:' + features[i] + ","+ features[j]]
            else:
                a[i][j] = cov_matrix['cov:Q:' + features[j] + ","+ features[i]]
        if i == j:
            a[i][i] += alpha
    
    for i in range(len(features)):
        a[i][len(features)] = cov_matrix['cov:s:' + features[i]]
        a[len(features)][i] = cov_matrix['cov:s:' + features[i]]
        if 'cov:Q:' + result + "," + features[i] in cov_matrix:
            b[i] = cov_matrix['cov:Q:' + result + "," + features[i]]
        else:
            b[i] = cov_matrix['cov:Q:' + features[i] + "," + result]
    
    b[len(features)] = cov_matrix['cov:s:' + result]
    
    a[len(features)][len(features)] = cov_matrix['cov:c']
    return np.linalg.solve(a, b)


def square_error(cov_matrix, features, result, parameter):
    se = cov_matrix['cov:Q:'  + result + "," + result]
    
    for i in range(len(features)):
        for j in range(len(features)):
            if 'cov:Q:'  + features[i] + "," + features[j] in cov_matrix:
                se += parameter[i] * parameter[j] * cov_matrix['cov:Q:'  + features[i] + "," + features[j]]
            else:    
                se += parameter[j] * parameter[i] * cov_matrix['cov:Q:'  + features[j] + "," + features[i]]
    
    for i in range(len(features)):
        se += 2 * parameter[i] * parameter[-1] * cov_matrix['cov:s:'  + features[i]]
        if 'cov:Q:' + result + "," + features[i] in cov_matrix:
            se -= 2 * parameter[i] *  cov_matrix['cov:Q:' + result + "," + features[i]]
        else:
            se -= 2 * parameter[i] *  cov_matrix['cov:Q:' + features[i] + "," + result]
    
    se -= 2 * parameter[-1] * cov_matrix['cov:s:'  + result]
    se += cov_matrix['cov:c'] * parameter[-1] * parameter[-1]

    return se

def total_sum_of_square(cov_matrix, result):
    return cov_matrix['cov:Q:'  + result + "," + result] - cov_matrix['cov:s:'  + result] * cov_matrix['cov:s:'  + result] / cov_matrix['cov:c']

def mean_squared_error(cov_matrix, features, result, parameter):
    return square_error(cov_matrix, features, result, parameter)/cov_matrix['cov:c']


def r2(cov_matrix, features, result, parameter):
    result =  1 - square_error(cov_matrix, features, result, parameter)/total_sum_of_square(cov_matrix, result)
    if result > 2:
        # overflow
        return -1
    return result

def adjusted_r2(cov_matrix, features, result, parameter):
    return 1 - (cov_matrix['cov:c']-1)*(1 - r2(cov_matrix, features, result, parameter))/(cov_matrix['cov:c'] - len(parameter) - 1)

# a wrapper class that keeps some meta data
class agg_dataset:
    # load data (in the format of dataframe)
    # user provides dimensions to join (these dimensions will be pre-aggregated)
    def load(self, data, X, dimensions):
        self.data = data
        self.dimensions = dimensions
        self.X = X
        
    def semi_ring_columns(self):
        return list(filter(lambda col: col.startswith("cov:"), self.data.columns))
    
    # compute the semi-ring aggregation for each dimension
    def compute_agg(self, lift = True):
        # build semi-ring structure
        if lift:
            self.lift(self.X)
        
        self.agg_dimensions = dict()
        
        for d in self.dimensions:
            self.agg_dimensions[d] = self.data[self.semi_ring_columns() + [d]].groupby(d).sum()
        
        # without groupby
        self.agg = self.data[self.semi_ring_columns()].sum()
        
    # build gram matrix semi-ring
    def lift(self, attributes):
        self.data['cov:c'] = 1
        for i in range(len(attributes)):
            for j in range(i, len(attributes)):
                self.data['cov:Q:' + attributes[i] + "," + attributes[j]] = self.data[attributes[i]] * self.data[attributes[j]]

        for attribute in attributes:
            self.data= self.data.rename(columns = {attribute:'cov:s:' + attribute})
            

# Given a gram matrix semi-ring, normalize it
def normalize(cov, kept_old= False):
    cols = []
    
    if isinstance(cov, pd.DataFrame):
        cols = cov.columns
    # this is for the final semiring, which is reduced to a single np array
    else:
        cov = cov.astype(float)
        cols = list(cov.axes[0])
        
    for col in cols:
        if col != 'cov:c':
            cov[col] = cov[col]/cov['cov:c']
    
    if kept_old:
        if isinstance(cov, pd.DataFrame):
            # kept the old to estimate join result size
            cov.rename(columns={'cov:c':'cov:c_old'}, inplace=True)
            
    cov['cov:c'] = 1
    return cov

def join_agg(aggdata1, aggdata2, dimension):
    if isinstance(dimension, list):
        dimension = tuple(dimension)
    
    agg1 = aggdata1.agg_dimensions[dimension]
    agg2 = aggdata2.agg_dimensions[dimension]
    
    left_attributes = aggdata1.X
    right_attributes = aggdata2.X

    join = pd.merge(agg1, agg2, how='inner', left_index=True, right_index=True)
    
    # Calculate covariance for each combination of attributes from both sets
    for att1 in left_attributes:
        for att2 in right_attributes:
            cov_name = f"cov:Q:{att1},{att2}"
            if cov_name not in join:
                cov_name = f"cov:Q:{att2},{att1}"
            join[cov_name] = (join[f'cov:s:{att2}'] * join[f'cov:s:{att1}'])
    
    # Update covariance for left attributes
    for att in left_attributes:
        join[f'cov:s:{att}'] *= join['cov:c_y']
    
    # Update covariance for right attributes
    for att in right_attributes:
        join[f'cov:s:{att}'] *= join['cov:c_x']
    
    # Calculate covariance for left attributes
    for i in range(len(left_attributes)):
        for j in range(i, len(left_attributes)):
            att1 = left_attributes[i]
            att2 = left_attributes[j]
            cov_name = f"cov:Q:{att1},{att2}"
            if cov_name not in join:
                cov_name = f"cov:Q:{att2},{att1}"
            join[cov_name] *= join['cov:c_y']
    
    # Calculate covariance for right attributes
    for i in range(len(right_attributes)):
        for j in range(i, len(right_attributes)):
            att1 = right_attributes[i]
            att2 = right_attributes[j]
            cov_name = f"cov:Q:{att1},{att2}"
            if cov_name not in join:
                cov_name = f"cov:Q:{att2},{att1}"
            join[cov_name] *= join['cov:c_x']
        
    # Calculate final covariance
    join['cov:c'] = join['cov:c_x'] * join['cov:c_y']
    
    # Remove intermediate columns
    join.drop(labels=['cov:c_x', 'cov:c_y'], axis=1, inplace=True)
    
    return join

def unbiased_est(cov, buyer_columns, seller_columns, d, n):
    for att1 in buyer_columns:
        for att2 in seller_columns:
            s1 = cov[f"cov:s:{att1}"]/cov['cov:c']
            s2 = cov[f"cov:s:{att2}"]/cov['cov:c']
            cov_name = f"cov:Q:{att1},{att2}"
            if cov_name not in cov:
                cov_name = f"cov:Q:{att2},{att1}"
            cov[cov_name] = (n - 1)/(d-1)*cov[cov_name]/cov['cov:c'] + (d- n)/(d-1)*s1*s2
    
    for i in range(len(buyer_columns)):
        for j in range(i, len(buyer_columns)):
            att1 = buyer_columns[i]
            att2 = buyer_columns[j]
            cov_name = f"cov:Q:{att1},{att2}"
            if cov_name not in cov:
                cov_name = f"cov:Q:{att2},{att1}"
            cov[cov_name] = cov[cov_name]/cov['cov:c']
    
    for i in range(len(seller_columns)):
        for j in range(i, len(seller_columns)):
            att1 = seller_columns[i]
            att2 = seller_columns[j]
            cov_name = f"cov:Q:{att1},{att2}"
            if cov_name not in cov:
                cov_name = f"cov:Q:{att2},{att1}"
            cov[cov_name] = cov[cov_name]/cov['cov:c']
    
    for att in buyer_columns + seller_columns:
        name = 'cov:s:' + att
        cov[name] = cov[name]/cov['cov:c']
    cov['cov:c'] = 1
    return cov
