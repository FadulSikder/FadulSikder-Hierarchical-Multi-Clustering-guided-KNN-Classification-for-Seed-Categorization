

import numpy as np
import pandas as pd
import math, sys

def load_data():
    #loading data from cvs file
    data = pd.read_csv("Seed_Data.csv")
    X = data.iloc[:, :-1]
    Y = data.iloc[:,-1:]
    return X, Y

#get_distance_matrix() use this to get euclidean_distance
def get_euclidean_distance(x1, x2):
        d = 0.0
        for i in range(0, len(x1)):
            d = d + (x1[i] - x2[i]) ** 2
        return math.sqrt(d)

def get_distance_matrix(data):

      temp_data = data.to_numpy()
      #creating a 210x210 distance matrix with zero
      distance_matrix = np.zeros((temp_data.shape[0],temp_data.shape[0]))

      #loop to fill the distance matrix
      for i in range(distance_matrix.shape[0]):
          for j in range(distance_matrix.shape[0]):
              distance_matrix[i][j] = get_euclidean_distance(temp_data[i], temp_data[j])

      #filling the distance matrix diagonal with big number from zero
      np.fill_diagonal(distance_matrix, sys.maxsize)
      
      return distance_matrix

def cluster_all_point(distance_matrix,cluster_algo):
      array_clusters = {}
      cluster_id = []
      row_id = 0
      col_id = 0

      # Storing all datapoint position in an list and naming it cluster id 
      
      for n in range(distance_matrix.shape[0]):
          cluster_id.append(n)
      
      # #storing the first cluster id on cluster list
      array_clusters[0] = cluster_id.copy()
  
      # after first cluster Creating a new cluster until we reach one single cluster
      for k in range(1, distance_matrix.shape[0]):
          min_val = sys.maxsize
          

          #collecting the min value from matrix and taking the position id. This position id will merge as a cluster
          for i in range(distance_matrix.shape[0]):
              for j in range(distance_matrix.shape[1]):
                  if distance_matrix[i][j] <= min_val:
                      min_val = distance_matrix[i][j]
                      row_id = i
                      col_id = j
          
          # Update the distance matrix according to algo
          for i in range(distance_matrix.shape[0]):
              if i != col_id:
                  if cluster_algo == 'average':
                      temp = 0.5*(distance_matrix[col_id][i]+distance_matrix[row_id][i])
                  elif cluster_algo == 'single':
                      temp = min(distance_matrix[col_id][i],distance_matrix[row_id][i])
                  else:
                      temp = max(distance_matrix[col_id][i],distance_matrix[row_id][i])
                  
                  # Symmetric update of distance matrix
                  distance_matrix[col_id][i] = temp
                  distance_matrix[i][col_id] = temp


          #setting the two merged position's values to max
          for i in range (distance_matrix.shape[0]):
              distance_matrix[row_id][i] = sys.maxsize
              distance_matrix[i][row_id] = sys.maxsize
          

          minimum = min(row_id,col_id)
          maximum = max(row_id,col_id)


          #generating new cluster id by encode on top of the previous cluster id  
          for n in range(len(cluster_id)):
              if cluster_id[n] == maximum:
                  cluster_id[n] = minimum


          #storing the new cluster id on cluster list
          array_clusters[k] = cluster_id.copy()


      return array_clusters

def cluster(data):
      # Distance Matrix
      distance_matrix = get_distance_matrix(data)

      #Finding the clusters
      array_clusters = cluster_all_point(distance_matrix)
      
      return array_clusters

def divide_in_specific_no_of_cluster(data_with_new_feature, algo_clusters,number_of_clusters):
      
      cluster_algo = ['average','single','complete']
      indices_of_clusters = []
      
      #algo cluster store all the cluster for the three algo
      # we will iterate through it and generate specific number of cluster

      for i in range(len(algo_clusters)):
          #print('\033[1m'+"Clustering algorihtm : ", cluster_algo[i],' with ', number_of_clusters,' cluster'+ '\033[0m')

          # Getting n clusters and save them backward
          array_clusters = algo_clusters[i]
          n = len(array_clusters) - number_of_clusters
          cluster = array_clusters[n]
          
          # Getting individual cluster
          unique_arr = np.unique(cluster)
          n_clusters = []
          for n in np.nditer(unique_arr):
              n_clusters.append(np.where(cluster == n))

          
          #storing the cluster id into the feature vector
          for j in range(len(n_clusters)):
              #print("Cluster ", j + 1, " : ", n_clusters[j][0])
              indices = n_clusters[j][0]
              data_with_new_feature.loc[indices,cluster_algo[i]] = j

          #storing the indices of the cluster of each elgo into indices_of_clusters variable
          indices_of_clusters.append(n_clusters)
    
      return indices_of_clusters, data_with_new_feature

def creating_new_feature_with_similarity_measuement(distance_matrix, indices_of_clusters, data_with_new_feature):
   
    # inverting the distance matrix as we will use this matrix to vote
    distance_matrix = 1/distance_matrix


    #iterating through each datapoint and measuring the similarity between cluster and datapoints 
    #creating a new feature vector based on that information
    for ind in data_with_new_feature.index:

        #taking a data point of cluster ids of each datapoints
        data_point =data_with_new_feature.loc[[ind]]
        data_point = data_point[['average','single','complete']].values.reshape(-1)

        temp2_arr = np.empty(0)
        temp3_arr = np.arange(start=0, stop=210, step=1)

        #Finding the cluster union and intersection
        for j in range(len(data_point)):
            temp1_arr = indices_of_clusters[j]
            n = int(data_point[j])
            temp2_arr = np.union1d(temp2_arr, temp1_arr[n][0])
            temp3_arr = np.intersect1d(temp3_arr, temp1_arr[n][0])

        #calculating similarity_coficient between the multiple clusters of a data point
        similarity_cofficient = len(temp3_arr)/len (temp2_arr)


        #updating the feature vactor with a voting algo in place of cluster_id
        for j in range(len(data_point)):
            temp1_arr = indices_of_clusters[j]
            n = int(data_point[j])
            vote = distance_matrix[ind, temp1_arr[n][0]].sum()
            data_with_new_feature.iloc[ind, 7 + j ] = 1 /(vote + similarity_cofficient)
            #print()

    return data_with_new_feature

#this one dimentional distance matrix is used by KNN algorithm 
def cartesian_distance(feature, converted_array_of_data_point):
    distance_matrix = np.power((feature - converted_array_of_data_point), 2)
    distance_matrix = distance_matrix.astype(float)
    return np.sqrt(np.sum(distance_matrix, axis=1))

def knn_algorithm(feature, labels, data_point, k):
    #intializing variable
    row = feature.shape[0]
    collum = feature.shape[1]
    weighted_vote_kama = 0
    weighted_vote_rosa = 0
    weighted_vote_canadian = 0

    #getting cartesian distance and sorting the feature vector and separating the k nearest point
    converted_array_of_data_point = np.full((row, collum), data_point)
    distance = cartesian_distance(feature, converted_array_of_data_point)
    distance_labels = np.column_stack((distance, labels))
    distance_labels = distance_labels[np.argsort(distance_labels[:, 0])]
    k_nearest_points = distance_labels[:k, :]

    #weighted voting from nearest points to select the label for the data point
    for i in range(k):
        if k_nearest_points[i,1] == 0 :
            weighted_vote_kama = weighted_vote_kama + (1/k_nearest_points[i,0])
        elif k_nearest_points[i,1] == 1:
            weighted_vote_rosa = weighted_vote_rosa + (1/k_nearest_points[i,0])
        else:
            weighted_vote_canadian = weighted_vote_canadian + (1/k_nearest_points[i,0])

    #slecting the label based on highest vote
    if (weighted_vote_kama >= weighted_vote_rosa) and (weighted_vote_kama >= weighted_vote_canadian):
        result = 0
    elif (weighted_vote_rosa > weighted_vote_kama) and (weighted_vote_rosa > weighted_vote_canadian):
        result = 1
    else:
        result = 2

    return result

def leave_one_out_evaluation(k, data_with_new_feature):

    loop_var = data_with_new_feature.shape[0]
    error_count = 0

    #itareting through feature vector
    for i in range(loop_var):
        #droping row
        dropped_row = data_with_new_feature.iloc[[i], :]
        data = data_with_new_feature.drop(data_with_new_feature.index[i])
        
        #creating new feature and labels after droping one point
        feature = data.iloc[:, :-1].values
        labels = data.iloc[:,-1:].to_numpy()

        #formating the droped data point
        data_point = dropped_row.iloc[:, :-1].to_numpy()
        expected_prediction = dropped_row.iloc[:,-1:].to_numpy().flatten()
        
        #sending the droped data point and rest feature and labels to perform kNN classification
        result = knn_algorithm(feature, labels, data_point, k)

        #counting error
        if result != expected_prediction:
            error_count = error_count + 1

    #calculating accuracy
    percent_of_error = error_count / (loop_var-1) * 100
    prediction_accuracy = 100 - percent_of_error

    return prediction_accuracy

def main():
    #intailizing variables
    data,labels = load_data()
    data_with_new_feature = data.copy()
    algo_clusters = []
    data_with_new_feature['average'] = ''
    data_with_new_feature['single'] = ''
    data_with_new_feature['complete'] = ''

    #getting distance matrix for performing clustering
    distance_matrix = get_distance_matrix(data)
    #performing hierarchical clustering and collecting the clustering result
    for cluster_algo in ['average','single','complete']:
        distance_matrix_copy = distance_matrix.copy()
        array_clusters = cluster_all_point(distance_matrix_copy,cluster_algo)
        algo_clusters.append(array_clusters)

    data = pd.concat([data, labels], axis=1, join='inner')
    accuracy2 = [0, 0, 0]
    j = 0
    for k in [3, 5, 7]:
        accuracy2[j] = leave_one_out_evaluation(k, data)
        print('\033[1m' + 'Prediction accuracy of species without Hierarchical clustering and K is ', k, ' ==>',
              accuracy2[j], '%' + '\033[0m')
        j = j + 1

    for number_of_clusters in [4,6,8,12,16,20,24,27,30,60]:

        #dividing the clusters into specific cluster number and adding cluster label on each datapoint of feature vector
        indices_of_clusters, data_with_new_feature = divide_in_specific_no_of_cluster(data_with_new_feature, algo_clusters, number_of_clusters)
        
        #creating new feature vector considering the similarity beatween clusters and data point 
        data_with_new_feature = creating_new_feature_with_similarity_measuement(distance_matrix,indices_of_clusters, data_with_new_feature)

        data_with_new_feature = pd.concat([data_with_new_feature, labels], axis=1, join='inner')
        accuracy = [0,0,0]
        j = 0

        for k in [3,5,7]:
          accuracy[j] = leave_one_out_evaluation(k, data_with_new_feature)
          print('\033[1m'+'Prediction accuracy of species when Cluster Number is ',number_of_clusters,'and K is ', k, ' ==>', accuracy[j],'%'+ '\033[0m')
          j = j + 1


    return

if __name__ == '__main__':
    main()