from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hdbscan

class ClusterMaster:
    '''
    This variable is important to select the new image similarity and tag frequency columns
    for the second, non-spatial clustering.
    This value is strongly dependant on the imported dataframe and the columns that already exist
    (in prospect of querying the database instead of FlickrAPI and the metadata .csv import!)
    '''
    core_df_range = 17
    index_lat = 10
    index_lng = 11
    def __init__(self, params_dict, used_lowe_ratio=0, data_path=None, subset_df=None, spatial_clustering=True, multi_clustering_inc_coordinates=False):
        print("--" * 30)
        print("Initialising Data Clustering with ClusterMaster Class")
        self.params = params_dict
        self.data_path = data_path
        self.spatial_clustering = spatial_clustering
        self.multi_clustering_inc_coordinates = multi_clustering_inc_coordinates
        self.subset_df = subset_df
        self.df = self.read_data()
        print("--" * 30)
        print("Reading metadata dataframe - done.")
        self.used_lowe_ratio = used_lowe_ratio
        self.unique_labels = self.clustering()
        print("--" * 30)
        print("Clustering process - done.")
        print("--" * 30)

    def read_data(self):
        if self.spatial_clustering:
            dataframe = pd.read_csv(self.data_path, delimiter=';', index_col='photo_id')
        else:
            dataframe = self.subset_df
        print(f"Dataframe shape: {dataframe.shape}")

        return dataframe

    def clustering(self):
        # cluster the data
        print(f"using {self.params['algorithm']} clustering on {self.df.shape[0]} entries")
        if self.params['algorithm'] == 'HDBSCAN':
            self.min_cluster_size = self.params['min_cluster_size']
            self.min_samples = self.params['min_samples']
            self.clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples)


        elif self.params['algorithm'] == 'DBSCAN':
            self.eps = self.params['eps']
            self.min_samples = self.params['min_samples']
            self.n_jobs = self.params['n_jobs']
            self.clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=self.n_jobs)  # n_jobs=-1 uses all available CPU cores

        if self.spatial_clustering:
            '''
            if primary clustering (spatial) is desired
            then coordinates lat/lng pairs are used as clustering features
            '''
            #performs clustering on X and returns cluster labels / if only .fit(X) is used, the labels can be accessed with dbscan.labels_
            input_features = self.df.loc[:, ['lat', 'lng']]

        else:
            '''
            if image similarity and tag clustering shall be performed
            the clustering input features will be defined below
            
            OPTION 1:
            multi clustering with only the image similiarity matrix
            '''
            if not self.multi_clustering_inc_coordinates:
                input_features = self.df.iloc[:, ClusterMaster.core_df_range:]
                '''
                Check for rows that consist of only Zero (0.0000) scores
                exclude them from further clustering
                '''
                boolean_array_zero_vals = (input_features.nunique(axis=0) == 1)# axis 0 represents index (rows), 1 unique element, namely 0
                input_features = input_features[~boolean_array_zero_vals]  # ~ is used to invert boolean series
            '''
            OPTION 2:
            multi clustering with the image similiarity matrix + again the coordinates
            '''
            if self.multi_clustering_inc_coordinates:
                input_image_similiarity_matrix = self.df.iloc[:, ClusterMaster.core_df_range:]
                input_coordinates = self.df.loc[:, ['lat', 'lng']]
                input_features_ = input_image_similiarity_matrix.join(input_coordinates, how='left')
                '''
                Check for rows that consist of only Zero (0.0000) scores
                exclude them from further clustering
                '''
                boolean_array_zero_vals = (input_image_similiarity_matrix.nunique(axis=0) == 1)  # axis 0 represents index (rows), 1 unique element, namely 0
                input_features = input_features_[~boolean_array_zero_vals]  # ~ is used to invert boolean series
                print("")

        if input_features.shape[0] == 0 or input_features.shape[1] == 0:
            if self.spatial_clustering:
                self.df.loc[:, 'spatial_cluster_label'] = -1
            else:
                self.df.loc[:, 'multi_cluster_label'] = -1

            print("No more input features left! No Clusters")
            return None

        else:
            clusters_labels = self.clusterer.fit_predict(input_features) #array of numbers where unique numbers identify points that belong to the same cluster; -1 is noise
            n_clusters = len(set(clusters_labels)) - (1 if -1 in clusters_labels else 0)
            n_noise = list(clusters_labels).count(-1)
            print(f"Clusters: {n_clusters}")
            '''
            Add the cluster labels as new column (series in Pandas) to the dataframe.
            The order will be consistant        
            '''
            if self.spatial_clustering:
                self.df.loc[:, 'spatial_cluster_label'] = pd.Series(clusters_labels, index=self.df.index)
            else:
                self.df.loc[:, 'multi_cluster_label'] = pd.Series(boolean_array_zero_vals, index=self.df.index)

                # assumption that the cluster labels are still in the same order as the df -> proven true
                index_c = 0
                for i, v in self.df.iterrows():
                    #check if True -> all Zero score values
                    if v['multi_cluster_label']:
                        self.df.loc[i, 'multi_cluster_label'] = -1
                    else:
                        self.df.loc[i, 'multi_cluster_label'] = clusters_labels[index_c]
                        #increment if spot was found which was used as cluster feature input -> iterate over cluster_labels
                        index_c += 1
            '''
            Plot
            only if clusters exist
            '''
            unique_labels = set(clusters_labels)

        if n_clusters > 0:
            # Black removed and is used for noise instead.
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
            '''
            Sort labels so that the noise points get plotted first 
            and don't cover important clusters
            '''
            for cluster_label, col in zip(sorted(unique_labels, key=lambda x: x), colors):
                #filter dataframe for rows of given cluster
                label = cluster_label
                if cluster_label == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]
                    label = 'noise'
                #returns boolean array with true when condition is met
                if self.spatial_clustering:
                    boolean_array = self.df['spatial_cluster_label'] == cluster_label
                else:
                    boolean_array = self.df['multi_cluster_label'] == cluster_label
                rows = self.df[boolean_array]
                '''
                Adjust plot X and Y extend on
                all points except noise
                
                1. query all points that are not noise from dataframe
                '''
                boolean_array_not_noise = self.df.spatial_cluster_label != -1
                not_noise = self.df[boolean_array_not_noise]

                buffer = 0.0005

                xlim_left = not_noise.lng.min() - buffer
                xlim_right = not_noise.lng.max() + buffer
                ylim_bottom = not_noise.lat.min() - buffer
                ylim_top = not_noise.lat.max() + buffer
                plt.xlim(left=xlim_left, right=xlim_right)
                plt.ylim(bottom=ylim_bottom, top=ylim_top)
                '''
                latitude and longitude
                must most likely be exchanged for some reason
                to match the ArcMap reprentation
                '''
                plt.plot(rows.loc[:, 'lng'], rows.loc[:, 'lat'], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14, label=label) #replaced xy with X

            if self.params['algorithm'] == 'HDBSCAN':
                plt.title(f'nr clusters: {n_clusters}, min_cluster_size: {self.min_cluster_size}, '
                          f'min samples: {self.min_samples}, image LOWE ratio: {self.used_lowe_ratio}')
            elif self.params['algorithm'] == 'DBSCAN':
                plt.title(f'nr clusters: {n_clusters}, eps: {self.eps}, min samples: {self.min_samples}, image threshold: {self.used_lowe_ratio}')
            plt.legend()
            plt.show()

        else:
            print("No clusters found.")
        return unique_labels
        #or add cluster_labels as new column to the dataframe and return that
