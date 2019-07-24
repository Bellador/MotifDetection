from query_flickr_api import FlickrQuerier
from clustering import ClusterMaster
from image_feature_detection import ImageSimilarityAnalysis
from network_analysis import NetworkAnalyser
import os
import warnings
import datetime

def pickle_dataframes(index, dataframe, cluster_params, image_params):
    try:
        pickle_path = os.path.join(main_dir_path, project_name, 'dataframe_pickles')
        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
        name = '{}_{}_{}_{}_{}_{}_{:%m_%d_%H}.pkl'.format(cluster_params['algorithm'], cluster_params['min_cluster_size'], cluster_params['min_samples'],
                                        image_params['algorithm'], image_params['lowe_ratio'], index, datetime.datetime.now())
        dataframe.to_pickle(os.path.join(pickle_path, name))
        print(f"Pickling: {name}")

    except Exception as e:
        print(f"Error: {e} occurred while pickling dataframe {index}")

def cluster_html_inspect(index, dataframe, cluster_params, image_params):
    '''
    create an html file that can be insepcted in the browser that links
    images contained in clusters directly to their source path for
    easy inspection

    :param index:
    :param dataframe:
    :return:
    '''
    #create folder in project_path with the name cluster_hmtl_inspect
    folder_name = 'cluster_hmtl_inspect'
    html_path = os.path.join(project_path, folder_name)
    if not os.path.exists(html_path):
        os.makedirs(html_path)
        print(f"Creating project folder {folder_name} in current directory - done.")
    else:
        print(f"Project folder {folder_name} exists already.")

    file_name = '{}_{}_{}_{}_{}_{}_{:%m_%d_%H}.html'.format(cluster_params['algorithm'], cluster_params['min_cluster_size'], cluster_params['min_samples'],
                                         image_params['algorithm'], image_params['lowe_ratio'], index,
                                         datetime.datetime.now())

    with open(os.path.join(html_path, file_name), 'w') as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<hmtl>\n")
        f.write("<head>\n")
        f.write(f"<title>Multi Cluster {index}</title>\n")
        f.write("</head>\n")
        f.write("<body>\n")
        f.write(f"<h1>Image Similarity Clustering: {index}</h1>")
        # get the amount of cluster
        cluster_labels = set(dataframe.loc[:, 'multi_cluster_label'])
        n_clusters = sum([1 for c in cluster_labels if c != -1])

        cluster_dict = {}
        for label in cluster_labels:
            cluster_dict[label] = []

        # append media objects to correct cluster
        for i, row in dataframe.iterrows():
            cluster_label = row['multi_cluster_label']
            cluster_dict[cluster_label].append(i)

        for counter, (k, v) in enumerate(cluster_dict.items()):
            f.write(f'<h2>Cluster {k}</h2>\n')
            f.write(f'<ul>\n')

            for id in v:
                img_path = os.path.join(project_path, f'images_{project_name}', str(id) + '.jpg').replace('\\', '/')
                f.write(f'<li><img src="{img_path}" alt="{id}", height="300", width="300"><h3>{id}</h3></li>\n')

            f.write(f'</ul>\n')
        f.write("</body>\n")
        f.write("</html>\n")

    print(f"Created inspection html file {file_name} in folder {folder_name}")


if __name__ == '__main__':
    '''
    -----
    CLUSTERING INPUT PARAMETERS
    - min_samples: the higher the value the more conservative the clustering. Meaning that more points will be 
    considered noise which results in more dense clusters! (Also good parameter to play with)
    -----
    '''
    cluster_params_HDBSCAN_spatial = {
        'algorithm': 'HDBSCAN',
        'min_cluster_size': 20, #20
        'min_samples': 20 #20
    }

    cluster_params_HDBSCAN_multi = {
        'algorithm': 'HDBSCAN',
        'min_cluster_size': 2,
        'min_samples': 2
    }

    cluster_params_DBSCAN = {
        'algorithm': 'DBSCAN',
        'eps': 0.00015,
        'min_samples': 10,
        'n_jobs': 0
    }
    '''
    -----
    IMAGE SIMILARITY INPUT PARAMETERS
    -----
    '''
    SIFT_params = {
        'algorithm': 'SIFT',
        'lowe_ratio': 0.7, #0.775
    }

    SURF_params = {
        'algorithm': 'SURF',
        'lowe_ratio': 0.7,
    }

    ORB_params = {
        'algorithm': 'ORB',
        'lowe_ratio': 0.7,
    }

    spatial_clustering_params = cluster_params_HDBSCAN_spatial
    multi_clustering_params = cluster_params_HDBSCAN_multi
    image_similarity_params = SIFT_params

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main_dir_path = os.path.dirname(os.path.realpath(__file__))
        # project_name = input("Enter a project name. Will be integrated in folder and filenames: \n")

        project_name = 'wildkirchli'
        project_path = os.path.join(main_dir_path, project_name)

        if not os.path.exists(project_path):
            os.makedirs(project_name)
            print(f"Creating project folder {project_name} in current directory - done.")
        else:
            print(f"Project folder {project_name} exists already.")
        '''
        1. Set bounding box (lower left & upper right corner) for the desired research 
        area in the following way (note the quotes!):
        bbox = ['lat_lowerleft, lng_lowerleft, lat_upperright, lng_upperright']

        NOTE: The class already handles multiple result pages and returns all flickr entries
        '''
        bbox_wildkirchli = ['9.413564,47.282421,9.415497,47.285627']
        bbox_small = ['9.414564,47.284421,9.415497,47.285627']
        bbox_big = ['9.313564,47.282421,9.415497,47.285627']
        bbox_bridge_scotland = ['-6.175232,57.289046,-6.171761,57.290533']

        # flickr_obj = FlickrQuerier(project_name, bbox=bbox_bridge_scotland)
        # flickr_metadata_path = flickr_obj.csv_output_path
        '''
        2. Set desired Cluster algorithm and its parameters
        choice between HDBSCAN and DBSCAN - set input dictionary as seen above
        '''
        test_path = "C:/Users/mhartman/PycharmProjects/MotiveDetection/wildkirchli/metadata_wildkirchli_2019_07_12.csv"

        cluster_obj = ClusterMaster(spatial_clustering_params, data_path=test_path,
                                    spatial_clustering=True)  # flickr_metadata_path
        cluster_df = cluster_obj.df
        unique_cluster_lables = cluster_obj.unique_labels
        '''
        the dataframe will be need to be split into different dataframes because the added features
        will vary according to the clusters!
        '''
        subset_dfs = {}
        for cluster_label in unique_cluster_lables:
            if cluster_label == -1:
                continue
            else:
                boolean_array = (cluster_df.spatial_cluster_label == cluster_label)
                subset_dataframe = cluster_df[boolean_array]
                subset_dfs[f'cluster_{cluster_label}'] = subset_dataframe
        '''
        3. Create image similarity matrix for
        all media objects inside a spatial cluster
        and add the feature matrix to the cluster_dataframe
         -> adding new columns (series in Pandas) with score values for each media object
         -> Possible cv algorithms: SIFT, SURF, ORB
        '''
        index = 1
        cluster_obj_dict = {}
        for subset in subset_dfs:
            print("##" * 30)
            print(f"{index} of {len(subset_dfs.keys())} Processing spatial clustering subset: {subset}")
            cv_obj = ImageSimilarityAnalysis(project_name, image_similarity_params,
                                             subset_dfs[subset])
            subset_dfs[subset] = cv_obj.subset_df
            index += 1
        print("Image analysis for all spatial sub-clusters - done.")
        '''
        4. Create tag vocabulary (bag of words approach) for
        all media object tags inside a spatial cluster
        and add the tf-idf values as features to the cluster_dataframe
        IMPORTANT TO RE-ITERATE OVER THE SUBSET TO GET THE UPDATED REFERENCES FOR 'SUBSET'!
        '''
        for subset in subset_dfs:
            pass


        #############!!!!!DEBRICATED!!!!!########################################################
        # '''
        # 5. second layer Clusering
        # with image similarity and tag frequency input features
        # '''
        # for subset in subset_dfs:
        #     print("##" * 30)
        #     print(f"Multi dimensional clustering of subset: {subset}")
        #     multi_cluster_obj = ClusterMaster(multi_clustering_params, subset_df=subset_dfs[subset],
        #                                 spatial_clustering=False, multi_clustering_inc_coordinates=True, used_lowe_ratio=image_similarity_params['lowe_ratio'])
        #     subset_dfs[subset] = multi_cluster_obj.df  # is named df (not sub_df) in the class to handle both cluster methods

        '''
        5. Network analysis
        Finding and linking scores above a given threshold to clusters
        which shall represent possible motives in the spatial clusters
        '''
        for subset in subset_dfs:
            print("##" * 30)
            print(f"Network analysis of subset: {subset}")
            net_analysis = NetworkAnalyser(subset)
            subset_dfs[subset] = net_analysis.new_dataframe
        '''
        6. Dumping all dataframes to pickle
        in the project folder
        '''
        for k, subset in subset_dfs.items():
            pickle_dataframes(k, subset, multi_clustering_params, image_similarity_params)
            cluster_html_inspect(k, subset, multi_clustering_params, image_similarity_params)

        print("Finished.")