from query_flickr_api import FlickrQuerier
from db_querier import DbQuerier
from clustering import ClusterMaster
from image_feature_detection import ImageSimilarityAnalyser
from network_analysis import NetworkAnalyser
from random import randint
import warnings
import datetime
import os
import sys

def filter_authors(label, subset):
    '''
    Retaining only one media object per unique author and SUBCLUSTER

    1. iterate over subclusters
    '''
    rows_before_filter = len(subset.index.values)
    # radomly choose to either keep the first or last record
    rand = randint(0, 1)
    if rand == 0:
        to_keep = 'first'
    elif rand == 1:
        to_keep = 'last'
    # postgres column_name = user_nsid
    if data_source == 1:
        subset = subset.drop_duplicates(subset='user_nsid', keep=to_keep)
    # flickrAPI colum_name = author_id
    elif data_source == 2:
        subset = subset.drop_duplicates(subset='author_id', keep=to_keep)

    rows_after_filter = len(subset.index.values)

    with open(project_path + '/author_filter_log.txt', 'at') as log:
        log.write("**" * 30 + '\n')
        log.write(f"Entries before: {rows_before_filter}\n")
        log.write(f"Entries after: {rows_after_filter}\n")
        log.write(f"Difference: {rows_before_filter - rows_after_filter}; -{round((rows_before_filter - rows_after_filter) / rows_before_filter * 100, 1)}%\n")
        log.write("**" * 30 + '\n')

    print("**" * 30)
    print(f"Filter result Subset {label}:")
    print(f"Entries before: {rows_before_filter}")
    print(f"Entries after: {rows_after_filter}")
    print(f"Difference: {rows_before_filter - rows_after_filter}; -{round((rows_before_filter - rows_after_filter) / rows_before_filter * 100, 1)}%")
    print("**" * 30)
    return subset

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
    #Database
    if data_source == 1:
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
                image_url = row['download_url']
                cluster_label = row['multi_cluster_label']
                cluster_dict[cluster_label].append((i, image_url))

            for counter, (k, v) in enumerate(cluster_dict.items()):
                f.write(f'<h2>Cluster {k}</h2>\n')
                f.write(f'<ul>\n')

                for tuple_ in v:
                    id = tuple_[0]
                    img_path = tuple_[1]
                    f.write(f'<li><img src="{img_path}" alt="{id}", height="300", width="300"><h3>{id}</h3></li>\n')

                f.write(f'</ul>\n')
            f.write("</body>\n")
            f.write("</html>\n")
    #FlickrAPI
    elif data_source == 2:
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
    Database queries:
    
    '''
    eu_protected_sites = """
    SELECT x.photo_id, x.user_nsid, x.download_url, x.lat, x.lng
    FROM data_100m as x
    JOIN switzerland as y
    ON ST_WITHIN(x.geometry, y.geom)
    WHERE x.georeferenced = 1
    AND x.date_uploaded >= 1388534400
    AND x.date_uploaded <= 1420070400"""
    #unixtimestamps for 2014 - 2015
    '''
    Flickr API: Set bounding box (lower left & upper right corner) for the desired research 
    area in the following way (note the quotes!):
    bbox = ['lat_lowerleft, lng_lowerleft, lat_upperright, lng_upperright']
    NOTE: The class already handles multiple result pages and returns all flickr entries
    '''
    bbox_wildkirchli = ['9.413564,47.282421,9.415497,47.285627']
    bbox_small = ['9.414564,47.284421,9.415497,47.285627']
    bbox_big = ['9.313564,47.282421,9.415497,47.285627']
    bbox_bridge_scotland = ['-6.175232,57.289046,-6.171761,57.290533']
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
    ##############################################################
    ####################ADJUST#PARAMETERS#########################
    ##############################################################
    '''
    Define data source:
    1. PostGIS database
    2. Flickr API
    '''
    data_source = 2
    flickr_bbox = bbox_wildkirchli
    db_query = eu_protected_sites
    filter_authors_switch = True
    spatial_clustering_params = cluster_params_HDBSCAN_spatial
    # multi_clustering_params = cluster_params_HDBSCAN_multi
    image_similarity_params = SIFT_params
    ################################################################
    ################################################################

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main_dir_path = os.path.dirname(os.path.realpath(__file__))
        project_name = input("Enter a project name. Will be integrated in folder and filenames: \n")

        # project_name = 'wildkirchli' #'db_test_switzerland'
        project_path = os.path.join(main_dir_path, project_name)

        if not os.path.exists(project_path):
            os.makedirs(project_name)
            print(f"Creating project folder {project_name} in current directory - done.")
        else:
            print(f"Project folder {project_name} exists already.")
        '''
        1. Loading data
        check from which source data will be loaded
        '''
        if data_source == 1:
            print("About to import data from database...")
            db_obj = DbQuerier(db_query, project_name)
            data_path = db_obj.csv_output_path
        elif data_source == 2:
            print("About to import data from Flickr API...")
            flickr_obj = FlickrQuerier(project_name, bbox=flickr_bbox)
            data_path = flickr_obj.csv_output_path
        else:
            print("Invalid data source")
            sys.exit(1)
        '''
        2. Set desired Cluster algorithm and its parameters
        choice between HDBSCAN and DBSCAN - set input dictionary as seen above
        '''
        # test_path = "C:/Users/mhartman/PycharmProjects/MotiveDetection/bridge/metadata_bridge_2019_07_23.csv"
        # data_path = "C:/Users/mhartman/PycharmProjects/MotiveDetection/wildkirchli/metadata_wildkirchli_2019_07_12.csv"
        cluster_obj = ClusterMaster(data_source, spatial_clustering_params, data_path=data_path,
                                    spatial_clustering=True, handle_authors=True)
        original_db_size = cluster_obj.original_df_size
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
        Conditionally apply
        Author filtering
        '''
        if filter_authors_switch:
            for label, subset in subset_dfs.items():
                print("Filtering authors...")
                subset_dfs[label] = filter_authors(label, subset)
        '''
        3. Create image similarity matrix for
        all media objects inside a spatial cluster
        and add the feature matrix to the cluster_dataframe
         -> adding new columns (series in Pandas) with score values for each media object
         -> Possible cv algorithms: SIFT, SURF, ORB
        '''
        cluster_obj_dict = {}
        for index, (label, subset) in enumerate(subset_dfs.items(), 1):
            print("##" * 30)
            print(f"{index} of {len(subset_dfs.keys())} Processing spatial clustering subset: {label}")
            cv_obj = ImageSimilarityAnalyser(project_name, data_source, image_similarity_params,
                                             subset)
            subset_dfs[label] = cv_obj.subset_df

        print("Image analysis for all spatial sub-clusters - done.")
        '''
        4. Create tag vocabulary (bag of words approach) for
        all media object tags inside a spatial cluster
        and add the tf-idf values as features to the cluster_dataframe
        IMPORTANT TO RE-ITERATE OVER THE SUBSET TO GET THE UPDATED REFERENCES FOR 'SUBSET'!
        '''
        for label, subset in subset_dfs.items():
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
        print("##" * 30)
        for label, subset in subset_dfs.items():
            print(f"\rNetwork analysis of subset: {label}", end='')
            net_analysis = NetworkAnalyser(label, subset)
            subset_dfs[label] = net_analysis.new_dataframe
        '''
        6. Dumping all dataframes to pickle
        in the project folder
        '''
        print("##" * 30)
        print("Create output files")
        for k, subset in subset_dfs.items():
            pickle_dataframes(k, subset, spatial_clustering_params, image_similarity_params)
            print("--" * 30)
            cluster_html_inspect(k, subset, spatial_clustering_params, image_similarity_params)
            print("--" * 30)

        print("Finished.")