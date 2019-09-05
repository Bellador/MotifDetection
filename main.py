from query_flickr_api import FlickrQuerier
from db_querier import DbQuerier
from clustering import ClusterMaster
from image_feature_detection import ImageSimilarityAnalyser
from network_analysis import NetworkAnalyser
from random import randint
import matplotlib.pyplot as plt
import numpy as np
import statistics
import warnings
import datetime
import os
import sys

def plot_clusters(subset_name, subset):
    unique_labels = subset.multi_cluster_label.unique()
    if len(unique_labels) > 1:
        # Black removed and is used for noise instead.
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        '''
        Sort labels so that the noise points get plotted first 
        and don't cover important clusters
        '''
        for label_counter, (cluster_label, col) in enumerate(zip(sorted(unique_labels, key=lambda x: x), colors)):
            # filter dataframe for rows of given cluster
            label = f"c_{label_counter}"
            if cluster_label == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
                label = 'noise'
            # returns boolean array with true when condition is met
            boolean_array = subset['multi_cluster_label'] == cluster_label
            rows = subset[boolean_array]
            '''
            latitude and longitude
            must most likely be exchanged for some reason
            to match the ArcMap reprentation
            '''
            plt.plot(rows.loc[:, 'lng'], rows.loc[:, 'lat'], 'o', markerfacecolor=tuple(col), markeredgecolor='k',
                     markersize=14, label=label)  # replaced xy with X
        '''
        Adjust plot X and Y extend on
        all points except noise

        query all points that are not noise from dataframe
        '''
        boolean_array_not_noise = subset.multi_cluster_label != -1
        not_noise = subset[boolean_array_not_noise]

        buffer = 0.0005

        xlim_left = not_noise.lng.min() - buffer
        xlim_right = not_noise.lng.max() + buffer
        ylim_bottom = not_noise.lat.min() - buffer
        ylim_top = not_noise.lat.max() + buffer
        plt.xlim(left=xlim_left, right=xlim_right)
        plt.ylim(bottom=ylim_bottom, top=ylim_top)
        plt.title(f"Image similarity {subset_name}")
        plt.legend()
        plt.show()

def check_coordinate_extend(label, subset):
    extend_lat = (subset.lat.max() - subset.lat.min())
    extend_lng = (subset.lng.max() - subset.lng.min())
    '''
    check if cluster is too widely spread which makes the abundance of motive
    images unlikely.
    '''
    if extend_lat > max_lat_extend or extend_lng > max_lng_extend:
        print(f"Too large spatial extend for {label} - Will not be further considered")
        with open(project_path + '/spatial_extend_filter.txt', 'at') as f:
            f.write(f"{label} removed. Extend: lat {extend_lat}, lng {extend_lng}\n")
        return False
    else:
        return True

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
        log.write(f"Cluster: {label}\n")
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

    '''
    Check:
    If size of subcluster is still bigger than the predefined minimum cluster size 
    '''
    if rows_after_filter < spatial_clustering_params['min_cluster_size']:
        return (subset, 'to_delete')
    else:
        return (subset, 'accepted')

def pickle_dataframes(index, dataframe, cluster_params, image_params, cluster_scores):
    try:
        pickle_path = os.path.join(main_dir_path, project_name, 'dataframe_pickles')
        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
        name = '{}_score_{}_{}_{}_{}_{}_{}_{:%m_%d_%H}.pkl'.format(cluster_scores[index]['final_score'], cluster_params['algorithm'], cluster_params['min_cluster_size'], cluster_params['min_samples'],
                                        image_params['algorithm'], image_params['lowe_ratio'], index, datetime.datetime.now())
        dataframe.to_pickle(os.path.join(pickle_path, name))
        print(f"Pickling: {name}")

    except Exception as e:
        print(f"Error: {e} occurred while pickling dataframe {index}")

def cluster_html_inspect(index, dataframe, cluster_params, image_params, cluster_scores):
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

    file_name = '{}_score_{}_{}_{}_{}_{}_{}_{:%m_%d_%H}.html'.format(cluster_scores[index]['final_score'], cluster_params['algorithm'], cluster_params['min_cluster_size'], cluster_params['min_samples'],
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
            f.write(f"<h2>Used parameters:</h2>")
            f.write(f"<h3>Data_source: PostgresqlDB</h3>")
            f.write(f"<h3>     SQL_query: {db_query}</h3>")
            f.write(f"<h3>Filter - Spatial extend: {filter_spatial_extend}, lng {max_lng_extend}, lat {max_lat_extend}; Authors: {filter_authors_switch}; Min_motifs: {min_motives_per_cluster}</h3>")
            f.write(f"<h3>Clustering - Algorithm: {cluster_params['algorithm']}; Min_cluster_size: {cluster_params['min_cluster_size']}, Min_samples: {cluster_params['min_samples']}</h3>")
            f.write(f"<h3>CV - Algorithm: {image_params['algorithm']}; Lowe_ration: {image_params['lowe_ratio']}</h3>")
            f.write(f"<h3>Motif network analysis - Threshold: {image_params['network_threshold']}</h3>")
            f.write(f"<h3>--------------------------------------------------------------------------------</h3>")
            f.write(f"<h2>Cluster score: </h2>")
            f.write(f"<h3>overall_mean_nr_subclusters: {round(cluster_scores['overall_mean_nr_subclusters'], 3)}</h3>")
            f.write(f"<h3>overall_mean_subcluster_size: {round(cluster_scores['overall_mean_subcluster_size'], 3)}</h3>")
            f.write(f"<h3>overall_mean_unique_authors: {round(cluster_scores['overall_mean_unique_authors'], 3)}</h3>")
            f.write(f"<h3>      nr_subclusters: {cluster_scores[index]['nr_subclusters']}</h3>")
            f.write(f"<h3>      mean_subcluster_size: {cluster_scores[index]['mean_subcluster_size']}</h3>")
            # f.write(f"<h3>      mean_timespan: {cluster_scores[index]['mean_timespan']}</h3>")
            f.write(f"<h3>      mean_unique_authors: {cluster_scores[index]['mean_unique_authors']}</h3>")
            f.write(f"<h3>      final_score: {cluster_scores[index]['final_score']}</h3>")
            f.write(f"<h3>--------------------------------------------------------------------------------</h3>")
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
            f.write(f"<h2>Used parameters: </h2>")
            f.write(f"<h3>Data_source: FlickrAPI</h3>")
            f.write(f"<h3>  Bounding box: {flickr_bbox}</h3>")
            f.write(f"<h3>Filter - Spatial extend: {filter_spatial_extend}, lng {max_lng_extend}, lat {max_lat_extend}; Authors: {filter_authors_switch}; Min_motifs: {min_motives_per_cluster}</h3>")
            f.write(f"<h3>Clustering - Algorithm: {cluster_params['algorithm']}; Min_cluster_size: {cluster_params['min_cluster_size']}, Min_samples: {cluster_params['min_samples']}</h3>")
            f.write(f"<h3>CV - Algorithm: {image_params['algorithm']}; Lowe_ration: {image_params['lowe_ratio']}</h3>")
            f.write(f"<h3>Motif network analysis - Threshold: {image_params['network_threshold']}</h3>")
            f.write(f"<h3>--------------------------------------------------------------------------------</h3>")
            f.write(f"<h2>Cluster score: </h2>")
            f.write(f"<h3>overall_mean_nr_subclusters: {round(cluster_scores['overall_mean_nr_subclusters'], 3)}</h3>")
            f.write(f"<h3>overall_mean_subcluster_size: {round(cluster_scores['overall_mean_subcluster_size'], 3)}</h3>")
            f.write(f"<h3>overall_mean_unique_authors: {round(cluster_scores['overall_mean_unique_authors'], 3)}</h3>")
            f.write(f"<h3>      nr_subclusters: {cluster_scores[index]['nr_subclusters']}</h3>")
            f.write(f"<h3>      mean_subcluster_size: {cluster_scores[index]['mean_subcluster_size']}</h3>")
            # f.write(f"<h3>      mean_timespan: {cluster_scores[index]['mean_timespan']}</h3>")
            f.write(f"<h3>      mean_unique_authors: {cluster_scores[index]['mean_unique_authors']}</h3>")
            f.write(f"<h3>      final_score: {cluster_scores[index]['final_score']}</h3>")
            f.write(f"<h3>--------------------------------------------------------------------------------</h3>")
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

                for id in v:
                    img_path = os.path.join(project_path, f'images_{project_name}', str(id) + '.jpg').replace('\\', '/')
                    f.write(f'<li><img src="{img_path}" alt="{id}", height="300", width="300"><h3>{id}</h3></li>\n')

                f.write(f'</ul>\n')
            f.write("</body>\n")
            f.write("</html>\n")

    print(f"Created inspection html file {file_name} in folder {folder_name}")

def calc_cluster_scores(dataset):
    '''
    Use these scores to evaluete and sort the final cluster outputs
    The scores will be added to the html and pickle dataframe dump files

    :param dataset:
    :return:
    '''
    cluster_scores = {}
    for sb_name, sb_data in dataset.items():
        final_score = None
        #nr of subclusters
        cluster_labels = sb_data['multi_cluster_label'].unique()
        cluster_labels_no_noise = [label for label in cluster_labels if label != -1]
        nr_subclusters = len(cluster_labels_no_noise)
        if nr_subclusters != 0:
            #--------------------------------------------------------------------------
            #calculate average cluster size
            sizes = []
            for label in cluster_labels_no_noise:
                labels = sb_data[sb_data['multi_cluster_label'] == label]
                sizes.append(len(labels.index.values))
            mean_subcluster_size = statistics.mean(sizes)

            #---------------------------------------------------------------------------
            #calculate the subcluster timespans and take the mean

            # timespans = []
            # for label in cluster_labels_no_noise:
            #     timestamps = sb_data.loc[sb_data['multi_cluster_label'] == label]['date_uploaded']
            #     sordted_timestamps = list(timestamps.sort_values(ascending=False))
            #     try:
            #         timespans.append(int(sordted_timestamps[0]) - int(sordted_timestamps[-1]))
            #     except Exception as e:
            #         mean_timespan = None
            # mean_timespan = statistics.mean(timespans)
            #---------------------------------------------------------------------------
            #calculate number of unique authors per subcluster and take the mean
            count_unique_auhtors = []
            for label in cluster_labels_no_noise:
                user_nsids = sb_data.loc[sb_data['multi_cluster_label'] == label]['user_nsid']
                count_unique_auhtors.append(len(user_nsids.unique()))
            mean_unique_authors = statistics.mean(count_unique_auhtors)
            # ---------------------------------------------------------------------------
            # Add values to dictionary
            cluster_scores[sb_name] = {'nr_subclusters': nr_subclusters,
                                       'mean_subcluster_size': mean_subcluster_size,
                                       # 'mean_timespan': mean_timespan,
                                       'mean_unique_authors': mean_unique_authors,
                                       'final_score': final_score}
        else:
            final_score = 0
            cluster_scores[sb_name] = {'nr_subclusters': 0,
                                       'mean_subcluster_size': 0,
                                       # 'mean_timespan': 0,
                                       'mean_unique_authors': 0,
                                       'final_score': final_score}
    # ---------------------------------------------------------------------------
    # Calculate the final subcluster scores based on the subscore of each (their range) relativ to one another
    overall_mean_nr_subclusters = []
    overall_mean_subcluster_size = []
    # overall_mean_timespan = []
    overall_mean_unique_authors = []

    for k, v in cluster_scores.items():
        #Only consider HDBSCAN clusters that consist of motifs!
        if v['final_score'] == None:
            overall_mean_nr_subclusters.append(v['nr_subclusters'])
            overall_mean_subcluster_size.append(v['mean_subcluster_size'])
            # overall_mean_timespan.append(v['mean_timespan'])
            overall_mean_unique_authors.append(v['mean_unique_authors'])

    overall_mean_nr_subclusters = round(statistics.mean(overall_mean_nr_subclusters),2)
    overall_mean_subcluster_size = round(statistics.mean(overall_mean_subcluster_size),2)
    # overall_mean_timespan = statistics.mean(overall_mean_timespan)
    overall_mean_unique_authors = round(statistics.mean(overall_mean_unique_authors),2)

    if overall_mean_nr_subclusters == 0:
        overall_mean_nr_subclusters = 1
    if overall_mean_subcluster_size == 0:
        overall_mean_subcluster_size = 1
    # if overall_mean_timespan == 0:
    #     overall_mean_timespan = 1
    if overall_mean_unique_authors == 0:
        overall_mean_unique_authors = 1

    for k, v in cluster_scores.items():
        if v['final_score'] != 0:
            try:
                v['final_score'] = round(((v['nr_subclusters']/overall_mean_nr_subclusters) + (v['mean_subcluster_size']/overall_mean_subcluster_size) \
                                   + (v['mean_unique_authors']/overall_mean_unique_authors)*100), 2) #(1 / (v['mean_timespan']/overall_mean_timespan)) +
            except Exception as e:
                print(f"Cluster score error: {e}")
                print(f"Final score set to 0")
                v['final_score'] = 0

    cluster_scores['overall_mean_nr_subclusters'] = overall_mean_nr_subclusters
    cluster_scores['overall_mean_subcluster_size'] = overall_mean_subcluster_size
    cluster_scores['overall_mean_unique_authors'] = overall_mean_unique_authors

    return cluster_scores

if __name__ == '__main__':
    '''
    Database queries:
    
    '''
    natura2000_query = """
        SELECT x.photo_id, x.user_nsid, x.download_url, x.date_uploaded ,x.lat, x.lng
        FROM data_100m as x
        JOIN natura2000_projected as y
        ON ST_WITHIN(x.geometry, y.geom)
        WHERE x.georeferenced = 1
        """

    switzerland_query = """
        SELECT x.photo_id, x.user_nsid, x.download_url, x.date_uploaded ,x.lat, x.lng
        FROM data_100m as x
        JOIN switzerland as y
        ON ST_WITHIN(x.geometry, y.geom)
        WHERE x.georeferenced = 1
        """

    switzerland_old_query = """
        SELECT x.photo_id, x.user_nsid, x.download_url, x.date_uploaded ,x.lat, x.lng
        FROM data_100m as x
        JOIN switzerland as y
        ON ST_WITHIN(x.geometry, y.geom)
        WHERE x.georeferenced = 1
        AND new_data = 0
        """

    wildkirchli_query = """
        SELECT x.photo_id, x.user_nsid, x.download_url, x.date_uploaded ,x.lat, x.lng
        FROM data_100m as x
        JOIN wildkirchli as y
        ON ST_WITHIN(x.geometry, y.geom)
        WHERE x.georeferenced = 1
        """

    loewendenkmal_query = """
        SELECT x.photo_id, x.user_nsid, x.download_url, x.date_uploaded ,x.lat, x.lng
        FROM data_100m as x
        JOIN loewendenkmal as y
        ON ST_WITHIN(x.geometry, y.geom)
        WHERE x.georeferenced = 1
        """
    # AND x.date_uploaded >= 1262304000
    # AND x.date_uploaded <= 1420070400
    #unixtimestamps for 2010 - 2015
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
        'min_cluster_size': 10,
        'min_samples': 10,
        'cluster_selection_method': 'leaf' #default 'eom'
    }
    cluster_params_HDBSCAN_multi = {
        'algorithm': 'HDBSCAN',
        'min_cluster_size': 2,
        'min_samples': 2,
        'cluster_selection_method': 'leaf'  # default 'eom'
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
        'lowe_ratio': 0.7,
        'network_threshold': 100
    }
    SURF_params = {
        'algorithm': 'SURF',
        'lowe_ratio': 0.7,
        'network_threshold': 100
    }
    ORB_params = {
        'algorithm': 'ORB',
        'lowe_ratio': 0.7,
        'network_threshold': 100
    }
    ##############################################################
    ####################ADJUST#PARAMETERS#########################
    ##############################################################
    data_source = 1 #1 = PostGIS database; 2 = Flickr API
    db_query = natura2000_query
    flickr_bbox = bbox_small
    filter_authors_switch = False
    filter_spatial_extend = False
    max_lng_extend = 0.05 #change / neglect when running on Cluster
    max_lat_extend = 0.05 #change / neglect when running on Cluster
    spatial_clustering_params = cluster_params_HDBSCAN_spatial
    # multi_clustering_params = cluster_params_HDBSCAN_multi
    image_similarity_params = SIFT_params
    min_motives_per_cluster = None #None if this step shall be skipped
    ################################################################
    ################################################################
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main_dir_path = os.path.dirname(os.path.realpath(__file__))
        # project_name = input("Enter a project name. Will be integrated in folder and filenames: \n")
        project_name = 'eu_natura'
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
            to_dbquery = True
            data_path = None
            #check if metadatafile already exists
            # r=root, d=directories, f = files
            try:
                for r, d, f in os.walk(project_path):
                    for file in f:
                        if file.endswith('.csv'):
                            print("Metadata for this db query already exists.")
                            print("Skipping invocation of DbQuerier.")
                            print("--"*30)
                            to_dbquery = False
                            data_path = os.path.join(r, file)
            except Exception as e:
                print(f"Error: {e}")

            if to_dbquery:
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
        cluster_obj = ClusterMaster(data_source, spatial_clustering_params, data_path=data_path, spatial_clustering=True)
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
        Checking spatial extend of subclusters
        to far spread clusters do not support the existance of motives
        and are therefor not further considered
        '''
        if filter_spatial_extend:
            #keys that need to be deleted after the iteration over the dictionary
            del_keys = []
            print('**' * 30)
            print("Checking subcluster spatial extend...")
            for label, subset in subset_dfs.items():
                check_ok = check_coordinate_extend(label, subset)
                # True means the spatial extend is in the boundary limits
                if check_ok:
                    continue
                # False, spatial extend of subclaster too large
                elif not check_ok:
                    del_keys.append(label)
            #delete filtered subcluster keys
            print(f"{len(del_keys)} clusters have a too large spatial extend and will be removed...")
            for key in del_keys:
                del subset_dfs[key]
            print('**' * 30)
        '''
        Conditionally apply
        Author filtering
        '''
        if filter_authors_switch:
            del_keys = []
            for label, subset in subset_dfs.items():
                print("Filtering authors...")
                output = filter_authors(label, subset)
                filtered_subset = output[0]
                conditional = output[1]
                if conditional == 'to_delete':
                    del_keys.append(label)
                elif conditional == 'accepted':
                    subset_dfs[label] = filtered_subset
            # delete filtered subcluster keys
            print(f"{len(del_keys)} clusters are below min_cluster_size and will be removed...")
            for key in del_keys:
                del subset_dfs[key]
            print('**' * 30)
        print(f"{len(subset_dfs.keys())} clusters left after filtering process")
        print('**' * 30)
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
            net_analysis = NetworkAnalyser(label, subset, threshold=image_similarity_params['network_threshold'])
            subset_dfs[label] = net_analysis.new_dataframe
        '''
        5.1
        Check the final sub-cluster (exc. Noise) sizes to be above the defined
        min_motives_clusters value
        if None no filter shall be applied
        '''
        print("##" * 30)
        print(f"Checking for motive clusters with minimum size of {min_motives_per_cluster}...")
        del_keys = []
        final_len_before = len(subset_dfs.keys())
        if min_motives_per_cluster != None:
            for cluster_name, subset in subset_dfs.items():
                exclude = True
                #find unique cluster labels to filter out noise clusters
                unique_cluster_labels = subset.multi_cluster_label.unique()
                if len(unique_cluster_labels) != 0:
                    for label in unique_cluster_labels:
                        if label != -1:
                            boolean_array = (subset['multi_cluster_label'] == label)
                            len_labels = len(subset[boolean_array])
                            if len_labels >= min_motives_per_cluster:
                                exclude = False
                if exclude:
                    del_keys.append(cluster_name)
            #delete clusters below the minimum size
            for key in del_keys:
                del subset_dfs[key]
            final_len_after = len(subset_dfs.keys())
            print(f"Removed {final_len_before-final_len_after} of {final_len_before} sub-clusters")
            print(f"Remaining: {final_len_after}")
        '''
        6. Cluster Scores
        calculate and evaluate clusters based on certain parameters related to the included sub-clusters:
            - amount of authors
            - timespan between oldest and newest media object (temporarily neglected)
            - nr. of sub-clusters
            - amount of clsutered images compared to noise media objects
            ...
        '''
        #dictionary with cluster id key and its score as value
        cluster_scores = calc_cluster_scores(subset_dfs)
        '''
        6. Dumping all dataframes to pickle
        in the project folder
        '''
        print("##" * 30)
        print("Create output file(s)")
        '''
        Sorting of Output:
        According to the preceeding calculated cluster scores
        '''
        for k, subset in subset_dfs.items():
            pickle_dataframes(k, subset, spatial_clustering_params, image_similarity_params, cluster_scores)
            print("--" * 30)
            cluster_html_inspect(k, subset, spatial_clustering_params, image_similarity_params, cluster_scores)
            print("--" * 30)
        '''
        Plot
        resulting image motive clusters
        '''
        #for label, subset in subset_dfs.items():
        #    plot_clusters(label, subset)
        print("Finished.")
