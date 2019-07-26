import pandas as pd
import os
import statistics
import matplotlib.pyplot as plt
from collections import Counter

df_path = "C:/Users/mhartman/PycharmProjects/MotiveDetection/wildkirchli/dataframe_pickles/HDBSCAN_2_2_SIFT_0.8_cluster_0_07_24_12.pkl"
html_output = ""
df = pd.read_pickle(df_path)
print("")

def plot_score_dist(dataframe):
    images = len(dataframe.index.values)
    dataframe = dataframe.iloc[:, -images:-1]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    scores = []
    for c, v in dataframe.iteritems():
        for score in v:
            scores.append(score)
    df_median = statistics.median(scores)
    df_average = statistics.mean(scores)
    #create Counter dictionary
    counted_scores = Counter(scores)
    #identify amount of zero values:
    try:
        zeros = counted_scores[0]
    except KeyError:
        print("No zero values")

    score_labels = []
    score_counters = []
    for k, v in counted_scores.items():
        score_labels.append(k)
        score_counters.append(v)

    ax.set_title(f"Total amount of scores: {len(scores)}, median: {df_median}, average: {df_average}, Amount of 0 scores: {zeros}")
    bar = ax.bar(score_labels, score_counters, width=0.75)
    plt.yscale('log')
    plt.show()

def cluster_html_inspect(data, ratio):
    '''
    create an html file that can be insepcted in the browser that links
    images contained in clusters directly to their source path for
    easy inspection

    :param index:
    :param dataframe:
    :return:
    '''
    main_dir_path = os.path.dirname(os.path.realpath(__file__))
    # project_name = input("Enter a project name. Will be integrated in folder and filenames: \n")

    project_name = 'wildkirchli'
    project_path = os.path.join(main_dir_path, project_name)
    #create folder in project_path with the name cluster_hmtl_inspect
    folder_name = 'cluster_hmtl_inspect'
    html_path = os.path.join(project_path, folder_name)
    if not os.path.exists(html_path):
        os.makedirs(html_path)
        print(f"Creating project folder {folder_name} in current directory - done.")
    else:
        print(f"Project folder {folder_name} exists already.")

    file_name = f'cluster_inspect_ORB_entire_wildkirchli_threshold_ratio_{ratio}.html'

    with open(os.path.join(html_path, file_name), 'w') as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<hmtl>\n")
        f.write("<head>\n")
        f.write(f"<title>Multi Cluster</title>\n")
        f.write("</head>\n")
        f.write("<body>\n")
        f.write(f"<h1>Multi Cluster</h1>\n")
        #get the amount of cluster
        if isinstance(data, pd.DataFrame):
            cluster_labels = set(data.loc[:, 'multi_cluster_label'])
            n_clusters = sum([1 for c in cluster_labels if c != -1])

            cluster_dict = {}
            for label in cluster_labels:
                cluster_dict[label] = []

            #append media objects to correct cluster
            for i, row in data.iterrows():
                cluster_label = row['multi_cluster_label']
                cluster_dict[cluster_label].append(i)

        elif isinstance(data, dict):
            cluster_dict = data

        for counter, (k, v) in enumerate(cluster_dict.items()):
            f.write(f'<h2>Cluster {k}</h2>\n')
            f.write(f'<ul>\n')

            for id in v:
                img_path = os.path.join(project_path, f'images_{project_name}', str(id) + '.jpg').replace('\\', '/')
                f.write(f'<li><img src="{img_path}" alt="{id}", height="300", width="300"></li>\n')

            f.write(f'</ul>\n')
        f.write("</body>\n")
        f.write("</html>\n")
    print(f"Html output: {os.path.join(html_path, file_name)}")

def network_analysis(dataframe, threshold=100):
    #the amount of rows determines how many images where compared which is needed for the cropping
    images = len(dataframe.index.values)
    df = dataframe.iloc[:, -images:-1]
    '''
    1. Define edge threshold to connect vertecies
    Calculate the median of similarity scores above ZERO in df
    '''
    scores_above_zero = []

    #iterate over colums which returns colum_label and the content as tuple
    for c, v in df.iteritems():
        for element in v:
            if element != 0:
                scores_above_zero.append(element)
    df_median = statistics.median(scores_above_zero)
    df_average = statistics.mean(scores_above_zero)

    #boolean to see which values are above the threshold, second parameter specifies the value for if the condition is false
    mask_df = df.where(df > threshold, False).where(df <= threshold, True)

    '''
    2. Crawling through df to find similar neighbours
    Iterate over rows, find neighbours/images with a similarity score above the threshold
    '''
    #iterate over the rows which returns the index_name and the row as dictionary with column_labels as keys
    clusters = {}
    #list of indexes which are still not in an cluster and over which will still be iterated
    valid_indexes = list(df.index.values)
    for counter, (i, v) in enumerate(df.iterrows()):
        if i in valid_indexes:
            clusters[counter] = []
            #iterate over the initial row to find neighbours
            for k, score in v.items():
                if k in valid_indexes:
                    if score > threshold:
                        clusters[counter].append(k)
                        # again, remove processed neighbour index
                        valid_indexes.remove(k)
            valid_indexes.remove(i)
            #if at least one neighbour has been found; next iterate over these neighbours
            if len(clusters[counter]) != 0:
                # if at least one neighbour has been found for this index, then the index itself is also added to the list
                clusters[counter].append(i)

                #iterate over previously found neighbours and add any thereof neigbours to the same cluster
                for neighbour in clusters[counter]:
                    #get row from df mit said neighbour id
                    row = df.loc[neighbour, :]
                    series_index = row.index
                    for counter2, score in enumerate(row):
                        if score > threshold and series_index[counter2] in valid_indexes:
                            clusters[counter].append(series_index[counter2])
                            valid_indexes.remove(series_index[counter2])
    return clusters, threshold

plot_score_dist(df)
clusters, threshold, ratio = network_analysis(df, 3.5)
cluster_html_inspect(clusters, ratio)
print("Loaded")