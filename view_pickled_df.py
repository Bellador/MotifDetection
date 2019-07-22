import pandas as pd
import numpy as np
import os

df_path = "C:/Users/mhartman/PycharmProjects/MotiveDetection/wildkirchli/dataframe_pickles/HDBSCAN_SIFT_LOWE_80_cluster_1_07_18_10.pkl"
html_output = ""
df = pd.read_pickle(df_path)

def cluster_html_inspect(dataframe):
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

    file_name = f'cluster_inspect.html'

    with open(os.path.join(html_path, file_name), 'w') as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<hmtl>\n")
        f.write("<head>\n")
        f.write(f"<title>Multi Cluster</title>\n")
        f.write("</head>\n")
        f.write("<body>\n")
        f.write(f"<h1>Multi Cluster</h1>\n")
        #get the amount of cluster
        cluster_labels = set(dataframe.loc[:, 'multi_cluster_label'])
        n_clusters = sum([1 for c in cluster_labels if c != -1])

        cluster_dict = {}
        for label in cluster_labels:
            cluster_dict[label] = []

        #append media objects to correct cluster
        for i, row in dataframe.iterrows():
            cluster_label = row['multi_cluster_label']
            cluster_dict[cluster_label].append(i)

        for counter, (k, v) in enumerate(cluster_dict.items()):
            f.write(f'<h2>Cluster {k}</h2>\n')
            f.write(f'<ul>\n')

            for id in v:
                img_path = os.path.join(project_path, f'images_{project_name}', str(id) + '.jpg').replace('\\', '/')
                f.write(f'<li><img src="{img_path}" alt="{id}", height="300", width="300"></li>\n')

            f.write(f'</ul>\n')
        f.write("</body>\n")
        f.write("</html>\n")

cluster_html_inspect(df)

print("Loaded")