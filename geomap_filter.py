import os
import re
import sys
import json
import pickle
import numpy as np
import pandas as pd
from random import uniform
from statistics import mean

root_path = "C:/Users/<user>/PycharmProjects/ClusterData"

def read_dfs():
    '''
    List all current projects with an index to easy chose
    '''
    select_choice = {}
    index = 1
    for subdirs, dirs, files in os.walk(root_path):
        for directory in dirs:
            select_choice[str(index)] = directory
            index += 1
        break
    print("Projects: ")
    print(json.dumps(select_choice, indent=2))

    choice = input("Enter project name: ")
    project_name = select_choice[choice]
    project_path = os.path.join(root_path, project_name)
    pickles_path = os.path.join(root_path, project_name, 'dataframe_pickles')

    gdf_bucket = {}

    for file in os.listdir(pickles_path):
        # ignore clusters with score 0
        if not file.startswith('0'):
            print(file)
            df = pd.read_pickle(os.path.join(pickles_path, file))

            gdf_bucket[file] = df
    return gdf_bucket, project_path

def gis_export(gdf_bucket, project_path, min_motifs=10, min_authors=1, max_overlap = 0.4):
    '''
    Filter motif clusters according to the hightest scored motif cluster inside a dataframe according to motif size
    and unique authors who contributed to the incorporated posts.
    :param gdf_bucket:
    :param project_path:
    :param min_motifs:
    :param min_authors:
    :return:
    '''
    motif_duplicates = 0
    processed_ids = []
    matches = 0
    PATH = os.path.join(project_path, f'TESTgis_export_min_size_{min_motifs}_min_authors_{min_authors}.csv')
    with open(PATH, 'wt') as f:
        f.write("cluster_nr;motif_label;motif_score;motif_size;unique_authors;pickle_filename;html_filename;x;y\n")
    for filename, gdf in gdf_bucket.items():
        mutli_cluster_labels = gdf['multi_cluster_label'].unique()
        motif_labels = [label for label in mutli_cluster_labels if label != -1]
        '''
        Finding matching html file to the pickle file
        according to the specific 'cluster_XXX' specification
        '''
        cluster_score = re.match(r'[^_]+', filename).group(0)
        cluster_nr = re.search(r'(?<=cluster_)[^_]+', filename).group(0)
        html_path = os.path.join(project_path, 'cluster_hmtl_inspect')
        found = False
        for file in os.listdir(html_path):
            # ignore clusters with score 0
            if not file.startswith('0'):
                cluster_string = f'cluster_{cluster_nr}_'
                try:
                    re.search(cluster_string, file).group(0)
                    try:
                        re.search(str(cluster_score), file).group(0)
                        found = True
                        html_file = os.path.join(html_path, file)
                        break
                    except:
                        continue
                except:
                    continue
        if found:
            for motif_label in motif_labels:
                include_label = False
                x_all = []
                y_all = []
                '''
                identify the number of unique authors for the given motif label
                '''
                motif_images = gdf[gdf['multi_cluster_label'] == motif_label]
                motif_image_ids = motif_images.index
                unique_authors = len(motif_images['user_nsid'].unique())
                #check motif size and unique authors for filtering
                motif_size = len(motif_images.index.values)
                if motif_size >= min_motifs and unique_authors >= min_authors:
                    '''
                    HANDLING DUBLICATE MOTIFS
                    check if most of the images were not already processed before
                    
                    max_overlap defines the max. percenage with already processed ids (otherwise rejected)
                    '''
                    overlap = 0
                    for id in motif_image_ids:
                        if id in processed_ids:
                            overlap += 1
                    if (overlap / motif_size) < max_overlap:
                        include_label = True
                        matches += 1
                        '''
                        calculate mean x and y coordinates
                        '''
                        for i, r in motif_images.iterrows():
                            x_all.append(r['lng'])
                            y_all.append(r['lat'])
                        '''
                        add photo_ids to processed list to handle motif dublicates
                        '''
                        for id in motif_image_ids:
                            processed_ids.append(id)
                    else:
                        motif_duplicates += 1
                        print(f"\rSo far {motif_duplicates} duplicate detected", end='')

                if include_label:
                    x_all_mean = mean(x_all)
                    y_all_mean = mean(y_all)
                    with open(PATH, 'at') as f:
                        f.write(f"{cluster_nr};{motif_label};{cluster_score};{motif_size};{unique_authors};{filename};{html_file};{x_all_mean};{y_all_mean}\n")
        else:
            print("No matching html file found to the pkl file.")
            sys.exit(1)

    print(f"\nOutput file at: {PATH} \nEntries: {matches}")

gdf_bucket, project_path = read_dfs()
gis_export(gdf_bucket, project_path, min_motifs=10, min_authors=10)