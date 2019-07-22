import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import math
import re
import time


class ImageSimilarityAnalysis:
    score_same_image = 0
    # score_same_image = 1
    core_df_range = 17

    def __init__(self, project_name, algorithm_params, subset_df, pickle=False):
        start = time.time()
        self.project_name = project_name
        self.algorithm_params = algorithm_params
        self.subset_df = subset_df
        self.project_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.project_name)
        self.images_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.project_name, f'images_{self.project_name}')
        self.algorithm = self.algorithm_params['algorithm']
        self.lowe_ratio = self.algorithm_params['lowe_ratio']
        self.score_multiplier = self.algorithm_params['score_multiplier']
        self.top_matches_for_score = self.algorithm_params['top_matches_for_score']


        print("--" * 30)
        print("Initialising Computer Vision with ImageSimilarityAnalysis Class")
        self.image_objects, self.feature_dict, self.nr_files = self.file_loader()
        print("--" * 30)
        print("Load images - done.")
        self.compute_keypoints()
        print("--" * 30)
        print("Compute image keypoints - done.")
        self.df, self.df_similarity = self.match_keypoints(top_matches=20, lowe_ratio=self.lowe_ratio, pickle_similarity_matrix=pickle)
        print("--" * 30)
        print("Compute image similarity dataframe - done.")


        end = time.time()
        print(f"Duration: {end - start} seconds")

        with open(path_performance_log, 'a') as log:
            log.write(f"{self.algorithm}, processed files: {self.nr_files}, duration: {end-start}\n")

        # self.visualise_matches('img9', 'img10', top_matches=20)
        # self.visualise_matches('img8', 'img9', top_matches=20)
        # self.visualise_matches('img8', 'img10', top_matches=20)
        print("Adding new features to dataframe")
        self.add_features()
        # self.plot_results(top_comparisons=20, top_matches=20)

        print("--" * 30)
        print("--" * 30)
        print("ImageSimilarityAnalysis Class - done")

    def file_loader(self):
        #load image as grayscale since following 3 algoithms ignore RGB information
        image_objects = {}
        feature_dict = {}
        filename_tracker = {}

        files = [os.path.join(self.images_path, file) for file in os.listdir(self.images_path) if os.path.isfile(os.path.join(self.images_path, file))]
        '''
        load images
        ONLY of the specific subset!
        '''
        needed_ids = self.subset_df.index.values
        print(f"Number of images to process: {len(needed_ids)}")
        for index, file in enumerate(files):
            pattern = r"([\d]*)\.jpg$"
            file_id = int(re.search(pattern, file).group(1))
            if file_id in needed_ids:
                image_objects[file_id] = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                feature_dict[file_id] = {}

        '''
        creating already the required keys with empty dict's in the feature dictionary
        which will store the corresponding keypoints and descriptors
        '''
        # print(f"{index+1} images read.")
        return image_objects, feature_dict, len(files)

    def compute_keypoints(self):
        '''
        compute the keypoints and the corresponding descriptors
        which allow the keypoints to be functional even with 
        image rotation and distortion

        Algorithms: SIFT, SURF, ORB
        '''
        if self.algorithm == 'SIFT':
            alg = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.04, edgeThreshold=10)

        elif self.algorithm == 'SURF':
            alg = cv2.xfeatures2d.SURF_create()

        elif self.algorithm == 'ORB':
            alg = cv2.ORB_create(nfeatures=1500)

        print(f"Using algorithm {self.algorithm}")

        for obj in self.image_objects:
            #the None defines if a mask shall be used or not
            keypoints, descriptors = alg.detectAndCompute(self.image_objects[obj], None)
            self.feature_dict[obj]['kp'] = keypoints
            self.feature_dict[obj]['ds'] = descriptors


    def match_keypoints(self, top_matches=20, lowe_ratio=0.8, pickle_similarity_matrix=True):
        '''
        Brute Force
        Matching between the different images
        - Create pandas dataframe to store the matching output
        of every image (n) with the set in a 2D matrix of n^2 entries
        '''
        df_pickle_path = "C:/Users/mhartman/Documents/100mDataset/"
        top_considered_matches = 20
        #one to store the matches, the other for the later computed similarity scores
        df = pd.DataFrame(columns=self.image_objects.keys(), index=self.image_objects.keys())
        df_similiarity = pd.DataFrame(columns=self.image_objects.keys(), index=self.image_objects.keys())
         # = pd.DataFrame(columns=image_objects.keys(), index=image_objects.keys())
        #normalising function dependant on used algorithm - NORM_HAMMING good for orb
        bf = cv2.BFMatcher() #cv2.NORM_L1, crossCheck=True
        #store already sorted match results in corresponding matrix cells

        indexes = df.index.values
        columns = df.columns.values
        '''
        here reduce the redundancy
        by avoiding to check the same comparisons twice                
        '''
        # [df.set_value(index, column, bf.knnMatch(self.feature_dict[index]['ds'], self.feature_dict[column]['ds']))
        #         for check1, index in enumerate(indexes)
        #             for check2, column in enumerate(columns) if check2 >= check1 and check2 != check1]


        for check1, index in enumerate(indexes):
            for check2, column in enumerate(columns):
                if check2 >= check1 and check2 != check1:
                    df.set_value(index, column, bf.knnMatch(self.feature_dict[index]['ds'], self.feature_dict[column]['ds'], k=2))

        print("Populated dataframe with image matches - done.")
        recorded_match_lengths = []
        #iterate over dataframe to calculate similarity score and populate second df
        for row in indexes: #better use .iteritems()
            for col in columns:
                # distances = []
                similar_regions = []
                matches = df.loc[row, col]
                if isinstance(matches, (list,)): #would work too: type(matches) == list
                    if len(matches) == 0:
                        df_similiarity.set_value(row, col, 0)
                    else:
                        for m, n in matches:
                            if m.distance < self.lowe_ratio * n.distance:
                                similar_regions.append([m])
                        # similar_regions = [region for region in sorted([match.distance for match in matches], key=lambda x: x)[:self.top_matches_for_score] if region <= threshold]
                        '''
                        Following:
                        Score based on similar regions compared to match length:
                        range of match length was witnessed to be between 150 and 1300 (~factor 10)
                        thats why new approach
                        '''
                        # score = len(similar_regions) / len(matches)
                        '''
                        New:
                        Score based on similar regions among the best, top 100 matches
                         between two images
                        '''
                        score = len(similar_regions) #/ self.top_matches_for_score

                        df_similiarity.set_value(row, col, score)

                    recorded_match_lengths.append(len(matches))

                elif math.isnan(matches):
                    #has to be zero not nan for the clustering to work
                    df_similiarity.set_value(row, col, 0)

        recorded_match_lengths = sorted(recorded_match_lengths, key=lambda x: x)
        '''
        fill in missing fields that were previously skipped due to computational reasons caused by same image comparison 
        -> Mirror dataframe along the diagonal
        All values are needed to form the individual image similarity features for each media object!
        '''
        [df_similiarity.set_value(index, column, df_similiarity.loc[column, index])
             for check1, index in enumerate(indexes)
                for check2, column in enumerate(columns) if check2 < check1 and check1 != 0]
        '''
        Fill in similarity score of 0
        for all comparisons between the same images -> the diagonal
        '''
        [df_similiarity.set_value(index, column, ImageSimilarityAnalysis.score_same_image)
             for check1, index in enumerate(indexes)
                for check2, column in enumerate(columns) if check2 == check1]

        if pickle_similarity_matrix:
            print("Pickling similarity dataframe...")
            if re.search(r"motive", self.workpath):
                df_similiarity.to_pickle("{}similarity_matrix_motive_{}_{:%Y_%m_%d}_{}.pkl".format(self.project_path, self.threshold, datetime.datetime.now(), self.algorithm))

            elif re.search(r"noise", self.workpath):
                df_similiarity.to_pickle("{}similarity_matrix_noise_{}_{:%Y_%m_%d}_{}.pkl".format(self.project_path, self.threshold, datetime.datetime.now(), self.algorithm))

        return df, df_similiarity

    def visualise_matches(self, img1, img2, top_matches=20):
        img_1_object = self.image_objects[img1]
        img_2_object = self.image_objects[img2]
        kp_1 = self.feature_dict[img1]['kp']
        kp_2 = self.feature_dict[img2]['kp']
        matches = self.df.loc[img1, img2][:top_matches]

        result = cv2.drawMatches(img_1_object, kp_1, img_2_object, kp_2, matches, None, flags=2)
        cv2.imshow("Image", result)
        cv2.waitKey(0)
        cv2.destroyWindow("Image")

    def plot_results(self, top_comparisons=20, top_matches=20, score_plot=False, plot=True, barchart=False):

        font = {'size': 5}
        plt.rc('font', **font)

        print("Plotting...")

        # if re.search(r'motive', self.workpath):
        #     image_type = 'motive'
        # elif re.search(r'noise', self.workpath):
        #     image_type = 'noise'

        image_type = self.project_name

        columns = self.df.columns.values
        indexes = self.df.index.values
        distance_dict = {}
        # Create plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        if plot or barchart:
            for check1, index in enumerate(indexes):
                distance_dict[index] = {}
                for check2, column in enumerate(columns):
                    '''
                    here try to reduce the redundancy
                    by avoiding to check the same comparisons twice                
                    '''
                    if check2 >= check1 and check2 != check1:
                        matches = self.df.loc[index, column]
                        distances = [match.distance for match in matches][:top_matches]
                        distance_dict[index][column] = distances

        if plot:
            distance_of_first_kp = []
            for item in distance_dict:
                for sub_item in distance_dict[item]:
                    tot_distances = sum(distance_dict[item][sub_item][:top_matches])
                    label = f"{item}_{sub_item}"
                    distance_of_first_kp.append((tot_distances, label))
            distance_of_first_kp = sorted(distance_of_first_kp, key=lambda x: x[0])

            top_labels = {}
            for item in distance_of_first_kp[:top_comparisons]:
                #important: here the distances array is again assiged instead of the total distance
                split = item[1].split('_')
                index = int(split[0])
                column = int(split[1])
                top_labels[item[1]] = distance_dict[index][column]

            for item in distance_dict:
                for sub_item in distance_dict[item]:
                    label = f"{item}_{sub_item}"
                    try:
                        distances = top_labels[label]
                        ind = list(range(len(top_labels.keys())))
                        ax.plot(ind, distances, alpha=0.5, label=label)
                        ax.text(ind[-int(len(ind)/2)], distances[-int(len(distances)/2)], f'{label}')
                    except KeyError:
                        continue

            ax.set_ylabel(f'keypoint distance')
            ax.set_xlabel(f'top 20 keypoints per comparison')
            ax.set_title(f'{self.algorithm}: top 20 {image_type} image comparisons distance profiles')
            ax.set_xticks(ind[:top_comparisons])
            ax.set_xticklabels(ind[:top_comparisons])

        elif barchart:
            #list of tuples: tot_distances and corresponding labels
            data = []
            counter = 0
            for item in distance_dict:
                for sub_item in distance_dict[item]:
                    counter += 1
                    tot_distances = sum(distance_dict[item][sub_item][:top_matches])
                    labels = f"{item}_{sub_item}"
                    data.append((tot_distances, labels))

            ind = np.arange(counter)
            data = sorted(data, key=lambda x: x[0])

            tot_distances = []
            labels = []
            for element in data:
                tot_distances.append(element[0])
                labels.append(element[1])

            bar = ax.bar(ind[:top_comparisons], tot_distances[:top_comparisons])
            '''
            if the bars should be labeled
            with the exact values on top uncomment the code below
            '''
            # for rect in bar:
            #     height = rect.get_height()
            #     ax.annotate('{}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height))
            ax.set_ylabel(f'summed up distance')
            ax.set_xlabel(f'top {top_matches} image comparison variations')
            ax.set_title(f'{self.algorithm}: total distance {image_type} images')
            ax.set_xticks(ind[:top_comparisons])
            ax.set_xticklabels(labels[:top_comparisons])

        elif score_plot:
            # motive_filename = "similarity_matrix_motive_400_2019_07_05_ORB.pkl"
            # noise_filename = "similarity_matrix_noise_400_2019_07_05_ORB.pkl"
            motive_filename = "similarity_matrix_motive_0.45_2019_07_05_SURF.pkl"
            noise_filename = "similarity_matrix_noise_0.45_2019_07_05_SURF.pkl"
            # motive_filename = "similarity_matrix_motive_750_2019_07_05_SIFT.pkl"
            # noise_filename = "similarity_matrix_noise_750_2019_07_05_SIFT.pkl"
            motive_df = pd.read_pickle(f"{path_pickle_similarity}{motive_filename}")
            motive_indexes = motive_df.index.values
            motive_columns = motive_df.columns.values
            noise_df = pd.read_pickle(f"{path_pickle_similarity}{noise_filename}")
            noise_indexes = noise_df.index.values
            noise_columns = noise_df.columns.values

            score_tuplelist_motive = []
            score_tuplelist_noise = []

            for index in motive_indexes:
                for column in motive_columns:
                    score = motive_df.loc[index, column]
                    if math.isnan(score):
                        score = 0
                    #include only positiv scores
                    if score != 0:
                        score_tuplelist_motive.append((index, column, score * self.score_multiplier))

            for index in noise_indexes:
                for column in noise_columns:
                    score = noise_df.loc[index, column]
                    if math.isnan(score):
                        score = 0
                    #include only positiv scores
                    if score != 0:
                        score_tuplelist_noise.append((index, column, score * self.score_multiplier))

            #sort scores
            score_tuplelist_motive = sorted(score_tuplelist_motive, reverse=True, key=lambda x: x[2])
            score_tuplelist_noise = sorted(score_tuplelist_noise, reverse=True, key=lambda x: x[2])
            #get labels, scores
            scores_motive = []
            scores_noise = []
            labels_motive = []
            labels_noise = []

            for tuple in score_tuplelist_motive:
                labels_motive.append(f"{tuple[0][3:]}_{tuple[1][3:]}")
                scores_motive.append(tuple[2])

            for tuple in score_tuplelist_noise:
                labels_noise.append(f"{tuple[0][3:]}_{tuple[1][3:]}")
                scores_noise.append(tuple[2])

            ind_motive = np.arange(len(score_tuplelist_motive))
            ind_noise = np.arange(len(score_tuplelist_noise))

            ax.bar(ind_motive, scores_motive, label='Motives') #[:top_comparisons]
            ax.bar(ind_noise, scores_noise, label='Noise')

            ax.set_ylabel(f'score * {self.score_multiplier}')
            ax.set_xlabel(f'image comparison variations')
            ax.set_title(f'{self.algorithm}: calculated image comparison scores, threshold {self.threshold}')
            ax.set_xticks(ind_motive)
            ax.set_xticklabels(ind_motive)
            # plt.yscale('log')
            plt.legend()

        plt.show()

    def add_features(self):
        # 1. add the needed amount of new columns according
        # to the length of the similarity matrix
        # for index, row in self.df_similarity.iterrows():

        _columns = self.df_similarity.columns.values
        for index, row in self.df_similarity.iterrows():
            for column, element in zip(_columns, row):
                self.subset_df.at[index, column] = element
        '''
        missing images have to be handeled here
        metadata of media objects with missing images are included in the subset_df
        but are missing in the df_similarity since the images were not loaded in the first place
        the resulting NaN values have to be replaced to 0, otherwise the clustering will raise errors
        UPDATE: these rows where everything is NaN have to be excluded from clustering, because
        they will cause irrelevant clusters!
        '''
        #when reasigning the following expression again to self.subset_df (with inplace=False)
        #then only this subsection of the dataframe will be reasigned to the original.
        #Testing what happens if no reasign but with inplace=True.
        #fillna() doesn't work as intendend as soon as one enters a list of rows AND columns

        #isnull checks for NaN, None -> missing values. Zeros will NOT be removed!
        boolean_array_no_nan_rows = self.subset_df.iloc[:, -1].notnull()
        self.subset_df = self.subset_df[boolean_array_no_nan_rows]

path_IMAGES = "C:/Users/mhartman/Documents/100mDataset/wildkirchli_images"
path_IMAGES_test = "C:/Users/mhartman/Documents/100mDataset/wildkirchli_images_test"
path_IMAGES_motives = "C:/Users/mhartman/Documents/100mDataset/wildkirchli_images_motives"
path_IMAGES_noise = "C:/Users/mhartman/Documents/100mDataset/wildkirchli_images_noise"
path_pickle_similarity = "C:/Users/mhartman/Documents/100mDataset/df_similarity_pickles/"
path_performance_log = "C:/Users/mhartman/Documents/100mDataset/performance_log.txt"

# inst1 = ImageSimilarityAnalysis(path_IMAGES_motives, 'SURF', pickle=False)
