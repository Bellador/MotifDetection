import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import math
import re
import time

SIFT_params = {
        'algorithm': 'SIFT',
        'lowe_ratio': 0.7, #0.775
    }
ORB_params = {
    'algorithm': 'ORB',
    'lowe_ratio': 0.7,
}

def url_img_read_test():
    image_url = "http://farm1.staticflickr.com/1/28605_d138ef7b45.jpg"
    image_path = "C:/Users/mhartman/PycharmProjects/MotiveDetection/wildkirchli/images_wildkirchli/2943200125.jpg"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print("")

class ImageSimilarityAnalysis:
    score_same_image = 0
    # score_same_image = 1
    core_df_range = 17

    def __init__(self, algorithm_params, pickle=False):
        start = time.time()
        self.project_name = 'wildkirchli'
        self.algorithm_params = algorithm_params
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
        self.df, self.df_similarity = self.match_keypoints(lowe_ratio=self.lowe_ratio, pickle_similarity_matrix=pickle)
        print("--" * 30)
        print("Compute image similarity dataframe - done.")

    def file_loader(self):
        # load image as grayscale since following 3 algoithms ignore RGB information
        image_objects = {}
        feature_dict = {}
        filename_tracker = {}

        files = [os.path.join(self.images_path, file) for file in os.listdir(self.images_path) if
                 os.path.isfile(os.path.join(self.images_path, file))]

        pattern = r"([\d]*)\.jpg$"
        image_objects = {}
        feature_dict = {}
        for file in files:
            file_id = int(re.search(pattern, file).group(1))
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
            # the None defines if a mask shall be used or not
            keypoints, descriptors = alg.detectAndCompute(self.image_objects[obj], None)
            self.feature_dict[obj]['kp'] = keypoints
            self.feature_dict[obj]['ds'] = descriptors


    def match_keypoints(self, lowe_ratio=0.8, pickle_similarity_matrix=True):
        '''
        Brute Force
        Matching between the different images
        - Create pandas dataframe to store the matching output
        of every image (n) with the set in a 2D matrix of n^2 entries
        '''
        df_pickle_path = "C:/Users/mhartman/Documents/100mDataset/"
        self.lowe_ratio = lowe_ratio
        # one to store the matches, the other for the later computed similarity scores
        df = pd.DataFrame(columns=self.image_objects.keys(), index=self.image_objects.keys())
        df_similiarity = pd.DataFrame(columns=self.image_objects.keys(), index=self.image_objects.keys())
        # normalising function dependant on used algorithm - NORM_HAMMING good for orb
        bf = cv2.BFMatcher()  # cv2.NORM_L1, crossCheck=True
        # store already sorted match results in corresponding matrix cells
        indexes = df.index.values
        columns = df.columns.values
        '''
        here reduce the redundancy
        by avoiding to check the same comparisons twice                
        '''
        [df.set_value(index, column, bf.knnMatch(self.feature_dict[index]['ds'], self.feature_dict[column]['ds'], k=2))
         for check1, index in enumerate(indexes)
            for check2, column in enumerate(columns)
                if check2 > check1]

        print("Populated dataframe with image matches - done.")
        recorded_match_lengths = []
        # iterate over dataframe to calculate similarity score and populate second df
        for row in indexes:
            for col in columns:
                # distances = []
                similar_regions = []
                matches = df.loc[row, col]

                if isinstance(matches, (list,)):  # would work too: type(matches) == list
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
                        score = len(similar_regions) #/ len(matches)

                        df_similiarity.set_value(row, col, score)

                    recorded_match_lengths.append(len(matches))

                elif math.isnan(matches):
                    # has to be zero not nan for the clustering to work
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
                df_similiarity.to_pickle(
                    "{}similarity_matrix_motive_{}_{:%Y_%m_%d}_{}.pkl".format(self.project_path, self.threshold,
                                                                              datetime.datetime.now(), self.algorithm))

            elif re.search(r"noise", self.workpath):
                df_similiarity.to_pickle(
                    "{}similarity_matrix_noise_{}_{:%Y_%m_%d}_{}.pkl".format(self.project_path, self.threshold,
                                                                             datetime.datetime.now(), self.algorithm))

        return df, df_similiarity

# ImageSimilarityAnalysis(ORB_params)
url_img_read_test()