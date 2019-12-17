import numpy as np
import matplotlib.pyplot as plt


class NetworkAnalyser:

    def __init__(self, subset_name, dataframe, threshold=100, m_agreement=2):
        self.subset_name = subset_name
        self.dataframe = dataframe
        self.threshold = threshold
        self.m_agreement = m_agreement
        #check for too low threshold. Assert raised if conndition is NOT true
        assert self.threshold >= 10, "Specified threshold is lower than (min: 50, best:100) recommended to successfully identify motive images!"
        self.new_dataframe = self.network_analysis()
        print("")

    def network_analysis(self):
        try:
            # the amount of rows determines how many images where compared which is needed for the cropping
            images = len(self.dataframe.index.values)
            df = self.dataframe.iloc[:, -images:]
            similarities = {} #lists all images an image is similar to based on the given similarity threshold
            motifs = {} #contains the final motifs
            '''
            1. Crawling through the dataframe to find all similar images for each image based 
            on the given similarity threshold
            '''
            valid_indexes = list(df.index.values)
            for counter, (i, r) in enumerate(df.iterrows()):
                similarities[i] = []
                for k, score in r.items():
                    #if score above threshold, append that photo_id to the similarities of the current image
                    if score > self.threshold:
                        similarities[i].append(k)
            ''' 
            2. Iteratively neglect the images which have less then m_agreement similar images.
            This is done until no more images are removed and therefore every image has more or equal similar images
            as defined by m_agreement which formes robust motifs
            '''
            iterations = 0
            len_start = len(similarities.keys())
            noise = [] #will be attributed motif_cluster = -1
            while True:
                iterations += 1
                neglected = []
                #find images with fewer neighbours than demanded by m_agereement
                for k, neighbours in similarities.items():
                    if len(neighbours) < self.m_agreement:
                        neglected.append(k)
                # #delete the neglected keys from the dictionary
                # for n in neglected:
                #     del similarities[n]
                #remove the neglected images from the remaining neighbour lists
                for k, neighbours in similarities.items():
                    if k not in neglected:
                        for n in neglected:
                            neighbours.remove(n)
                noise = noise + neglected
                #break if no more images were neglected
                if len(neglected) == 0:
                    break
            print("Took {}")
            '''
            3. Define motifs
            '''













        #             # if at least one neighbour has been found; next iterate over these neighbours
        #             if len(clusters[counter]['entities']) != 0:
        #                 # if at least one neighbour has been found for this index, then the index itself is also added to the list
        #                 clusters[counter]['entities'].append(i)
        #                 # iterate over previously found neighbours and add any thereof neigbours to the same cluster
        #                 for neighbour in clusters[counter]['entities']:
        #                     # get row from df mit said neighbour id
        #                     row = df.loc[neighbour, :]
        #                     series_index = row.index
        #                     for counter2, score in enumerate(row):
        #                         if isinstance(score, str):
        #                             print(f"score value: {score}, threshold:{self.threshold}, line 56")
        #                             score = 0
        #                         if score > self.threshold and neighbour != clusters[counter]['seed']:
        #                             clusters[counter]['motif_score'] += score
        #                         if score > self.threshold and series_index[counter2] in valid_indexes:
        #                             clusters[counter]['entities'].append(series_index[counter2])
        #
        #                             valid_indexes.remove(series_index[counter2])
        #     # add necessary new df columns
        #     self.dataframe.loc[:, 'multi_cluster_label'] = np.nan
        #     self.dataframe.loc[:, 'motif_score'] = np.nan
        #     '''
        #     2. Add Cluster labels
        #     to the dataframe (subset)
        #     '''
        #     for cluster_nr, v in clusters.items():
        #         if len(v['entities']) != 0:
        #             for id in v['entities']:
        #                 self.dataframe.loc[id, 'multi_cluster_label'] = cluster_nr
        #                 self.dataframe.loc[id, 'motif_score'] = v['motif_score']
        #
        #     #everything else (still value NaN) needs to be set to noise -1
        #     boolean_array = self.dataframe.multi_cluster_label.isnull()
        #     self.dataframe.loc[boolean_array, 'multi_cluster_label'] = -1
        #
        except Exception as e:
            '''
            If any error occurres the entire cluster will be treated as non-motif and all contained images will be
            labeled noise with a multi_cluster_label of -1
            '''
            print(f"Error during NetworkAnalysis: {e}")
            print(f"Cluster {self.subset_name} will be excluded.")
            self.dataframe.loc[:, 'multi_cluster_label'] = -1
        return self.dataframe