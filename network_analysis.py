import numpy as np
import matplotlib.pyplot as plt


class NetworkAnalyser:

    def __init__(self, subset_name, dataframe, threshold=100):
        self.subset_name = subset_name
        self.dataframe = dataframe
        self.threshold = threshold
        #check for too low threshold. Assert raised if conndition is NOT true
        assert self.threshold >= 10, "Specified threshold is lower than (min: 50, best:100) recommended to successfully identify motive images!"
        self.new_dataframe = self.network_analysis(dataframe, threshold=self.threshold)
        print("")

    def network_analysis(self, dataframe, threshold=100):
        try:
            # the amount of rows determines how many images where compared which is needed for the cropping
            images = len(dataframe.index.values)
            df = dataframe.iloc[:, -images:]
            # boolean to see which values are above the threshold, second parameter specifies the value for if the condition is false

            # mask_df = df.where(df > threshold, False).where(df <= threshold, True)
            '''
            1. Crawling through df to find similar neighbours
            Iterate over rows, find neighbours/images with a similarity score above the threshold
            '''
            # iterate over the rows which returns the index_name and the row as dictionary with column_labels as keys
            clusters = {}
            # list of indexes which are still not in an cluster and over which will still be iterated
            valid_indexes = list(df.index.values)
            for counter, (i, r) in enumerate(df.iterrows()):
                if i in valid_indexes:
                    clusters[counter] = {'entities': [],
                                         'seed': i,
                                         'motif_score': 0}
                    # iterate over the initial row to find neighbours
                    for k, score in r.items():
                        if k in valid_indexes:
                            if isinstance(score, str):
                                print(f"score value: {score}, threshold:{threshold}, line 38")
                                score = 0
                            if score > threshold:
                                clusters[counter]['entities'].append(k)
                                clusters[counter]['motif_score'] += score
                                # again, remove processed neighbour index
                                valid_indexes.remove(k)
                    valid_indexes.remove(i)
                    # if at least one neighbour has been found; next iterate over these neighbours
                    if len(clusters[counter]['entities']) != 0:
                        # if at least one neighbour has been found for this index, then the index itself is also added to the list
                        clusters[counter]['entities'].append(i)
                        # iterate over previously found neighbours and add any thereof neigbours to the same cluster
                        for neighbour in clusters[counter]['entities']:
                            # get row from df mit said neighbour id
                            row = df.loc[neighbour, :]
                            series_index = row.index
                            for counter2, score in enumerate(row):
                                if isinstance(score, str):
                                    print(f"score value: {score}, threshold:{threshold}, line 56")
                                    score = 0
                                if score > threshold and neighbour != clusters[counter]['seed']:
                                    clusters[counter]['motif_score'] += score
                                if score > threshold and series_index[counter2] in valid_indexes:
                                    clusters[counter]['entities'].append(series_index[counter2])

                                    valid_indexes.remove(series_index[counter2])
            # add necessary new df columns
            dataframe.loc[:, 'multi_cluster_label'] = np.nan
            dataframe.loc[:, 'motif_score'] = np.nan
            '''
            2. Add Cluster labels
            to the dataframe (subset)
            '''
            for cluster_nr, v in clusters.items():
                if len(v['entities']) != 0:
                    for id in v['entities']:
                        dataframe.loc[id, 'multi_cluster_label'] = cluster_nr
                        dataframe.loc[id, 'motif_score'] = v['motif_score']

            #everything else (still value NaN) needs to be set to noise -1
            boolean_array = dataframe.multi_cluster_label.isnull()
            dataframe.loc[boolean_array, 'multi_cluster_label'] = -1

            '''
            3. Plot
            if minimum one cluster besides noise exists
            '''
            # unique_labels = dataframe.multi_cluster_label.unique()

            # if len(unique_labels) > 1:
            #     # Black removed and is used for noise instead.
            #     colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
            #     '''
            #     Sort labels so that the noise points get plotted first
            #     and don't cover important clusters
            #     '''
            #     for label_counter, (cluster_label, col) in enumerate(zip(sorted(unique_labels, key=lambda x: x), colors)):
            #         # filter dataframe for rows of given cluster
            #         label = f"c_{label_counter}"
            #         if cluster_label == -1:
            #             # Black used for noise.
            #             col = [0, 0, 0, 1]
            #             label = 'noise'
            #         # returns boolean array with true when condition is met
            #         boolean_array = self.dataframe['multi_cluster_label'] == cluster_label
            #         rows = self.dataframe[boolean_array]
            #         '''
            #         latitude and longitude
            #         must most likely be exchanged for some reason
            #         to match the ArcMap reprentation
            #         '''
            #         plt.plot(rows.loc[:, 'lng'], rows.loc[:, 'lat'], 'o', markerfacecolor=tuple(col), markeredgecolor='k',
            #                  markersize=14, label=label)  # replaced xy with X
            #     '''
            #     Adjust plot X and Y extend on
            #     all points except noise
            #
            #     query all points that are not noise from dataframe
            #     '''
            #     boolean_array_not_noise = self.dataframe.multi_cluster_label != -1
            #     not_noise = self.dataframe[boolean_array_not_noise]
            #
            #     buffer = 0.0005
            #
            #     xlim_left = not_noise.lng.min() - buffer
            #     xlim_right = not_noise.lng.max() + buffer
            #     ylim_bottom = not_noise.lat.min() - buffer
            #     ylim_top = not_noise.lat.max() + buffer
            #     plt.xlim(left=xlim_left, right=xlim_right)
            #     plt.ylim(bottom=ylim_bottom, top=ylim_top)
            #     plt.title(f"Image similarity {self.subset_name}")
            #     plt.legend()
            #     plt.show()
        except Exception as e:
            '''
            If any error occurres the entire cluster will be treated as non-motif and all contained images will be 
            labeled noise with a multi_cluster_label of -1
            '''
            print(f"Error during NetworkAnalysis: {e}")
            print(f"Cluster {self.subset_name} will be excluded.")
            dataframe.loc[:, 'multi_cluster_label'] = -1
        return dataframe