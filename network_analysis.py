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
        self.network_analysis()
        self.new_dataframe = self.dataframe
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
                #will contain tuples of images that are similar to the image in question with the respective score
                similarities[i] = []
                for k, score in r.items():
                    #if score above threshold, append that photo_id to the similarities of the current image
                    if score > self.threshold:
                        similarities[i].append((k, score))
            ''' 
            2. Iteratively neglect the images which have less then m_agreement similar images.
            This is done until no more images are removed and therefore every image has more or equal similar images
            as defined by m_agreement which formes robust motifs
            '''
            iterations = 0
            noise = [] #will be attributed motif_cluster = -1
            while True:
                iterations += 1
                neglected = []
                #find images with fewer neighbours than demanded by m_agereement
                for k, neighbours in similarities.items():
                    if len(neighbours) < self.m_agreement:
                        neglected.append(k)
                #delete neglected keys
                for neg in neglected:
                    del similarities[neg]
                #remove the neglected images from the remaining neighbour lists
                for k, neighbours in similarities.items():
                    neighbours_update = list(filter(lambda x: x[0] not in neglected, neighbours))#[x for x in neighbours if x[0] not in neglected]
                    similarities[k] = neighbours_update
                noise = noise + neglected
                #break if no more images were neglected
                if len(neglected) == 0:
                    break
            '''
            3. Define motifs
            '''
            processed = []
            for counter, (key, neighs) in enumerate(similarities.items()):
                if key not in processed:
                    motifs[counter] = {'entities': [],
                                       'motif_score': 0}
                    #add all neigbours of the image to the created motif cluster
                    for n in neighs:
                        motifs[counter]['entities'].append(n[0])
                        motifs[counter]['motif_score'] += n[1]
                        processed.append(n[0])
                    #add images from the neighbours that are not yet processed also to the same motif cluster
                    for e in motifs[counter]['entities']:
                        for neighbour_n in similarities[e]:
                            if neighbour_n[0] not in processed:
                                motifs[counter]['entities'].append(neighbour_n[0])
                                motifs[counter]['motif_score'] += neighbour_n[1]
                                processed.append(neighbour_n[0])
            '''
            4. Add Cluster labels
            to the dataframe (subset)
            '''
            # add necessary new df columns
            self.dataframe.loc[:, 'multi_cluster_label'] = np.nan
            self.dataframe.loc[:, 'motif_score'] = np.nan
            for cluster_nr, v in motifs.items():
                if len(v['entities']) != 0:
                    for id in v['entities']:
                        self.dataframe.loc[id, 'multi_cluster_label'] = cluster_nr
                        self.dataframe.loc[id, 'motif_score'] = v['motif_score']
            # everything else (still value NaN) needs to be set to noise -1
            boolean_array = self.dataframe.multi_cluster_label.isnull()
            self.dataframe.loc[boolean_array, 'multi_cluster_label'] = -1

            print(f"iterations {iterations}; noise: {len(noise)}; motifs: {len(motifs.keys())}")

        except Exception as e:
            '''
            If any error occurres the entire cluster will be treated as non-motif and all contained images will be
            labeled noise with a multi_cluster_label of -1
            '''
            print(f"\nError during NetworkAnalysis: {e}")
            print(f"Cluster {self.subset_name} will be excluded\n.")
            self.dataframe.loc[:, 'multi_cluster_label'] = -1