# Motif Detection

## Introduction
Through social media induced tourism people are drawn to places that were previously not experiencing much touristic attentation. Drastic and rapid changes in a location's popularity can have serious effects on the local population, the infrastructure and, last but not least on the environment. Therefore, the monitoring and analysis of social media induced tourism is more important than ever before to ensure local agencies to act in time and occurding to the observed circumstances. Locations that experienced strong attention on social media such as Instagram where shown to be characterised by distinct, reoccuring motif images. We regard motifs as a collection of images which depict the same landscape element(s) from a similar viewpoint. Therefore, the scenic focus, angle and position of the observer are important criteria to decide if two images belong to the same motif or not. This implies that images which show the same landscape element(s) are not automatically considered a motif. These motifs are hypothesised to function as proxy to indicate the locations of exsiting or upcoming social media induced tourism hot spots. We were able to confirm in a casestuy that motif locations pinpoint to more specific and concrete spots inside already popular places that are expossed to additional visitation strain. Therefore, a more granular spatial resolution in regards to popular locations was achieved. Also a temporal analysis of motif development was conducted which did not significantly show the expected exponential increase after a certain trigger event occured. This is up to further investigation.

## Setup
This pipeline uses spatial clustering (HDBSCAN) and computer vision image analysis (SIFT) to detect motifs in images.

1. Initialise all required python packages by using the requirements.txt file to create an identical, compatible anaconda environemnt with the following command: ``conda create --name myenv --file requirements.txt``
2. Adjust the core parameters of main.py regarding:
  - Project description
  - SIFT configuration
  - HDBSCAN configuration
  - Motif filtering settings
  - data source (FlickrAPI, external database, local directory with .csv data files)
3. Adjust db_querier.py which handles database handling and querying if a database shall be used as data storage for the pipeline
4. Run main.py

![alt text](https://github.com/Bellador/MotiveDetection/blob/master/motif_map_w_legend.png)
The map shows the spatial distribution and occurrence of the motifs by our processing pipeline in the Natura 2000 protected areas. In total 119 motifs were identified of which 68 are cultural motifs, 36 natual motifs and 15 false positive motifs. All displayed motifs contain a minimum of five images by a minimum of five unique authors. The data foundation for this map is the updated Yahoo Flickr Creative Commons 100 Million dataset. For visualisation purposes a sub sample of the 119 motifs are displayed with an image while indicating their motif type in color
