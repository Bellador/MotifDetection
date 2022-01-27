# Motif Detection

Published in Journal of Outdoor Recreation and Tourism titled:

"Automated motif identification: Analysing Flickr images to identify popular viewpoints in Europe’s protected areas"

https://doi.org/10.1016/j.jort.2021.100479

## Introduction
Through social media induced tourism people are drawn to places that were previously not experiencing much touristic attention. Drastic and rapid changes in a location's popularity can have serious effects on the local population, the infrastructure and, last but not least on the environment. Therefore, the monitoring and analysis of social media induced tourism is more important than ever before to ensure local agencies to act in time and occurring to the observed circumstances. Locations that experienced strong attention on social media such as Instagram where shown to be characterised by distinct, reoccurring motif images. We regard motifs as a collection of images which depict the same landscape element(s) from a similar viewpoint. Therefore, the scenic focus, angle and position of the observer are important criteria to decide if two images belong to the same motif or not. This implies that images which show the same landscape element(s) are not automatically considered a motif. These motifs are hypothesised to function as proxy to indicate the locations of existing or upcoming social media induced tourism hot spots. We were able to confirm in a case-study that motif locations pinpoint to more specific and concrete spots inside already popular places that are exposed to additional visitation strain. Therefore, a more granular spatial resolution in regards to popular locations was achieved. Also a temporal analysis of motif development was conducted which did not significantly show the expected exponential increase after a certain trigger event occurred. This is up to further investigation.

## Setup
This pipeline uses spatial clustering (HDBSCAN) and computer vision image analysis (SIFT) to detect motifs in images.

1. Initialise all required python packages by using the `requirements.txt` file to create an identical, compatible anaconda environment with the following command: 
`conda create --name myenv --file requirements.txt`
1.1 Additionally, the Github repo 'FlickrFrame' is required if one plans to query the FlickrAPI: 
`git clone https://github.com/Bellador/FlickrFrame.git` and change line 13 in main.py to the path were FlickrFrame was cloned to locally.
2. Adjust the core parameters of `main.py` regarding:
  - All path parameters for API keys & secrets, database login, storage etc. Search for: 'URL here' and enter the required paths manually
  - Project description
  - SIFT configuration
  - HDBSCAN configuration
  - Motif filtering settings (normally set to None to allow dynamic filtering on the entire output with `geomap_filter_for_motifsize.py`) 
  - data source (FlickrAPI, external (Postgresql) database, local directory with .csv data files)
3. Adjust `db_querier.py` which handles database handling and querying if a database shall be used as data storage for the pipeline
4. Run `main.py`

## Output
For each new project a designated directory will be created which stores all the output produced by the pipeline. This output includes:
  - .csv file containing original FlickrAPI data (if FlickrAPI was selected as data source)
  - downloaded Flickr images (if FlickrAPI was selected as data source)
  - Folder named 'cluster_html_inspect' which holds html-files for each HDBSCAN cluster and the therein identified motifs. The filenames include the motif score which is a measure for motif quality as well as the used filter parameters. These filter parameters are also listed in the header of the html file itself along with the actual visual motif output.
  - Folder named `dataframe_pickles` which holds the entire pipeline calculation output for each HDBSCAN cluster. The data-frame encompasses all photo_ids of the contained images, their image similarity score matrix and their final motif cluster label. This data can used in subsequent processing steps to further filter and plot results!
  
## Further data processing
We added a script called `geomap_filter.py` to the repo which makes further processing of the created data-frame pickle output easy. The script allows to filter all HDBSCAN clusters for motifs with a given minimum size as well as a minimum amount of unique authors. Additionally, it outputs a .CSV file containing the motifs that meet the filter requirements together among others with the motifs mean coordinates. This enables convenient visualisation and plotting in any GIS.


## Processed Results
The following results were based on an individually updated version of the original Yahoo Flickr Creative Commons 100 Million Database ([YFCC100m](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/). The research area was set exclusively to the [Nature 2000 protected areas of Europe]( https://ec.europa.eu/environment/nature/natura2000/index_en.htm).

![map](https://github.com/Bellador/MotiveDetection/blob/master/map_v2.png)
The map shows the spatial distribution and occurrence of the motifs by our processing pipeline in the Natura 2000 protected areas. In total 119 motifs were identified of which 68 are cultural motifs, 36 natural motifs and 15 false positive motifs. All displayed motifs contain a minimum of five images by a minimum of five unique authors. The data foundation for this map is the updated Yahoo Flickr Creative Commons 100 Million dataset. For visualisation purposes a sub sample of the 119 motifs are displayed with an image while indicating their motif type in colour

----------------

![motif_types](https://github.com/Bellador/MotiveDetection/blob/master/motif_type_figure.png)
Visual examples of an identified culture and nature motif

----------------

<p align="center">
  <img src="https://github.com/Bellador/MotiveDetection/blob/master/le_mont_saint_michel_submap_newlegend2.png" title="class specific model precision">
</p>

The map shows the spatial distribution and occurrence of the motifs by our processing pipeline in the Natura 2000 protected areas. In total 119 motifs were identified of which 68 are cultural motifs, 36 natural motifs and 15 false positive motifs. 
All displayed motifs contain a minimum of five images by a minimum of five unique authors. The data foundation for this map is the updated Yahoo Flickr Creative Commons 100 Million dataset. 
For visualisation purposes a sub sample of the 119 motifs are displayed with an image while indicating their motif type in colour


### Cite

Maximilian C. Hartmann, Olga Koblet, Manuel F. Baer, Ross S. Purves,
Automated motif identification: Analysing Flickr images to identify popular viewpoints in Europe’s protected areas,
Journal of Outdoor Recreation and Tourism,
Volume 37,
2022,
100479,
ISSN 2213-0780,
https://doi.org/10.1016/j.jort.2021.100479.
(https://www.sciencedirect.com/science/article/pii/S2213078021001158)


Abstract: Visiting landscapes and appreciating them from specific viewpoints is not a new phenomenon. Such so-called motifs were popularised by travel guides and art in the romantic era, and find their contemporary digital twins through images captured in social media. We developed and implemented a conceptual model of motifs, based around spatial clustering, image similarity and the appreciation of a motif by multiple individuals. We identified 119 motifs across Europe, using 2146176 georeferenced Creative Commons Flickr images found in Natura 2000 protected areas. About 65% of motifs contain cultural elements such as castles or bridges. The remaining 35% are natural features, and biased towards coastal elements such as cliffs. Characterisation and localisation of motifs could allow identification of locations subject to increased pressure, and thus disturbance, especially since the visual characteristics of motifs allow managers to explore why sites are being visited. Future work will include methods of calculating image similarity using tags, explore different algorithms for assessing content similarity and study the behaviour of motifs through time.
Keywords: Landscape perception; User-generated content; Computer vision; Image similarity; Places of interest; Social media
