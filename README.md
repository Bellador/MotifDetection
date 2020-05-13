# Motif Detection

## Introduction
Through social media induced tourism people are drawn to places that were previously not experiencing much touristic attentation. Drastic and rapid changes in a location's popularity can have serious effects on the local population, the infrastructure and, last but not least on the environment. Therefore, the monitoring and analysis of social media induced tourism is more important than ever before to ensure local agencies to act in time and occurding to the observed circumstances. Locations that experienced strong attention on social media such as Instagram where shown to be characterised by distinct, reoccuring motif images. We regard motifs as a collection of images which depict the same landscape element(s) from a similar viewpoint. Therefore, the scenic focus, angle and position of the observer are important criteria to decide if two images belong to the same motif or not. This implies that images which show the same landscape element(s) are not automatically considered a motif. These motifs are hypothesised to function as proxy to indicate the locations of exsiting or upcoming social media induced tourism hot spots. We were able to confirm in a casestuy that motif locations pinpoint to more specific and concrete spots inside already popular places that are expossed to additional visitation strain. Therefore, a more granular spatial resolution in regards to popular locations was achieved. Also a temporal analysis of motif development was conducted which did not significantly show the expected exponential increase after a certain trigger event occured. This is up to further investigation.

## Setup
This pipeline uses spatial clustering (HDBSCAN) and computer vision image analysis (SIFT) to detect motifs in images.

1. Initialise all required python packages by using the requirements.txt file to create an identical, compatible anaconda environemnt with the following command: `conda create --name myenv --file requirements.txt`
1.1 Additionally, the github repo 'FlickrFrame' is required if one plans to query the FlickrAPI: 
`git clone https://github.com/Bellador/FlickrFrame.git` and change line 13 in main.py to the path were FlickrFrame was cloned to locally.
2. Adjust the core parameters of main.py regarding:
  - All path parameters for API keys & secrets, database login, storage etc. Search for: 'URL here' and enter the required paths manually
  - Project description
  - SIFT configuration
  - HDBSCAN configuration
  - Motif filtering settings (normally set to None to allow dynamic filtering on the entire output with 'geomap_filter_for_motifsize.py') 
  - data source (FlickrAPI, external (Postgresql) database, local directory with .csv data files)
3. Adjust db_querier.py which handles database handling and querying if a database shall be used as data storage for the pipeline
4. Run main.py

## Output
For each new project a designated directory will be created which stores all the output produced by the pipeline. This output includes:
  - .csv file containing original FlickrAPI data (if FlickrAPI was selected as data source)
  - downloaded Flickr images (if FlickrAPI was selected as data source)
  - Folder named 'cluster_hmtl_inspect' which holds html-files for each HDBSCAN cluster and the therein identified motifs. The filenames include the motif score which is a measure for motif quality as well as the used filter parameters. These filter parameters are also listed in the header of the html file itself along with the actual visual motif output.
  - Folder named 'dataframe_pickles' which holds the entire pipeline calculation output for each HDBSCAN cluster. The dataframe encompasses all photo_ids of the contained images, their image similarity score matrix and their final motif cluster label. This data can used in subsequent processing steps to further filter and plot results!
  
## Further data processing
We added a script called 'geomap_filter.py' to the repo which makes further processing of the created dataframe pickle output easy. The script allows to filter all HDBSCAN clusters for motifs with a given minimum size as well as a minimum amount of unique authors. Additionally, it outputs a .CSV file containing the motifs that meet the filter requirements together among others with the motigs mean coordinates. This enables convinient visualisation and plotting in any GIS.

## Processed Results
The following results were based on an individually updated version of the original Yahoo Flickr Creative Commons 100 Million Database (YFCC100m at https://multimediacommons.wordpress.com/yfcc100m-core-dataset/). The research area was set exclusively to the Nature 2000 protected areas of Europe (see https://ec.europa.eu/environment/nature/natura2000/index_en.htm).

![alt text](https://github.com/Bellador/MotiveDetection/blob/master/map_v2.png)
The map shows the spatial distribution and occurrence of the motifs by our processing pipelinein the Natura 2000 protected areas.  In total 119 motifs were identified of which 68 are culturalmotifs, 36 natural motifs and 15 false positive motifs.  All displayed motifs contain a minimum offive images by a minimum of five unique authors.  The data foundation for this map is theupdated Yahoo Flickr Creative Commons 100 Million dataset. For visualisation purposes a subsample of the 119 motifs is displayed with an image while indicating their motif type in color.

*Image sources:  (place name, flickr username, flickr image id, user license)*

*Le Mont-Saint-Michel: Pepe Martınez Camara - 15286893754 - [CC BY-NC-SA 2.0]*

*Durdle Door:  KC2000 - 3620585132 - [CC BY-NC-ND 2.0]*

*Cliffs of Moher:  gnu1742 - 6580681781 - [CC BY-NC-SA 2.0]*

*Kylemore Abbey:  Johnny Graber - 15730653587 - [CC BY-NC-SA 2.0]*

*Forth Bridge:  bryan...  - 26331288697 - [CC BY-SA 2.0]*

*Predjamski Grad:  Tom - 14889396090 - [CC BY-ND 2.0]*

*Skradinski Buk:  Igor Gushchin - 37953928496 - [CC BY 2.0]*

*Kap Sounion:  Marc - 4835837372 - [CC BY-NC-ND 2.0]*

*Blue Grotto:  Michael Holler - 19879729385 - [CC BY-NC 2.0]*

*Schloss Neuschwanstein:  Jiuguang Wang - 5134934131 - [CC BY-SA 2.0]*

*Ponta de S ̃ao Louren ̧co:  ERREACHE - 48587252767 - [CC BY-NC-ND 2.0]*

----------------

![alt text](https://github.com/Bellador/MotiveDetection/blob/master/motif_type_figure_new_300dpi.jpg)
One example per motif class respectively of the 68 culture and 36 nature motifs that were found inside the Natura 2000 sites.
*Image sources from left-to-right (Flickr username, Flickr image id, user license):
*Le Mont-Saint-Michel
*1. Pepe Martínez Cámara - 15286893684 [CC BY-NC-SA 2.0]
*2. Pablo Garbarino - 15381699077 [CC BY-NC-ND 2.0]
*3. marottef - 8845388500 [CC BY-NC 2.0]
*4. Paolo Ramponi - 12415418924 - [CC BY-NC-SA 2.0]
*Blue Grotto
*1. Michael Holler - 19879729385 - [CC BY-NC 2.0]
*2. Joris Gruber - 25475401601 - [CC BY-NC-ND 2.0]
*3. Sin Amigos - 2963000725 - [CC BY 2.0]
*4. Chris Jagers - 32406808006 - [CC BY-NC-ND 2.0]*

----------------

<p align="center">
  <img src="https://github.com/Bellador/MotiveDetection/blob/master/le_mont_saint_michel_submap_newlegend2.png" title="class specific model precision">
</p>
We identified two motifs of Le Mont-Saint-Michel based on Creative Commons Flickr imageswith a minimum of five unique authors and motif images.  The figure illustrates the spatialdistribution and proportion between the input data, the formed spatial clusters and finally theresulting motifs.

