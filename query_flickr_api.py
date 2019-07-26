import flickrapi
import re
import json
import urllib
import ssl
import datetime
import time
import os
from functools import wraps

class FlickrQuerier:

    path_CREDENTIALS = "C:/Users/mhartman/PycharmProjects/MotiveDetection/FLICKR_API_KEY.txt"
    path_saveimages_wildkirchli = "C:/Users/mhartman/Documents/100mDataset/wildkirchli_images/"
    path_LOG = "C:/Users/mhartman/PycharmProjects/MotiveDetection/LOG_FLICKR_API.txt"
    # path_CSV = "C:/Users/mhartman/PycharmProjects/MotiveDetection/wildkirchli_metadata.csv"

    class Decorators:
        # decorator to wrap around functions to log if they are being called
        @classmethod
        def logit(self, func):
            #preserve the passed functions (func) identity - so I doesn't point to the 'wrapper_func'
            @wraps(func)
            def wrapper_func(*args, **kwargs):
                with open(FlickrQuerier.path_LOG, 'at') as log_f:
                    #print("Logging...")
                    log_f.write('-'*20)
                    log_f.write(f'{datetime.datetime.now()} : function {func.__name__} called \n')
                return func(*args, **kwargs)
            return wrapper_func

    def __init__(self, project_name, bbox, min_upload_date=None, max_upload_date=None):
        print("--"*30)
        print("Initialising Flickr Search with FlickrQuerier Class")
        self.project_name = project_name
        self.project_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), project_name)
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.bbox = bbox
        self.min_upload_date = min_upload_date
        self.max_upload_date = max_upload_date
        self.api_key, self.api_secret = self.load_creds(FlickrQuerier.path_CREDENTIALS)
        print("--" * 30)
        print(f"Loading flickr API credentials - done.")
        print("--" * 30)
        print(f"Quering flickr API with given bbox: \n{self.bbox}")
        self.unique_ids, self.flickr = self.flickr_search()
        print("--" * 30)
        print(f"Search - done.")
        print("--" * 30)
        print(f"Fetching metadata for search results and writing to file...")
        self.get_info()
        print("--" * 30)
        print(f"Acquiring metadata - done.")
        print("--" * 30)
        print(f"Downloading images into folder {project_name} to current directory.")
        self.get_images(self.unique_ids, self.flickr)
        print("\n--" * 30)
        print(f"Download images - done.")
        print("--" * 30)
        print("--" * 30)
        print("FlickrQuerier Class - done")

    @Decorators.logit
    def load_creds(self, path):
        key_found = False
        secret_found = False
        with open(FlickrQuerier.path_CREDENTIALS, 'r') as f:
            for line in f:
                if key_found:
                    api_key = line.strip().encode('utf-8')
                    key_found = False

                if secret_found:
                    api_secret = line.strip().encode('utf-8')
                    secret_found = False

                if re.match(r'<KEY>', line):
                    key_found = True
                    continue
                elif re.match(r'<SECRET>', line):
                    secret_found = True
        return api_key, api_secret

    def flickr_search(self):
        flickr = flickrapi.FlickrAPI(self.api_key, self.api_secret, format='json')
        photos = flickr.photos.search(bbox=self.bbox, min_upload_date=self.min_upload_date, max_upload_date=self.max_upload_date, per_page=250) #is_, accuracy=12, commons=True, page=1, min_taken_date='YYYY-MM-DD HH:MM:SS'
        # print(json.dumps(json.loads(photos.decode('utf-8')), indent=2))
        result = json.loads(photos.decode('utf-8'))
        '''
        Handling for multipage results stored in result_dict
        '''
        pages = result['photos']['pages']
        result_dict = {}
        result_dict['page_1'] = result
        if pages != 1 and pages != 0:
            print(f"Search returned {pages} result pages")
            for page in range(2, pages+1):
                print(f"Querying page {page}...")
                try:
                    result_bytes = flickr.photos.search(bbox=self.bbox, min_upload_date=self.min_upload_date,
                                         max_upload_date=self.max_upload_date, page=page, per_page=250)
                    result_dict[f'page_{page}'] = json.loads(result_bytes.decode('utf-8'))
                except Exception as e:
                    print("*" * 30)
                    print("*" * 30)
                    print("Error occurred: {}".format(e))
                    print("sleeping 5s...")
                    print("*" * 30)
                    print("*" * 30)
                    time.sleep(5)

        print("All pages handled.")
        #get ids of returned flickr images
        ids = []
        for dict_ in result_dict:
            for element in result_dict[dict_]['photos']['photo']:
                ids.append(element['id'])
        unique_ids = set(ids)

        print(f"Total results found: {len(unique_ids)}")

        return unique_ids, flickr

    def get_images(self, ids, flickr):
        self.image_path = os.path.join(self.project_path, f'images_{self.project_name}')
        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path)
            print(f"Creating image folder 'images_{self.project_name}' in sub-directory '/{self.project_name}/' - done.")
        else:
            print(f"Image folder 'images_{self.project_name}' exists already in the sub-directory '/{self.project_name}/'.")

        for index, id in enumerate(ids):
            results = json.loads(flickr.photos.getSizes(photo_id=id).decode('utf-8'))
            # print(json.dumps(json.loads(results.decode('utf-8')), indent=2))
            try:
                # Medium 640 image size url
                url_medium = results['sizes']['size'][6]['source']
                # urllib.request.urlretrieve(url_medium, path) # context=ssl._create_unverified_context()
                resource = urllib.request.urlopen(url_medium, context=ssl._create_unverified_context())
                with open(self.image_path + '/' + f"{id}.jpg", 'wb') as image:
                    image.write(resource.read())
                print(f"\rretrieved {index} of {len(ids)} images", end='')

            except Exception as e:
                print(f"image not found: {e}")

    def get_info(self):
        csv_separator = ';'
        tag_connector = '+'

        def remove_non_ascii(s):
            return "".join(i for i in s if ord(i) < 126 and ord(i) > 31)

        def create_header(data_dict):
            header_string = f'photo_id{csv_separator}'
            for tracker, element in enumerate(data_dict.keys(), 1):
                if tracker < len(data_dict.keys()):
                    header_string = header_string + str(element) + csv_separator
                elif tracker == len(data_dict.keys()):
                    header_string = header_string + str(element)
            return header_string

        def create_line(id, data_dict):
            line = f'{id}{csv_separator}'
            tracker = 1
            for key, value in data_dict.items():
                if tracker < len(data_dict.keys()):
                    line = line + str(value) + csv_separator
                elif tracker == len(data_dict.keys()):
                    line = line + str(value)
                tracker += 1
            return line

        self.csv_output_path = self.dir_path + '/{}/metadata_{}_{:%Y_%m_%d}.csv'.format(self.project_name, self.project_name, datetime.datetime.now())

        with open(self.csv_output_path, 'w', encoding='utf-8') as f:
            for index, id in enumerate(self.unique_ids):
                results = json.loads(self.flickr.photos.getInfo(photo_id=id).decode('utf-8'))
                #get the top level
                try:
                    results = results['photo']
                except Exception as e:
                    print(f"{e} - No metadata found")
                    continue
                '''
                define which info fields should be fetched.
                ERASE ALL STRINGS OF CSV SEPERATOR! 
                '''
                # extract tags into an string separated by '+'!
                tag_string = ''
                for tag_index, tag in enumerate(results['tags']['tag']):
                    tag_string = tag_string + results['tags']['tag'][tag_index]['_content'].replace(csv_separator, '').replace(tag_connector, '') + tag_connector

                try:
                    locality = results['location']['locality']['_content'].replace(csv_separator, '')
                except Exception as e:
                    locality = ''
                    print(f"{e} not found. Continue")

                try:
                    county = results['location']['county']['_content'].replace(csv_separator, '')
                except Exception as e:
                    county = ''
                    print(f"{e} not found. Continue")

                try:
                    region = results['location']['region']['_content'].replace(csv_separator, '')
                except Exception as e:
                    region = ''
                    print(f"{e} not found. Continue")

                try:
                    country = results['location']['country']['_content'].replace(csv_separator, '')
                except Exception as e:
                    country = ''
                    print(f"\n{e} not found. Continue")

                '''
                text clean up
                of title and description
                - remove linebreaks etc.
                '''
                description = remove_non_ascii(results['description']['_content'].replace(csv_separator, ''))
                title = remove_non_ascii(results['title']['_content'].replace(csv_separator, ''))

                data = {
                    'author_id': results['owner']['nsid'].replace(csv_separator, ''),
                    'author_origin': results['owner']['location'].replace(csv_separator, ''),
                    'title': title,
                    'description': description,
                    'upload_date': results['dates']['posted'].replace(csv_separator, ''),
                    'taken_date': results['dates']['taken'].replace(csv_separator, ''),
                    'views': results['views'].replace(csv_separator, ''),
                    'url': results['urls']['url'][0]['_content'].replace(csv_separator, ''),
                    'tags':  tag_string,

                    #location information

                    'lat': results['location']['latitude'].replace(csv_separator, ''),
                    'lng': results['location']['longitude'].replace(csv_separator, ''),
                    'accuracy': results['location']['accuracy'].replace(csv_separator, ''),
                    'locality': locality,
                    'county': county,
                    'region': region,
                    'country': country
                }

                if index == 0:
                    header = create_header(data)
                    f.write(f"{header}\n")
                if index % 50 == 0 and index != 0:
                    print(f"\rLine {index} processed", end='')

                line = create_line(id, data)
                f.write(f"{line}\n")

        print(f"\nCreated output file: {self.csv_output_path}")