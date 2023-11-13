import os

# Root folder for project code vs computer host
roots = {"DESKTOP-JV9DACD":
             {"vct-ml-yolov8": "C:/repos/vct-ml-yolov8"},
         "PC-514445":
             {"vct-ml-yolov8": "C:/repos/vct-ml-yolov8"}}
PROJECT_PATH = roots[os.getenv("computername")]["vct-ml-yolov8"]

# Get Dataset paths
sources = {"DESKTOP-JV9DACD":
               {"datasets_path": "G:/Datasets/yolov8"},
           "PC-514445":
               {"datasets_path": "I:/Datasets/yolov8"}}

DATASETS_PATH = sources[os.getenv("computername")]["datasets_path"]
RESULTS_PATH = DATASETS_PATH + '/results'
WEIGHTS_PATH = RESULTS_PATH + '/weights'
PREDICTS_PATH = RESULTS_PATH + '/predicts'
SUMMARIES_PATH = RESULTS_PATH + '/summaries'
TEST_DATA_PATH = RESULTS_PATH + '/test/output_tests'

PRETRAINED = DATASETS_PATH + '/pretrained'
TEST_IMAGES = DATASETS_PATH + '/test_images'
TEST_VIDEOS = DATASETS_PATH + '/test_videos'
