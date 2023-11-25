# **Summary**

This repo contains a serie of utils that are utilized as part of another project which has as goal to generate short videos of footbal actions around action spoting.

The utils here contained provide the next 3 utilities.

## 1.- Input to annotator
A dataset generator from Soccernet of short videos, using the long videos from SoccerNet, but shortened aroung specified actions.
The short videos can be generated aroung any provided action of the 17 that SoccerNet support, selecting any of the annots of the ActionSpotting data,
and providing the expected padding.
This short videos are then concatenated, allowing to be utilized as input data for any futher annotations.

## 2.- Transcript to speech
For any given video, we extract speech audio and convert to text. This allows furher processing of video short cuts generation, per instance, to avoiud cuting in  the middle of speechs.

## 3.- Soccer frame classification
We train a footbal frame classifier, by finetunning a Resnet50 pretrained network with Imagenet.
The classifier distinghises between the next class frames: banquillo, trainer, general, medium, short, celebration, tribuna.
We just 500 anotated frames, we get 98% training and 92% test accuracy.
Ray tune (https://docs.ray.io/en/latest/tune/index.html) has been utilized to investigate per hyperparamenters selection.
Testing has been done selecting between learning_rates, batch_size, and weight_secays.


# **Installation instructions**

1. Install a conda enviroment with Python version 3.9.6 or higher:
   conda create -n your_env python=3.9.6
2. Activate the enviroment:
   conda activate your_env
3. Clone the github repository
   git clone https://github.com/username/repo-name.git
   cd repo-name
4. Install the package from setup.py file which resides in the root of the repository.
   python setup.py install



# Datasets requirements

![Soccernet dataset in local storage](images/r1.jpg)

Figure shows how the footballceleb-3 dataset looks like ince created

![footballceleb-3 dataset in local storage](images/r2.jpg)

Figure shows the steps followed to generate the footbalceleb-3 dataset from SoccetNet.
![Steps involved in generating footballceleb-3](images/r3.jpg)

# Console options

The console options are as follow:

* dataset: using soccernet dataset (it is a pre-requisite to have this dataset downloaded and installed), allows to select a given set of annotations as folder of such dataset, in between Datasets/soccer/actionspotting/[england_epl, europe_uefa-champios-league, france_ligue-1, germany_bundesliga, italy_serie-a, spain_laliga](default france_ligue-1), and between 17 classes (defaul 'Goal'), to generate a number, provided as argument (num_videos), of short videos around the provided action, with a timestamp provided as argument (padding). Finally, it concats all generated videos in a single one, that can be utilized for posteior processing (for instance, to generate a classification dataset using Roboflow -  https://roboflow.com/)
  * Parameters are:
    * output_path: relative destination folder for the generated and concat videos.
      * Default: f"videos_{datetime.now()}
    * annots: annotations utilized to generate the output videos. Must be one of the list: [england_epl, europe_uefa-champios-league, france_ligue-1, germany_bundesliga, italy_serie-a, spain_laliga]Default: france_ligue-1task: the soccernet task.
      * Default: actionspotting
    * padding: mid duration of the videos around the action.
      * Default: 20 seconds
    * action: selected action
      * Default: Goal
    * num_videos: number of videos to generate
      * Default: 20
    * log: logging level.
      * Default: INFO

- transcript: Takes an mp4 video file and provides the transcripted text of the audio speech.
  - Parameters are:
    - video_fn: absolute path of the input video.
    - output_path relative output path for the transcript.
    - log: logging level.
      - Default: INFO
- trainer: train a classifier using Resnet50 pretrained with Imagenet to classify in between 7 classes: ['banquillo', 'celebration', 'general', 'medium', 'short', 'trainer', 'tribuna']. It is a requirement to have the dataset 'footballcelebra' downladed and installed.
  - Parameters are:
    - epochs: number of epochs.
      - Default: 5.
    - batch_size: batch size
      - Default: 4
    - learning_rate: learning rate, SGD optimizer
      - Default: 0.001
    - checkpoint_frequency: how often is a checkpoint generated.
      - Default: 1 epoch
    - log: logging level.
      - Default: INFO
- predict: allows to predict the type of footbal frame in between 7 classes.
  - Parameters are:
    - input: input image file as jpeg.
    - plot: if plot must be generated of the image, with its predicted class and score.
      - Default: True
  - log: logging level.
    - Default: INFO
- tunner: runs ray to the trainer to select in between the best hyperparameters. Hyperparameters are set by code, so if you want to change them, you need to do it in code.
  - Parameters are:trials: number of trials in between the hardcoded hyperparameters.
    - Default: 10
  - epochs: maximum number of epochs to run per trial before stop.
    - Defaut: 30
  - log: logging level.
    - Default: INFO

# Transcript

![transcript](images/transcript.jpg)

# Example outputs

A single image predict (from command line)
![single_image_predict](images/p1.png)

A multiple image predict (from test function)
![multiple_image_predict](images/p2.png)

# Hyperparameters tunning with ray

![img.png](images/ct1.jpg)
![img.png](images/ct2.jpg)
![img.png](images/ct3.jpg)
