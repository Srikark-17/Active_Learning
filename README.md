# Active Learning Interface

This is a React app that can be used to conduct Active Learning locally for image annotation and train models. It can be used by non-professionals, as well. This is a React app that can be used to conduct Active Learning locally for image annotation and train models. It can be used by non-professionals, as well. Active Learning is an iterative machine learning approach where the model intelligently selects the most informative samples for human labeling, rather than requiring humans to label every piece of data. The system starts with a small set of labeled images, trains an initial model, and then identifies uncertain or ambiguous images that would most improve the model's performance when labeled. This significantly reduces the time and effort needed for data labeling while maintaining high model quality. Users can simply review and label the suggested images through an intuitive interface, and the model continuously learns and improves from this focused feedback. This approach is particularly valuable when working with large image datasets where labeling everything would be impractical.

## Installation

1. Install [Node](https://nodejs.org/en/download/), [Yarn](https://classic.yarnpkg.com/en/docs/install/), [Python](https://www.python.org/downloads/), and [Anaconda](https://www.anaconda.com/products/distribution)
2. Create an environment with conda
   `conda create -n NAME_OF_ENVIRONMENT`
3. Go into the `backend` folder and install the requirements
   `conda install --yes --file requirements.txt`
4. Change directory to the `frontend` folder and run
   `yarn`

## Usage

1. Change directory into the `backend` folder and run
   `python main.py`
2. Switch directory into the `frontend` and run
   `yarn run dev`
3. The website should open up on `localhost:3000`. Now, go to the left-hand column and input the project information. **LABELS ARE AUTOMATICALLY ADDED BY TYPING IN THE TEXT BOX. ONLY CLICK "ADD LABEL" IF YOU NEED TO ENTER A NEW LABEL.**
4. Upload the images/image directory
5. Click "Start Project"
6. Click "Start Training", which is right below the "Start Project" button
7. Now, for each image, select the appropriate label.
8. Click "Submit Lable" after selecting the appropriate label.
9. Repeat steps 7-8 for all images
10. While annotating, it's important to note that there are model predictions below the labels to view the model's performance at predicting the image's label
11. Once the batch is complete, the script will automatically retrieve the progress of the current episode as well as the new batch to review.
12. At the bottom of the left column will be checkpoint controls as well as import/export models. You can save checkpoints and/or export the model after each batch or before reviewing a batch.

## Questions

Contact Srikar on Slack
