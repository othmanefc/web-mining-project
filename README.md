# Web Mining Project

This project was made by team BofBof composed of Othmane Hassani & Antoine Lajoinie from TSE M2 Stat-Ecox.

## Architecture

For this project, all the components have been split in different modules for better readability and comprehensiveness. There are 2 main notebooks, to launch the whole project: One for the data transformation, and the second one for the modelling part. 

`preprocess.py` : contains the first preprocessor that is an object which outputs some features such as time and if the author is a root one.

`text_preprocess.py` : contains the text preprocessor, it will extract some text features from the comments such as the number of nouns or the subjectivity of the text.

`text_clustering.py` : contains the text clustering part, it will cluster the comments using FAISS algorithm on sentence embeddings from a BERT specialized on sentence embeddings from the sentence_transformers library.

`network_preprocesser.py` : contains the network features extraction, since the dataset is social media based. It should be interesting to extract some specialized network features.

## Getting started

Since the project is using the FAISS algorithm, it is highly recommended to run this project inside a conda environment

To install FAISS, do:

``` 

$ conda install -c pytorch faiss-gpu #or faiss-cpu if no gpu with CUDA
```

Finally install the requirements with:

``` 

$ pip3 install -r requirements.txt
```

## How to use it

You simply need to go the `data_transformation.ipynb` , execute the cells, 
then go to the `modelling.ipynb` , everything should be explained, and very verbose.

## Models

The main model is a LightGBM which is a very good architecture for Kaggle winning kernels. As its name stands, it is very fast and lightweight compared to a model like XGBoost or even a RandomForest with having same or even better results.
We first tried ensemble modelling, but the results weren't much better so we stuck with a simple LightGBM with cross-validation.

## Intuitions

After many experiments, we understood that it is very hard to predict score from the comment on itself. Some comments can be very good but will just be forgotten because posted at the wrong time or in the wrong comment chain. The network features are very important since a comment will likely have a higher score if in a chain that has comments with very high scores, that's just how social media works. If you have cool friends, you are very likely to become cool too, regardless of your content.
