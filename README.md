# ai-python-semantic-search

To run the demo, first install the following python packages

```
pip install pyarrow fastparquet
pip install sentence_transformers faiss-cpu
```

Then run 

```
python download_images.py
``` 

to download some sample images from imagenet

Finally run 

```
python search.py
```

for the command-line search engine.

# Exercises

  - Drop your own images into the images folder and test to see if semantic search still works.

  - Experiment with different search phrase and note all the interesting observations.

  - Create an app.py file that uses FastAPI to wrap the search function so it can be deployed as webapp.
