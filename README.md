# AI Models and Semantic Search

## 1. AI Models are Like Functions
- You give them an input, and they give you a prediction.
- The output is usually a guess about the input, often in the form of scores for different possibilities.

## 2. Hugging Face Models
- Hugging Face provides a large collection of pre-trained AI models.
- Python libraries make it easy to use these models for different tasks.

## 3. Common Uses
- Detecting the emotion in a sentence (sentiment analysis)
- Recognizing objects in images (image classification)
- Generating captions for pictures (image captioning)

## 4. Embeddings
- Embeddings turn text or images into a list of numbers.
- The numbers themselves don’t mean much by themselves.
- Comparing two sets of numbers shows how similar they are.  
  - The closer the numbers, the closer the meaning of the text or image.

## 5. Semantic Search
- We can use embeddings to build a search system.
- The system finds items not just by matching words, but by matching meaning.

# Demo

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
