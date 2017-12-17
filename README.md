# Price prediction of apartments on Hemnet using agency descriptions

Project in the course DD2418 Language Engineering. All files described below can be runed seperatly. 


## Regression.py

To run the regression, call the run function with the vocabulary file and the prediction data file in Python code:

```
run('output_new.json', 'sthlm_format.json')
```

The run function can use either tf-idf or Word2Vec for word embedding.

## Classification.ipynb

The classification is formated in a Jupyter Notebook file, classification.ipynb. Run all cells for classification using tf-idf as word embedding. Prints average metrics for the classfification.

## Visualization.ipynb

Notebook for vizualization of data before regression and classification. Run cells in order. This file was used for gaining an insight of how the data looked like to improve the predictions. 

## word2vec_visualization

To visulize the word2vec network created for the task. The wordtovec model is converted into json file and displayed using html and javascript. Create a HTML - server and display the index.html file. Write a word connected with housing objects to recive the closest words in cosine-sense. 


To run simple HTTP server: 

```
python -m http.server 8888
```

- Main files and folders:
	+ backend
		+ Convert_to_JSON
			scripts for converting word2vec models to JSON files.
	+ frontend<br>
		+ data
			all data for searching word and vizualizing them.
		+ js
			D3.js library (visualization javascript library).
		+ indexl.html
			the main web page.

Inspired by github project by Van-thuy Phi and Taishi Ikeda


