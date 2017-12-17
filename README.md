# Price prediction of apartments on Hemnet using agency descriptions

## Regression

To run the regression, call the run function with the vocabulary file and the prediction data file in Python code:

```
run('output_new.json', 'sthlm_format.json')
```

The run function can use either tf-idf or Word2Vec for word embedding.
