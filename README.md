# Predicting-gene-expression-of-millions-yeast-DNA-sequences-using-Deep-Learning
This project is about pre-processing millions of DNA sequences, building a Convolutional Neural Network using Tensorflow, and interpreting its predictions by conducting a series of computational experiments.

File 1 - CNN: 
I used uniformed and shuffled train data (around 440 000 sequences) to process it (elongate, onehotencode...) and pass it to the model. I created the model arhcitecture and all the details (learning rate, loss function ...). I trained the model and tested, plotting its predictions vs actual values and testing model performance.

File 2 - Uniform data:
Here I was taking original given data and uniforming them by setting a certain threshold - certain number of sequences for each bin, to mitigate low count of sequences in low bins.

File 3 - DNA seq SHAP: 
Here is where I did all the experiments. I took the best model, took the uniformed shuffled data, and conducted positioning experiments, spacing experiments, individiual bases experiments, experiment where I keep both motifs and then delete one and the other and so on. Also, shap importances are calculated here and corresponding logos plotted.

File 4 - Modisco:
I started generating h5py file

File 5 - Modisco file:
I took Charle's example and generated the final h5py file to give to Modisco.

File 6 - Kmer Analyses:
Just took the file we wrote in the shell and extracted the correlations for all the fivemers we tested. Looked at the outlier with highest correlation and ti was reb1 fragment.

File 7 - DNA seq:
Did some preliminary data analyses. Checked if there is correlation between AT or GC content and expression, lengths of a sequence and expression, having and not having reb1 or rap1 motifs and expression.


