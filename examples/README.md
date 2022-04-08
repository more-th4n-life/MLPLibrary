This directory contains Python notebooks that were used to experiment with training and testing models on the assignment data.
* ```Demo-MLP.ipynb``` contains a practical demo of MLPLibrary usage in creating a model using SGD, and another with Adam optimizer.
* ```Iterative-Experimentation-SGD.ipynb` began as an iterative study in applying hyperparameters to an SGD model - however, it
    became obvious in fifth iteration that the relationship between parameters (i.e. perms and combs of these selections are not
    obvious and can produce significantly different results). In applying the Adam optimizer last, we saw potential in exploring this
    optimizer for our final model.
* ```Adam-MLP-Hidden-Layer-Tuning.ipynb``` includes 6 scenario trials for hyperparameter tuning of the # hidden layers (i.e. 3 scenarios for removing a layer and 3 scenarios for adding a layer)
* ```Adam-MLP-Learning-Rate-Tuning.ipynb``` includes hyper-parameter analysis of learning-rate tuning for chosen Adam model
* ```Adam-MLP-Dropout-Tuning.ipynb``` includes hyper-parameter analysis of drouput tuning for chosen Adam model
* ```Adam-MLP-Batch-Norm-Tuning.ipynb``` includes hyper-parameter analysis of batch-norm tuning for chosen Adam model
* ```Adam-MLP-Batch-Size-Tuning.ipynb``` includes hyper-parameter analysis of batch-norm tuning for chosen Adam model
* ```Adam-MLP-Regularization-Tuning.ipynb``` includes hyper-parameter analysis of weight decay tuning for chosen Adam model
* ```Best-Model.ipynb``` includes our final model that was arrived by performing analyses detailed in the notebooks above

