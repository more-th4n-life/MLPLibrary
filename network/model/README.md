## Model

The 'model' directory contains models built using MLPLibrary. 
* These models are saved:
    * Automatically during training (if new minimum val loss is found); or
    * Directly by using the save_model() method within the Net class. 
* To load a model, simply use the load_model() class method in the Net class. 

After loading in a model, you can resume training of a prior-trained model or use it to make predictions. 
