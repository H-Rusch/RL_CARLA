# Anwendungen der KI 21-22 - Gruppe 5

## Prerequisites
The program was written with Python 3.7 using the 9.11 version of the CARLA Simulator. All needed modules can be installed running the `run.py` script.

## Running the program
### Training a model
Run `run.py` to install all needed modules and start the training process. The attribute `load_model_name` in `main.py` can be edited to load an existing model. The value of `None` will create new model. 

### Testing a model
Run the `test_model.py` script to test a model on the defined track. By editing the `load_model_name` in this script, a specific model can be tested. Testing the model for overfitting by going around the track backwards can be done by changing the `REVERSE` flag from `False` to `True`.

## Sources
The basic structure for the main loop and training algorithm comes from this tutorial on [pythonprogramming.net](https://pythonprogramming.net/introduction-self-driving-autonomous-cars-carla-python/)
