# Neural-Network
Window app to implement and visualize simple feedforward-backpropagation artificial neural network from scratch. Created for the purpose of Artificial Intelligence class on Białystok University of technology. The code itself is pretty bad, as I created this in more or less in 3 days with little sleep and a lot of coffee.
I've also put some testing scripts - they were created when there was no GUI to some functionalities and I wanted to have some logic tested.

Enjoy ;)

Language: Python 3.7
GUI: PyQt5

Main libraries used:
- numpy - all the calculations
- pandas - some basic data formatting and one-hot-encoding
- matplotlib - plotting graphs
- networkx - plottiing the network graphs (with matplotlib)

### Create tab- Creating the net 

 1. Load CSV file with data. 
 3. Adjust layer sizes by writing coma separated integers into the field i.e. "4,6,3" or "4, 6,   7" NOTE: First and last layers are created automatically
 2. Press the create button.

 
 <img src="https://github.com/TotoAfreeca/Neural-Network/blob/master/screenshots/img1.png" />
 
 ### Train tab - Adjusting the learning parameters
 
 1. Adjust learning rate, expected error and no of epochs.
 2. Move to the ERROR tab or initialize weights 
 <b> What's very important - do not try to train on this tab - the networkx engine to plot the network graph is too slow, it takes ages to update the graph after every adjustment of the net, I might remove the button from this tab later </b>
 
 <img src="https://github.com/TotoAfreeca/Neural-Network/blob/master/screenshots/img4.png" />
 
 ### Error tab - Learning the net and visualizing the error
 
 Press train or initialize weigh button
 
 I might add drawing of the test set evaluated error later.
 
 <img src="https://github.com/TotoAfreeca/Neural-Network/blob/master/screenshots/img3.png" />
 
 
 ### Summary tab - Checking the results
 
 After pressing the button, you will see the expected values on top of the predicted values <b> from the test set </b> - I did not set the thresholding to the outputted values so everyone can see the actual output of the net
 
  <img src="https://github.com/TotoAfreeca/Neural-Network/blob/master/screenshots/img2.png" />
 
