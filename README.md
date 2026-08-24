README - Stability of Gradient-Based Explanations on MNIST

This archive contains the code used for my MSc project. The project tests how much gradient-based explanations change when MNIST images are perturbed. It compares a smoothed CNN using Softplus and average pooling with an unsmoothed CNN using ReLU and max pooling.


CONTENTS OF THE ARCHIVE

mnist_model.py
Defines the smoothed CNN used in the main experiment.

train_mnist.py
Trains the smoothed CNN on MNIST and saves the trained model in the models folder.

mnist_lipschitz_captum_experiment.py
Runs the random perturbation and PGD experiments. It calculates the explanation change S and sensitivity ratio R for Saliency, Integrated Gradients or Input x Gradient.

plot_mnist_results.py
Reads the experiment results and produces the summary CSV tables and plots.

requirements.txt
Lists the Python libraries required to run the code.

AUTHORSHIP_DECLARATION.txt
Contains my signed and dated declaration of authorship.

models/
Contains the saved trained model.

results/
Contains the CSV results and summary tables for the smoothed CNN.

figure/
Contains the figures produced from the smoothed CNN results.

licenses/
Contains the MIT licence for the LeNet-5 code on which the model and training code were based.

unsmoothed/
Contains the comparison experiment using the ReLU and max-pooling CNN. It has its own mnist_model.py, train_mnist.py, mnist_lipschitz_captum_experiment.py and plot_simple.py files. Its model, results and figures are stored in the models, result and figure folders inside this directory.

train/, test/ and data/
Contain MNIST files downloaded by Torchvision. They can be downloaded again automatically if they are not present.


HOW TO RUN THE CODE

1. Open a terminal in the main MNIST folder.

2. Install the required libraries using:

   pip install -r requirements.txt

3. To train the smoothed model, run:

   python train_mnist.py

   This step can be skipped if the trained model is already present in the models folder.

4. To run the experiment, run:

   python mnist_lipschitz_captum_experiment.py

5. To create the plots and summary tables, run:

   python plot_mnist_results.py

The explanation method is selected by changing attr_method in mnist_lipschitz_captum_experiment.py. The available values are saliency, integrated_gradients and input_x_gradient. The outputpath in the same file should use the corresponding method name.

Before running plot_mnist_results.py, METHOD should be changed to the same explanation method. The supplied results are already included, so the experiment does not need to be rerun just to view them.

To run the comparison model, open a terminal in the unsmoothed folder and follow the same steps. Use plot_simple.py instead of plot_mnist_results.py. The full experiments use 500 images and 100 random perturbations per image for each epsilon value, so they may take a long time to finish.


THIRD-PARTY MATERIAL

The CNN model and training code were adapted from ChawDoe's LeNet5-MNIST-PyTorch repository, which is released under the MIT Licence. The licence is included in the licenses folder. PyTorch, Torchvision, Captum, NumPy, pandas and Matplotlib are third-party libraries and are not included in this archive.
