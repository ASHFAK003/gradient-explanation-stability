"""
This script runs the Liptchitz-style perturbation on gradient experiment MNIST part of the project.
We want to test how much our gradient-based explanation changes if we perturb the input image, while keeping model prediction the same.
We used the adapted classifier trained for this project: projectcodes/MNIST/models/mnist_lenet5_state_dict.pt
The script loads this trained CNN classifier, then it:
    1. Loads MNIST test images.
    2. Keeps only correctly classified images.
    3. Computes an attribution/explanation for the original image.
    4. Samples random perturbations from an L2 ball.
    5. Clips perturbed images to valid pixel range [0, 1].
    6. Keeps only perturbations that preserve the predicted class.
    7. Computes the explanation again on the perturbed image.
    8. Computes S and R metrics.
We are computing several key metrics:
    1. Explanation change S: S = || explanation(x') - explanation(x) ||_2
    2. Sensitivity Ratio R: R = S / || x' - x ||_2
We use Captum to compute our attribution method:Saliency,Integrated Gradient,InputXGradient."""

# We import the pre-trained model from the local mnist_model.py file.
# This gives access to the Model class so that the trained weights can be loaded.
from mnist_model import Model
import torch #we need it for loading model weights, tensor operations, and model prediction.
import numpy as np #we need it for numerical operations (in this work specifically for random perturbation sampling and norm calculations.)
import pandas as pd #we need it to store the results as a csv file.
from torchvision.datasets import mnist #this can make us access the MNIST dataset.
from torchvision.transforms import ToTensor #we want to convert MNIST images into tensor form with values in [0,1]
from torch.utils.data import DataLoader #this will let us loop through the MNIST dataset image by image
from captum.attr import Saliency,IntegratedGradients,InputXGradient #we import captum attribution method
from captum.robust import PGD #Captum's built-in Projected Gradient Descent adversarial attack. Documentation source: https://captum.ai/api/robust.html
#settings
mdpath = "models/mnist_lenet5_softplus_state_dict.pt" #path to the saved trained model weights.
outputpath = "results/mnist_lipschitz_metrics_saliency.csv" #path where the final experiment results CSV will be saved.
eps = [0.1, 0.25, 0.5, 0.75, 1.0,2,4,6,8,10,12,14,16] #lists of L2 perturbations, which controls the radius of the L2 ball used for random perturbation.
n_images = 500 #number of correctly classified test images to be analyzed.
n_samples = 100 #number of random perturbations sampled per image per epsilon.
n_attack_steps = 10 #number of PGD gradient steps for the adversarial (worst-case) attack (captum.robust.PGD).
attr_method = "saliency" #Attribution method selected for this run:
#we can select between "saliency","integrated_gradients" and "input_x_gradient"
np.random.seed(42) #we set random perturbations to be reproducible for comparison because we are comparing results across 3 different attribution methods
torch.manual_seed(42) #we set PyTorch-related randomness to be reproducible for same reason.


def load_trained_model():
    model = Model() #we create an untrained instance of the model architecture
    weights =torch.load(mdpath, map_location="cpu") #we load the saved weights from mdpath (model path) and we ensure to use cpu (because my laptop only supports cpu)
    model.load_state_dict(weights) #we put the saved weights into the model architecture
    model.eval() #because we don't want the model to be trained we use evaluation mode.
    return model

def load_mnist_test_loader():
    """
    This function loads the MNIST dataset and wraps it in DataLoader. We set root = "./test" because we want the data to be stored
    inside the test folder instead of the training set since the model has already been trained.
    transform=ToTensor() converts each image into a tensor in [0, 1] and download=True downloads MNIST if it is not already present, else 
    it doesn't download it. We want to ultimately evaluate explanation robustness on unseen images.
    """
    test_data =mnist.MNIST(root="./test",train=False,transform=ToTensor(), download=True) #(PyTorch documentation: https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)
    test_loader= DataLoader(test_data ,batch_size=1, shuffle=False) # Create a DataLoader for the test dataset.
    # batch_size=1 because we want one image per loop because saliency is computed per image.
    # shuffle=False to keep the test images in fixed order so that our experiment is reproducible.
    #Pytorch dataloader documentation: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html.
    return test_loader

def predict_digit(model, image_tensor): #predicts the class for one MNIST image with argument model being the trained classifier
    #and image_tensor being the MNIST image tensor
    with torch.no_grad():#predictions do not require gradients 
        #and torch.no_grad() makes the forward pass faster and uses less memory.(source:https://docs.pytorch.org/docs/2.12/generated/torch.no_grad.html)
        scores = model(image_tensor)#we pass image through the model. This variable stores the raw class scores for the 10 digit classes.
        predicted_class = torch.argmax(scores, dim=1).item()#Pick the class with the highest score.(source:https://docs.pytorch.org/docs/2.12/generated/torch.argmax.html)
        #dim=1 means we are taking argmax across the class dimension.
        #.item() converts this single-value tensor into a normal integer.
    return predicted_class #our output is an integer class label from 0 to 9

def make_attribution_method(model, method_name): #this creates the captum attribution object which takes as input the trained classifier
                                                 #and also the specific string associated to the attribution method to be used for analysis.

    """
        Note: The supported attribution method strings which we are interested for this analysis are:
        "saliency"
        "integrated_gradients"
        "input_x_gradient"
    """
    # If statements to create Captum objects corresponding to the specific chosen attribution method.
    if method_name == "saliency":
        return Saliency(model)
    if method_name == "integrated_gradients":
        return IntegratedGradients(model)
    if method_name == "input_x_gradient":
        return InputXGradient(model)

def compute_attribution(attribution_method,image_tensor, class_number, method_name): #this function computes 
    #an explanation map for one image.
    #and it takes as input attribution_method (captum attribution object), 
    # image_tensor (the image to explain), class_number (target class which
    #explanation has been computed), method_name (the name of our explanation method) 
    #and this function will output an explanation map as Numpy array.
    if method_name == "saliency":
        attribution_tensor = attribution_method.attribute(inputs=image_tensor,target=class_number,abs=False) #we set abs=False because we want to preserve signs. Documentation source: https://captum.ai/api/saliency.html
    elif method_name== "integrated_gradients":
        attribution_tensor = attribution_method.attribute(inputs=image_tensor, target=class_number,) #documentation source: https://captum.ai/api/integrated_gradients.html
    elif method_name == "input_x_gradient":
        attribution_tensor = attribution_method.attribute(inputs=image_tensor,target=class_number) #documentation source: https://captum.ai/api/input_x_gradient.html
    attribution_numpy = attribution_tensor.detach().numpy().copy()
    return attribution_numpy #our attribution map as a Numpy array.

def sample_l2_ball_like_image(image_numpy, epsilon):  #This function is an L2-ball sampler which 
#follows Voelker n-ball sampling idea summarised in Section 2.2 of the dissertation report which consists of:
#sampling random Gaussian vector
#normalising it to get a random unit direction
#sampling radius using U^(1/n)
#multiplying it by epsilon
#reshaping it back to image shape to give ||delta||_2 <= epsilon
    image_shape=image_numpy.shape #stores the shape of the input image
    n_dimensions=image_numpy.size #Number of total dimensions in the flattened image
    random_vector = np.random.normal(size=n_dimensions)#This samples a random Gaussian vector in n_dimensions to give a random direction.
    vector_norm =np.linalg.norm(random_vector, ord=2) #L2 norm of the random vector.
    #if vector_norm <= 1e-12:
        #vector_norm = 1e-12 #for safety, so we avoid zero division errors.
    direction = random_vector / vector_norm #normalize to unit direction
    u = np.random.uniform(0.0, 1.0) #uniform random sample from [0,1]
    radius = epsilon*(u**(1.0 / n_dimensions)) #radius for uniform sampling inside the  ball
    delta_flat = radius * direction #scale the direction by the sampled radius.
    delta = delta_flat.reshape(image_shape)#reshape the flat perturbation back to the original image shape.
    return delta #sampled perturbation

def adversarial_pgd_metrics(model,attribution_method,image_tensor,original_class,original_attribution,epsilon,method_name,n_steps): #computes the
    #worst-case (adversarial) explanation change, as the counterpart to the random average-case sampling above.
    #It uses Captum's Projected Gradient Descent (captum.robust.PGD), an iterative adversarial attack that crafts an input which maximises the
    #model's classification loss (i.e. maximises classification error) within an L2 ball of radius epsilon, while clipping outputs to the valid
    #pixel range [0,1] through lower_bound/upper_bound. This realises the "carefully crafted inputs that maximise classification error" worst-case.
    #Documentation source: https://captum.ai/api/robust.html . Captum is released by Meta under the BSD-3-Clause licence
    #(https://github.com/meta-pytorch/captum/blob/master/LICENSE).
    pgd = PGD(model, loss_func=torch.nn.CrossEntropyLoss(reduction="none"), lower_bound=0.0, upper_bound=1.0) #PGD attacker bounded to the valid image range [0,1]
    step_size = epsilon / 4.0 #per-step size; a few steps of this size traverse the epsilon-ball before projection
    adversarial_image_tensor = pgd.perturb(inputs=image_tensor, radius=epsilon, step_size=step_size, step_num=n_steps,
        target=torch.tensor([original_class]), norm="L2") #adversarial example maximising classification error within the L2 ball of radius epsilon
    adversarial_attribution = compute_attribution(attribution_method=attribution_method, image_tensor=adversarial_image_tensor,
        class_number=original_class, method_name=method_name) #explanation of the adversarial image, kept for the original class for a fair comparison
    attribution_difference = adversarial_attribution - original_attribution #explanation change under the adversarial perturbation
    S_adv = float(np.linalg.norm(attribution_difference.ravel(), ord=2)) #worst-case explanation change
    actual_delta = adversarial_image_tensor.detach().numpy() - image_tensor.detach().numpy() #effective adversarial perturbation
    delta_norm = float(np.linalg.norm(actual_delta.ravel(), ord=2)) #L2 norm of the adversarial perturbation
    R_adv = S_adv / delta_norm if delta_norm > 1e-12 else float("nan") #worst-case sensitivity ratio
    adv_flipped = bool(predict_digit(model, adversarial_image_tensor) != original_class) #did the attack change the predicted class?
    return S_adv, R_adv, adv_flipped #worst-case explanation change, worst-case sensitivity ratio, and prediction-flip flag

def analyse_one_image_one_epsilon(model,attribution_method,image_tensor,epsilon,method_name):
    """
    This functions analyses one image under one epsilon value by getting the original predicted class, computing the original attribution
    map, sampling n_samples perturbations, applying clipping to each of them and keep only the ones which preserves predictions.
    Then it computes the perturbed attribution maps and the key metrics S and R and then it returns the summary statistics.
    Our inputs are:
        model: trained classifier
        attribution_method: Captum attribution object
        image_tensor: one MNIST image
        epsilon: L2 perturbation radius
        method_name: string name associated to the attribution method
    The function outputs a dictionary of metrics for this image and epsilon
    """
    original_class = predict_digit(model, image_tensor) #predicts original class of the image.
    original_attribution = compute_attribution(attribution_method=attribution_method,image_tensor=image_tensor,class_number=original_class,method_name=method_name) #computes the original explanation map for the original clas
    original_image_numpy = image_tensor.numpy().copy()# Convert the original image tensor to a NumPy array as perturbation sampling and clipping are done in NumPy.
    #Empty lists for storing our metrics R and S
    S_values = []
    R_values = []

    #we repeat the perturbation n_samples times
    for sample_number in range(n_samples):#sample a random perturbation from the L2 ball with radius epsilon.
        delta = sample_l2_ball_like_image(image_numpy=original_image_numpy,epsilon=epsilon)#add the perturbation to the original image.
        perturbed_image_numpy=original_image_numpy+delta#perturbing the original image
        perturbed_image_numpy=np.clip(perturbed_image_numpy,0.0,1.0)#clipping pixels to appropriate values [0,1]
        perturbed_image_tensor=torch.tensor(perturbed_image_numpy,dtype=torch.float32) #perturbed NumPy image converted back into a PyTorch tensor.
        new_class = predict_digit(model, perturbed_image_tensor)#predicts the class of the perturbed image
        if new_class != original_class:
            continue  # prediction preserving constraint: class change perturbations rejected
        perturbed_attribution=compute_attribution(attribution_method=attribution_method, image_tensor=perturbed_image_tensor,
        class_number=original_class,method_name=method_name)#attribution map for the perturbed image. Target class stays the 
        #original class so that explanations are compared for the same output class.
        attribution_difference = perturbed_attribution - original_attribution#explanation change.
        S = np.linalg.norm(attribution_difference.ravel(),ord=2) #total explanation change (L2 norm of explanation change)
        actual_delta = perturbed_image_numpy - original_image_numpy #effective perturbation after clipping
        delta_norm = np.linalg.norm(actual_delta.ravel(),ord=2) #L2 norm of effective perturbation
        if delta_norm <= 1e-12:
            continue #avoids zero division errors
        R = S / delta_norm #computes sensitivity score, R

        #store the accepted S and R values in the list.
        S_values.append(S)
        R_values.append(R)

    n_accepted = len(S_values)#number of accepted perturbations

    #if no random perturbation was accepted for this image at this epsilon (e.g. large epsilon where every perturbation flips the
    #prediction), the statistics are undefined: we record NaN instead of calling np.mean/np.max on an empty list (which crashes).
    if n_accepted > 0:
        mean_S = float(np.mean(S_values))
        max_S = float(np.max(S_values))
        mean_R = float(np.mean(R_values))
        max_R = float(np.max(R_values))
    else:
        mean_S = max_S = mean_R = max_R = float("nan")

    #adversarial (worst-case) probe: craft an adversarial input with Captum PGD and measure the explanation change, to compare against the
    #random (average-case) probe above. This works for every attribution method because it attacks the classifier, not the explanation.
    S_adv, R_adv, adv_flipped = adversarial_pgd_metrics(model=model,attribution_method=attribution_method,image_tensor=image_tensor,
        original_class=original_class,original_attribution=original_attribution,epsilon=epsilon,method_name=method_name,n_steps=n_attack_steps)

    #summary statistics
    return {"epsilon":epsilon,"n_attempted":n_samples,"n_accepted":n_accepted,"acceptance_rate": n_accepted / n_samples,"mean_S":mean_S,"max_S":max_S,"mean_R":mean_R,"max_R":max_R,"S_adv":S_adv,"R_adv":R_adv,"adv_flipped":adv_flipped,}

def main():
    """This is our main experiment function because it controls the full experiment as it loads the trained model nd the test set.
    creates the selected attribution method and it loops thought correctly classified test umages and runs perturbation experiments for every 
    epsilon and then it saves the final results as a CSV.
    """
    model = load_trained_model() #loads the trained classifier.
    test_loader = load_mnist_test_loader() #do the same for the MNIST dataset
    attribution_method = make_attribution_method(model=model,method_name=attr_method) #Captum attirbution object
    rows = [] #we store result row in this list before converting to a DataFrame.
    used_images = 0 #counter for correctly classified images used
    for image_index, (image_tensor, true_label_tensor) in enumerate(test_loader):#loop through the MNIST test set one image at a time.
        true_label = true_label_tensor.item() #extract true label as an integer.
        predicted_label = predict_digit(model, image_tensor) # Predict the model's class for this image
        if predicted_label != true_label:
            continue
        used_images += 1 #ensures we only use correctly classified image and skip to next image otherwise, and it updates image coungter accordingly.
        print("Processing image", used_images, "out of", n_images) #print progress so I know the script is still running.
        for epsilon in eps: #run the experiment for every epsilon values
            result = analyse_one_image_one_epsilon(model=model,attribution_method=attribution_method,image_tensor=image_tensor,
epsilon=epsilon,method_name=attr_method) #analyses this one image at this one epsilon.
            #we add various rows to the result row
            result["image_index"] = image_index
            result["true_label"] = true_label
            result["predicted_label"] = predicted_label
            result["attribution_method"] = attr_method
            rows.append(result) #saves result row
        if used_images >= n_images:
            break # Stop when the n_images number of correctly classified images is reached.
    results_df = pd.DataFrame(rows) #convert to pandas dataframe
    column_order = ["image_index","true_label","predicted_label","attribution_method","epsilon","n_attempted","n_accepted",
        "acceptance_rate","mean_S","max_S","mean_R","max_R","S_adv","R_adv","adv_flipped"] #final column order
    results_df = results_df[column_order] #reorder dataframe columns according to final column order
    results_df.to_csv(outputpath, index=False) #save to csv into our results folder

#If this file is run directly it execute the main experiment (the latest defined function)
if __name__ == "__main__":
    main()