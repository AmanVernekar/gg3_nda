import numpy as np
from inference import hmm_expected_states, poisson_logpdf
from models_HMM import StepHMM_better, HMM_Ramp_Model

import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(x, x_hat, class_labels=None):
    """
    Compute and plot the confusion matrix for true labels `x` and predicted labels `x_hat`.

    Parameters:
    x (np.ndarray): Array of true labels.
    x_hat (np.ndarray): Array of predicted labels.
    class_labels (list, optional): List of class labels. Defaults to None.

    Returns:
    np.ndarray: The confusion matrix.
    """
    # Determine the number of unique classes
    classes = np.unique(np.concatenate((x, x_hat)))
    num_classes = len(classes)

    # Create a mapping from class labels to indices
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    # Initialize the confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)

    # Populate the confusion matrix
    for true_label, pred_label in zip(x, x_hat):
        true_index = class_to_index[true_label]
        pred_index = class_to_index[pred_label]
        cm[true_index, pred_index] += 1

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)

    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    plt.show()

    return cm


ramp = HMM_Ramp_Model(
    beta=1.2,
    sigma=0.2,
    x0= 0.2,
    K = 50
)

N_trials = 1
T = 100

spikes, xs, rates = ramp.simulate( N_trials, T)

# compute the log likelihood        ll[i,j]= log p(n_t=i| s_t = j)
ll = poisson_logpdf(counts=spikes, lambdas= ramp.Rh * ramp.state_space * ramp.dt)[0,:,:]

posterior_prob, normalizer = hmm_expected_states(ramp.pi,ramp.trans_mtx,ll)
print(np.argmax(posterior_prob,axis=1))

#Compute expectation
expected_xt = posterior_prob @ ramp.state_space
print(f"EM inference on xt: {expected_xt}")
print(f"Ground truth: {xs}")

plot_confusion_matrix(xs,expected_xt)
