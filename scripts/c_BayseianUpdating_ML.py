import numpy as np
import os


def bayesian_updating(model_path, num_classes, prior_prob=0.5):
    """
    Perform Bayesian updating on pixel-wise class probabilities.

    Args:
    - model_files (list of str): List of file paths to the model prediction .npy files.
    - num_classes (int): Number of classes. Default is 2 for binary classification.
    - prior_prob (float): Initial probability for each class. Default is 0.5 for binary.

    Returns:
    - updated_probs (numpy.ndarray): Array of updated probabilities after Bayesian updating.
    """
    # Initialize prior probabilities
    # Assume the first model file to get the shape of the probability arrays
    shape = model_path[0].shape[:-1]  # Assuming last dimension is class
    if num_classes == 2:
        # For binary classification, we only need one probability per pixel, as the other is implied.
        # prior = np.full(shape, prior_prob)  # 'shape' already excludes the class dimension.
        prior = np.full(shape + (2,), prior_prob)
    else:
        # For multiclass, we explicitly add the class dimension back to 'shape'.
        prior = np.full(shape + (num_classes,), 1.0 / num_classes)

    # Iterate over each model's predictions to update the prior
    for model_pred  in model_path:

        if num_classes == 2:  # Binary classification
            """
            transforms the prior probability of belonging to Class 1 (for example) into the odds of belonging to Class 1 versus not belonging to Class 1 (Class 2 in binary classification)
            """
            odds_prior = prior / (1 - prior)
            # similar transform
            odds_likelihood = model_pred / (1 - model_pred)

            """the Bayesian updating is performed in the space of odds rather than probabilities. """
            updated_odds = odds_prior * odds_likelihood
            # Convert odds back to probability
            updated_probs = updated_odds / (1 + updated_odds)

            # prior = updated_probs  # Set the updated probs as the new prior for the next iteration
            prior = np.where(updated_probs > prior, updated_probs, prior)  # Update only if the new prob is higher

        else:  # Multiclass correction

            # prior *= model_pred
            # prior = normalize_probs(prior)  # Normalize to ensure probabilities sum to 1 across classes

            normalization_factor = np.sum(model_pred * prior, axis=-1, keepdims=True)

            # prior = model_pred * prior / normalization_factor
            updated_probs = model_pred * prior / normalization_factor
            for i in range(num_classes):
                prior[:, :, i] = np.where(updated_probs[:, :, i] > prior[:, :, i], updated_probs[:, :, i],
                                          prior[:, :, i])

    return prior

