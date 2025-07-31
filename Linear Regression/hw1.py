# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform Standardization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The Standardized input data.
    - y: The Standardized true labels.
    """
    ###########################################################################
    # TODO: Implement the normalization function.                             #
    ###########################################################################

    # Standardization formula
    X = (X - X.mean()) / X.std() 
    y = (y - y.mean()) / y.std()
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (n instances over p features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (n instances over p+1).
    """
    ###########################################################################
    # TODO: Implement the bias trick by adding a column of ones to the data.  #
    ###########################################################################
    
    # we create the bias trick by adding a column of ones to the input data
    ones = np.ones(X.shape[0])
    X = np.column_stack((ones, X))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X

def compute_loss(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the loss associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the loss.
    ###########################################################################
    # TODO: Implement the MSE loss function.                                  #
    ###########################################################################
    
    # We compute the average of the squared difference between the predicted and actual values
    J = np.mean(((X @ theta) - y) ** 2)
    J = J / 2
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J

def gradient_descent(X, y, theta, eta, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: The parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the loss value in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    
    t = 0 # Iteration counter
    while t < num_iters:
        J_history.append(compute_loss(X, y, theta)) # Compute the loss and append it to the history
        error =  (X @ theta) - y # Compute prediction
        
        # Gradient computation and weights update
        gradient = (X.T @ error) / len(X)
        theta -= eta * gradient
        
        t += 1 
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################
    
    # We compute the pseudoinverse of the input data and use it to compute the optimal parameters
    pinv_theta = np.linalg.inv(X.T @ X) @ X.T @ y
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta

def gradient_descent_stop_condition(X, y, theta, eta, max_iter, epsilon=1e-8):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than epsilon. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: The parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - max_iter: The maximum number of iterations.
    - epsilon: The threshold for the improvement of the loss value.
    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the loss value in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent with stop condition optimization algorithm.  #
    ###########################################################################
    
    prev_loss = compute_loss(X, y, theta)
    J_history.append(compute_loss(X, y, theta)) # Compute the loss and append it to the history

    t = 0
    while t < max_iter:
        error = (X @ theta ) - y # Compute error

        # Compute the gradient and update the weights
        gradient = X.T @ error / len(X)
        theta -= eta * gradient
        
        loss = compute_loss(X, y, theta)
        if abs(loss - prev_loss) < epsilon: # check if the improvement is smaller than epsilon
            J_history.append(loss)
            break
        prev_loss = loss # Update the previous loss
        J_history.append(loss)
        t += 1
            
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def find_best_learning_rate(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of eta and train a model using 
    the training dataset. Maintain a python dictionary with eta as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - eta_dict: A python dictionary - {eta_value : validation_loss}
    """
    
    etas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    eta_dict = {} # {eta_value: validation_loss}
    ###########################################################################
    # TODO: Implement the function and find the best eta value.             #
    ###########################################################################
    
    theta = np.ones(X_train.shape[1])
    for eta in etas: # Iterate over the provided values of eta
        theta, J_history = gradient_descent(X_train, y_train, theta, eta, iterations) # Train a model using the training dataset
        if len(J_history) < 10: # Check if the loss history is too short
            continue 
        loss = compute_loss(X_val, y_val, theta) # Compute the loss on the validation set
        eta_dict[eta] = loss
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return eta_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_eta, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_eta: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    #####c######################################################################
    # TODO: Implement the function and find the best eta value.             #
    ###########################################################################
    
    features = X_train.shape[1] # get the number of features
    X_train = apply_bias_trick(X_train)
    X_val = apply_bias_trick(X_val)
    selected_features.append(0) # Add the bias feature to the list of selected features

    for i in range(5):
        best_loss = float('inf')
        best_feature = -1
        for j in range(1,(features)):
            if j in selected_features: # Skip the already selected features
                continue 
            
            theta = np.ones(i + 2) # Initialize the parameters to ones
            temp_features = selected_features + [j] # Create a temporary list of features
            X_train_temp = X_train[:, temp_features]
            X_val_temp = X_val[:, temp_features]
            theta, _ = gradient_descent_stop_condition(X_train_temp, y_train, theta, best_eta, iterations) # Train a model using the training dataset
            loss = compute_loss(X_val_temp, y_val, theta)
            
            if loss < best_loss:
                best_loss = loss
                best_feature = j # Update the best feature if the loss is lower than the previous best loss

        selected_features.append(best_feature) # Add the best feature to the list of selected features
    selected_features.remove(0) # Remove the bias feature from the list of selected features
    for i in range(len(selected_features)):
        selected_features[i] -= 1
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (n instances over p features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    
    columns = df_poly.columns
    
    poly_features = {f"{i}^2": df_poly[i] ** 2 for i in columns} # Create the square features

    column_combos = {f"{col1}*{col2}": df_poly[col1] * df_poly[col2] for i, col1 in enumerate(columns) for col2 in columns[i + 1:]} 
    # Create the pairwise interaction features
    poly_features = {**poly_features, **column_combos} #Ccombine the square features and interaction features
    # Combine the original features, square features, and interaction features
    df_poly = pd.concat([df_poly, pd.DataFrame(poly_features, index=df.index)], axis=1)
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly