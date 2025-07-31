import numpy as np
from scipy.special import gammaln

def poisson_log_pmf(k, rate):
    """
    k: integer or NumPy array instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    # The log pmf of a Poisson distribution is given by:
    # log_pmf(k; λ) = k * log(λ) - λ - log(k!) as we found in question 1 of the theoretical exercise
    # We can use the scipy special function gammaln to compute log(k!)
    log_p = k * np.log(rate) - rate - gammaln(k + 1)
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return log_p


def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    # the lambda hat expression we found in question 2 of the theoretical exercise is sum(x_i) / n
    # where x_i is the i-th sample and n is the number of samples
    mean = np.mean(samples)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return mean

def possion_confidence_interval(lambda_mle, n, alpha=0.05):
    """
    lambda_mle: an MLE for the rate parameter (lambda) in a Poisson distribution
    n: the number of samples used to estimate lambda_mle
    alpha: the significance level for the confidence interval (typically small value like 0.05)
 
    return: a tuple (lower_bound, upper_bound) representing the confidence interval
    """
    # Use norm.ppf to compute the inverse of the normal CDF
    from scipy.stats import norm
    lower_bound = None
    upper_bound = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    # Calculate the standard error of the mean
    std_error = np.sqrt(lambda_mle / n)
    # Calculate the z-score for the given alpha level
    z_score = norm.ppf(1 - alpha / 2)
    # Calculate the confidence interval
    lower_bound = lambda_mle - z_score * std_error
    upper_bound = lambda_mle + z_score * std_error
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return lower_bound, upper_bound

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    # first we initialize the likelihoods array to be of the same length as rates
    likelihoods = np.zeros(len(rates))
    
    # then, we iterate over the rates and calculate the log likelihood for each one
    for i,rate in enumerate(rates):
        # we use the poisson_log_pmf function to calculate the log likelihood
        # for each sample and sum them up
        likelihoods[i] = np.sum(poisson_log_pmf(samples, rate))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return likelihoods

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.05,
            (0, 1): 0.25,
            (1, 0): 0.25,
            (1, 1): 0.45
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.25,
            (0, 1): 0.05,
            (1, 0): 0.25,
            (1, 1): 0.45
        }  # P(X=x, C=c)

        self.Y_C = {
            (0, 0): 0.25,
            (0, 1): 0.05,
            (1, 0): 0.25,
            (1, 1): 0.45
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.125,
            (0, 0, 1): 0.005,
            (0, 1, 0): 0.125,
            (0, 1, 1): 0.045,
            (1, 0, 0): 0.125,
            (1, 0, 1): 0.045,
            (1, 1, 0): 0.125,
            (1, 1, 1): 0.405,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        # we iterate over all values of x and y and find the first pair where P(x,y) != P(x) * P(y)
        # if no such pair is found, then X and Y are independent, and we return False
        for x in X:
            for y in Y:
                if X_Y[(x, y)] != X[x] * Y[y]:
                    return True
        return False
    
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        
        # we iterate over all values of x, y and c and find the first triplet where P(x,y|c) != P(x|c) * P(y|c)
        # if no such triplet is found, then X and Y are independent given C, and we return True
        for x in X:
            for y in Y:
                for c in C:
                    if abs(X_Y_C[(x, y, c)] - (X_C[(x, c)] * Y_C[(y, c)] / C[c])) > 1e-6: # can fail due to a floating point, so we compare with a small epsilon
                        return False
        return True
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    p = (1 / np.sqrt(2 * np.pi * std**2)) * np.exp(-0.5 * ((x - mean) / std)**2)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates information on the feature-specific
        class conditional distributions for a given class label.
        Each of these distributions is a univariate normal distribution with
        separate parameters (mean and std).
        These distributions are fit to specified training data.
        
        Input
        - dataset: The training dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class label to calculate the class conditionals for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        self.class_value = class_value
        
        # Extract the features for the given class label
        class_data = dataset[dataset[:, -1] == class_value] # select all rows with label class_value
        features = class_data[:, :-1]
        
        # compute mean and std for each feature
        self.means = np.mean(features, axis=0)
        self.stds = np.std(features, axis=0)
        
        # compute prior
        self.prior = class_data.shape[0] / dataset.shape[0] # prior is calculated as number of class occurences / total number of samples
        
        # store the number of features
        self.num_features = features.shape[1]
        # store the number of samples
        self.num_samples = class_data.shape[0]
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        # the prior is already calculated in the constructor
        prior = self.prior
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the feature-specific classc conditionals fitted to the training data.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        # for each feature x_i, we compute the probability density under the normal distribution
        # fitted for the feature within the class, and because of our assumption - 
        # these probabilities are conditionally independent and thus we multiply them and get the likelihood (p(x|y=class_label))
        
        # initialize the likelihood to 1 (because we need to multiply the likelihoods)
        likelihood = 1.0
        # iterate over each feature
        for i in range(self.num_features):
            # calculate the likelihood of the feature given the class label
            likelihood *= normal_pdf(x[i], self.means[i], self.stds[i]) # product of the likelihoods of each feature
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        # we calculate the joint probability as the product of the prior (p(y=c)) and the likelihood (p(x|y=class_label))
        joint_prob = self.get_prior() * self.get_instance_likelihood(x)
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return joint_prob

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class holds a ClassDistribution object (either NaiveNormal or MultiNormal)
        for each of the two class labels (0 and 1). 
        Using these objects it predicts class labels for input instances using the MAP rule.
    
        Input
            - ccd0 : A ClassDistribution object for class label 0.
            - ccd1 : A ClassDistribution object for class label 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        # store the class distributions
        self.distributions = [ccd0, ccd1]
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        # we calculate the a-posteriori probabilities for each class using the joint probabilities
        posterior_0 = self.distributions[0].get_instance_joint_prob(x) # p(x|y=0) * p(y=0)
        posterior_1 = self.distributions[1].get_instance_joint_prob(x) # p(x|y=1) * p(y=1)
        
        pred = 0 if posterior_0 > posterior_1 else 1 # we return the class with the highest posterior probability (MAP rule)
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    
def multi_normal_pdf(x, mean, cov):
    """
    Calculate multivariate normal desnity function under specified mean vector
    and covariance matrix for a given x.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    # we get the dimension of x
    k = len(x)
    
    x_minus_mu = x - mean
    cov_inv = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)

    # calculate the normalization constant
    norm_const = 1.0 / (np.power(2 * np.pi, k / 2) * np.sqrt(det_cov))
    # calculate the exponent part
    exp_part = np.exp(-0.5 * x_minus_mu.T @ cov_inv @ x_minus_mu)
    
    pdf = norm_const * exp_part
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the multivariate normal distribution
        representing the class conditional distribution for a given class label.
        The mean and cov matrix should be computed from a given training data set
        (You can use the numpy function np.cov to compute the sample covarianve matrix).
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class label to calculate the parameters for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        self.class_value = class_value
        # Extract the features for the given class label
        self.class_data = dataset[dataset[:, -1] == class_value] # select all rows with label class_value
        self.features = self.class_data[:, :-1]
        # compute mean and cov matrix for the features
        self.mean = np.mean(self.features, axis=0)
        self.cov = np.cov(self.features, rowvar=False) # rowvar=False means that each column represents a variable, and each row is an observation
        # compute prior
        self.prior = self.class_data.shape[0] / dataset.shape[0]
        # store the number of features
        self.num_features = self.features.shape[1]
        # store the number of samples
        self.num_samples = self.class_data.shape[0]
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        # the prior is already calculated in the constructor
        prior = self.prior
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the multivariate classc conditionals fitted to the training data.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        # we calculate the likelihood of the instance given the class label
        # using the multivariate normal distribution pdf
        likelihood = multi_normal_pdf(x, self.mean, self.cov)
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        # we calculate the joint probability as the product of the prior (p(y=c)) and the likelihood (p(x|y=class_label))
        joint_prob = self.get_prior() * self.get_instance_likelihood(x)
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return joint_prob



def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given MAP classifier on a given test set.
    
    Input
        - test_set: The test data (Numpy array) on which to compute the accuracy. The class label is the last column
        - map_classifier : A MAPClassifier object that predicits the class label from a feature vector.
        
    Ouput
        - Accuracy = #Correctly Classified / number of test samples
    """
    acc = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    # initialize the number of correct predictions
    correctly_classified = 0
    # iterate over each instance in the test set
    for i in range(test_set.shape[0]):
        # get the features and the true label
        x = test_set[i, :-1] # get the features (all columns except the last one)
        true_label = test_set[i, -1]
        # predict the label using the MAP classifier
        predicted_label = map_classifier.predict(x)
        # check if the prediction is correct
        if predicted_label == true_label:
            correctly_classified += 1
    # calculate the accuracy
    acc = correctly_classified / test_set.shape[0] # accuracy calculation as instructed
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return acc

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the probabilites for a discrete naive bayes
        class conditional distribution for a given class label.
        The probabilites of each feature-specific class conditional
        are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class label to calculate the probabilities for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        self.class_value = class_value
        # Extract the features for the given class label
        self.class_data = dataset[dataset[:, -1] == class_value] # select all rows with label class_value
        self.features = self.class_data[:, :-1]
        self.dataset = dataset
        # compute the number of features
        self.num_features = self.features.shape[1]
        # compute the number of samples
        self.num_samples = self.class_data.shape[0]
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        prior = self.num_samples / self.dataset.shape[0] # prior is calculated as number of class occurences / total number of samples
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the product of feature-specific discrete class conidtionals fitted to the training data.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        # we initialize the likelihood to 1 (because we need to multiply the likelihoods)
        likelihood = 1
        
        # for each feature x_i, we compute the probability of the feature given the class label
        for i in range(self.num_features):
            # get the feature value
            feature_value = x[i]
            # get the number of samples for the feature value
            num_samples = np.sum(self.features[:, i] == feature_value)
            # get the number of samples for the class label
            num_class_samples = self.num_samples
            
            # calculate the likelihood using laplace smoothing (we add 1 to the numerator and the number of unique values to the denominator)
            likelihood *= (num_samples + 1) / (num_class_samples + len(np.unique(self.dataset[:, i])))
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        # we calculate the joint probability as the product of the prior (p(y=c)) and the likelihood (p(x|y=class_label))
        joint_prob = self.get_prior() * self.get_instance_likelihood(x)
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return joint_prob
