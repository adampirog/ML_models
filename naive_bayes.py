import scipy
import numpy as np
import pandas as pd

class GaussianNaiveBayes():

    def __init__(self, var_smoothing=1e-09):

        #mean
        self.theta_ = None
        #variance
        self.sigma_ = None

        #variance smoothing parameters
        self.epsilon_ = None
        self._var_smoothing = var_smoothing

        self.classes_ = None
        self.class_log_prior_ = None

        self._fitted = False

    def _set_prior(self, features, target):

        # log for numeric stability
        self.class_log_prior_ = (np.log(features.groupby(target).count().iloc[:, 0].to_numpy())
                                 - np.log(features.shape[0]))

    def _set_gaussian_parameters(self, features, target):

        self.theta_ = features.groupby(target).mean().to_numpy()
        self.sigma_ = features.groupby(target).var(ddof=0).to_numpy()
        
        #variance smoothing for numeric stability
        self.epsilon_ = self._var_smoothing * np.var(features, axis=0).max()
        self.sigma_ += self.epsilon_


    # logarithm pdf for set of features, log for numeric stability
    def _get_log_prob(self, class_index, features):

        mean = self.theta_[class_index]
        std = np.sqrt(self.sigma_[class_index])

        return scipy.stats.norm(mean, std).logpdf(features)

    def _get_posterior(self, features):

        posteriors = []

        for i in range(len(self.classes_)):
            prior = self.class_log_prior_[i]
            conditional_proba = np.sum(self._get_log_prob(i, features))
            posteriors.append(prior + conditional_proba)

        return posteriors


    def fit(self, features, target):

        assert isinstance(features, pd.core.frame.DataFrame)
        assert isinstance(target, pd.core.frame.DataFrame)
        target = target.iloc[:, 0]

        self.classes_ = np.unique(target)

        self._set_gaussian_parameters(features, target)
        self._set_prior(features, target)

        self._fitted = True

    def predict_log_proba(self, features):
        
        assert self._fitted
        assert isinstance(features, pd.core.frame.DataFrame)

        instances = features.to_numpy()
        assert instances.ndim == 2

        preds = [self._get_posterior(instance) for instance in instances]
        
        return preds
    
    def predict_proba(self, features):
        
        return np.exp(self.predict_log_proba(features))
    
    def predict(self, features):
        
        log_probas = self.predict_log_proba(features)
        preds = [self.classes_[np.argmax(log_proba)] for log_proba in log_probas]
            
        return preds
