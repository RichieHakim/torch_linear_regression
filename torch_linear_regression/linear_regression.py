from typing import Dict, Type, Any, Union, Optional, Callable, Tuple, List
import functools

import numpy as np
import sklearn
import sklearn.linear_model
import torch


class LinearRegression_sk(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin, torch.nn.Module):
    """
    Implements a basic linear regression estimator following the scikit-learn
    estimator interface.
    RH 2023
    
    Attributes:
        coef_ (Union[np.ndarray, torch.Tensor]):
            Coefficients of the linear model.
        intercept_ (Union[float, np.ndarray, torch.Tensor]):
            Intercept of the linear model.
    """

    def __init__(self, **kwargs):
        """
        Initializes the LinearRegression_sk model.
        """
        super(LinearRegression_sk, self).__init__()

        self.coef_ = None
        self.intercept_ = None

    def get_backend_namespace(
        self, 
        X: Union[np.ndarray, torch.Tensor], 
        y: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Dict[str, Callable]:
        """
        Determines the appropriate numerical backend (NumPy or PyTorch) based on
        the type of input data.

        Args:
            X (Union[np.ndarray, torch.Tensor]): 
                Input features array or tensor.
            y (Optional[Union[np.ndarray, torch.Tensor]]): 
                Optional target array or tensor. (Default is ``None``)

        Raises:
            NotImplementedError: 
                If the input type is neither NumPy array nor PyTorch tensor.
        
        Returns:
            Dict[str, Callable]: 
                Dictionary containing backend-specific functions and
                constructors.
        """
        if isinstance(X, torch.Tensor):
            backend = 'torch'
            if y is not None:
                assert X.device == y.device, 'X and y must be on the same device'
            device = X.device
        elif isinstance(X, np.ndarray):
            backend = 'numpy'
        else:
            raise NotImplementedError
        
        if y is not None:
            assert isinstance(y, type(X)), 'X and y must be the same type'
        
        dtype = X.dtype

        ns = {}

        if backend == 'numpy':
            ns['inv'] = np.linalg.inv
            ns['eye'] = functools.partial(np.eye, dtype=dtype)
            ns['ones'] = functools.partial(np.ones, dtype=dtype)
            ns['zeros'] = functools.partial(np.zeros, dtype=dtype)
            ns['cat'] = np.concatenate
            ns['svd'] = np.linalg.svd
            ns['eigh'] = np.linalg.eigh
        elif backend == 'torch':
            ns['inv'] = torch.linalg.inv
            ns['eye'] = functools.partial(torch.eye, device=device, dtype=dtype)
            ns['ones'] = functools.partial(torch.ones, device=device, dtype=dtype)
            ns['zeros'] = functools.partial(torch.zeros, device=device, dtype=dtype)
            ns['cat'] = torch.cat
            ns['svd'] = torch.linalg.svd
            ns['eigh'] = torch.linalg.eigh
        else:
            raise NotImplementedError
        
        return ns
    
    def _check_X(self, X: Union[np.ndarray, torch.Tensor]):
        """
        Passes X through the _check_array method, and gives a UserWarning if
        n_features >= n_samples.
        """
        X = self._check_array(X, atleast_2d=True)
            
        ## Give a UserWarning if n_features >= n_samples
        if X.shape[1] >= X.shape[0]:
            import warnings
            warnings.warn('Regression solution is expected to diverge from sklearn solution when n_features >= n_samples')
        return X

    def _check_y(self, y: Union[np.ndarray, torch.Tensor]):
        """
        Passes y through the _check_array method.
        """
        y = self._check_array(y, atleast_2d=False)
        return y
    
    def _check_array(self, arr: Union[np.ndarray, torch.Tensor], atleast_2d: bool = False):
        """
        Checks input array for type and dimensions. Must be numpy or torch
        array, and must be 1 or 2D, and if 1D it can be converted to 2D if
        atleast_2d is True.

        Args:
            arr (Union[np.ndarray, torch.Tensor]):
                Input array or tensor.
            atleast_2d (bool):
                If True, converts 1D array to 2D.

        Returns:
            Union[np.ndarray, torch.Tensor]:
                Input array or tensor, converted to 2D if 1D.

        Raises:
            AssertionError:
                If the input is not a NumPy array or PyTorch tensor.
            AssertionError:
                If the input is not 1D or 2D.
        """
        assert isinstance(arr, (np.ndarray, torch.Tensor)), 'array must be a NumPy array or PyTorch tensor'
        assert arr.ndim in [1, 2], 'array must be 1D or 2D'

        if (arr.ndim == 1) and atleast_2d:
            arr = arr[:, None]
        return arr


    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Predicts the target values using the linear model.

        Args:
            X (Union[np.ndarray, torch.Tensor]): 
                Input features array or tensor.
        
        Returns:
            Union[np.ndarray, torch.Tensor]: 
                Predicted target values.
        """
        if X.ndim==1:
            X = X[:, None]

        return X @ self.coef_ + self.intercept_
    
    def fit_predict(
        self, 
        X: Union[np.ndarray, torch.Tensor], 
        y: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Fits the model to the data and returns the predicted target values.

        Args:
            X (Union[np.ndarray, torch.Tensor]): 
                Input features array or tensor.
            y (Union[np.ndarray, torch.Tensor]): 
                Target values.
        
        Returns:
            Union[np.ndarray, torch.Tensor]: 
                Predicted target values.
        """
        self.fit(X, y)
        return self.predict(X)

    def score(
        self, 
        X: Union[np.ndarray, torch.Tensor], 
        y: Union[np.ndarray, torch.Tensor], 
        sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculates the coefficient of determination R^2 of the prediction.

        Args:
            X (Union[np.ndarray, torch.Tensor]): 
                Input features for which to predict the targets.
            y (Union[np.ndarray, torch.Tensor]): 
                True target values.
            sample_weight (Optional[np.ndarray]): 
                Sample weights. (Default is ``None``)
        
        Raises:
            NotImplementedError: 
                If the input type is neither NumPy array nor PyTorch tensor.

        Returns:
            float: 
                R^2 score indicating the prediction accuracy. Uses
                sklearn.metrics.r2_score if input is NumPy array, and implements
                similar calculation if input is PyTorch tensor.
        """
        y_pred = self.predict(X)
        if isinstance(X, torch.Tensor):
            if sample_weight is not None:
                assert sample_weight.ndim == 1
                assert isinstance(sample_weight, torch.Tensor)
                weight = sample_weight[:, None]
            else:
                weight, sample_weight = 1.0, 1.0

            numerator = torch.sum(weight * (y - y_pred) ** 2, dim=0)
            denominator = (weight * (y - torch.mean(y * sample_weight, dim=0)) ** 2).sum(dim=0)

            output_scores = 1 - (numerator / denominator)
            output_scores[denominator == 0.0] = 0.0
            return torch.mean(output_scores) ## 'uniform_average' in sklearn
        
        elif isinstance(X, np.ndarray):
            return sklearn.metrics.r2_score(
                y_true=y, 
                y_pred=y_pred, 
                sample_weight=sample_weight, 
                multioutput='uniform_average',
            )
        
        else:
            raise NotImplementedError
        
    def to(self, device):
        self.coef_ = torch.as_tensor(self.coef_, device=device)
        self.intercept_ = torch.as_tensor(self.intercept_, device=device)
        return self
    def cpu(self):
        return self.to('cpu')
    def numpy(self):
        def check_and_convert(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            else:
                return x
        self.coef_, self.intercept_ = (check_and_convert(x) for x in (self.coef_, self.intercept_))
        return self
    
    def forward(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return self.predict(X)
    
    # def __repr__(self):
    #     return f'LinearRegression_sk(coef={self.coef_}, intercept={self.intercept_})'


class OLS(LinearRegression_sk):
    """
    Implements Ordinary Least Squares (OLS) regression. \n

    NOTE: This class does not check for NaNs, Infs, singular matrices, etc. \n
    This method is fast and accurate for small to medium-sized datasets. When
    ``n_features`` is large and/or underdetermined (``n_samples`` <= ``n_features``),
    the solution will likely diverge from the sklearn solution. \n

    RH 2023
    
    Args:
        fit_intercept (bool):
            Specifies whether to calculate the intercept for this model. If set
            to ``True``, a column of ones is added to the feature matrix \(X\).
            (Default is ``True``)
        prefit_X (Optional[Union[np.ndarray, torch.Tensor]]):
            Optional. If the ``X`` array is provided here, then ``inv(X^T X) @
            X^T`` will be precomputed and stored in the model.

    Attributes:
        fit_intercept (bool):
            Specifies if the intercept should be calculated.
        X_precompute (bool):
            Indicates if ``inv(X.T @ X) @ X.T`` is precomputed.
        iXTXXT (Optional[np.ndarray]):
            Stores the precomputed result of ``inv(X.T @ X) @ X.T`` if provided,
            otherwise ``None``.
    """

    def __init__(
        self,
        fit_intercept=True,
        prefit_X=None,
        **kwargs,
    ):
        """
        Initializes the OLS regression model with the specified `fit_intercept`
        and an optional `prefit_X`.
        """
        super(OLS, self).__init__()
        self.fit_intercept = fit_intercept

        self.iXTXXT = self.prefit(prefit_X) if prefit_X is not None else None
        self.prefit_X = True if prefit_X is not None else False

    def prefit(self, X):
        """
        Precomputes the ``inv(X.T @ X) @ X.T`` to be used in the ``fit`` method.

        Args:
            X (np.ndarray): 
                The input features matrix.

        Returns:
            np.ndarray: 
                Precomputed result of ``inv(X.T @ X) @ X.T``.
        """
        ns = self.get_backend_namespace(X=X)
        cat, inv, ones = (ns[key] for key in ['cat', 'inv', 'ones'])
        if self.fit_intercept:
            X = cat((X, ones((X.shape[0], 1))), axis=1)
        
        inv_XT_X_XT = inv(X.T @ X) @ X.T
        return inv_XT_X_XT

    def fit(self, X, y):
        """
        Fits the OLS regression model to the data.

        Args:
            X (np.ndarray): 
                Training data features.
            y (np.ndarray): 
                Target values.

        Returns:
            OLS: 
                The instance of this OLS model.
        """
        self._check_X(X)

        self.n_features_in_ = X.shape[1]
        
        ns = self.get_backend_namespace(X=X, y=y)
        zeros = ns['zeros']

        ## Regression algorithm
        inv_XT_X_XT = self.prefit(X) if self.iXTXXT is None else self.iXTXXT
        theta = inv_XT_X_XT @ y

        ### Extract bias terms
        if self.fit_intercept:
            self.intercept_ = theta[-1]
            self.coef_ = theta[:-1]
        else:
            self.intercept_ = zeros((y.shape[1],)) if y.ndim == 2 else 0
            self.coef_ = theta

        return self
    
    
class Ridge(LinearRegression_sk):
    """
    Implements Ridge regression based on the closed-form / 'analytic solution'. \n
    NOTE: This class does not check for NaNs, Infs, singular matrices, etc. \n
    This method is fast and accurate for small to medium-sized datasets. When
    ``n_features`` is large and/or underdetermined (``n_samples`` <
    ``n_features``), the solution will likely diverge from the sklearn solution. \n

    RH 2023
    
    Args:
        alpha (float):
            The regularization strength which must be a positive float.
            Regularizes the estimate to prevent overfitting by constraining the
            size of the coefficients. Usually ~1e5. (Default is 1)
        fit_intercept (bool):
            Whether to calculate the intercept for this model. If set to
            ``False``, no intercept will be used in calculations (i.e., data is
            expected to be centered). (Default is ``False``)
        prefit_X (Optional[np.ndarray, torch.Tensor]):
            Optional. If the ``X`` array is provided here, then ``inv(X.T @ X +
            alpha*eye(X.shape[1])) @ X.T`` will be precomputed and stored in the
            model.

    Attributes:
        alpha (float):
            Regularization parameter.
        fit_intercept (bool):
            Specifies if the intercept should be calculated.
        iXTXaeXT (Optional[np.ndarray]):
            Stores the precomputed ``inv(X.T @ X + alpha*eye(X.shape[1])) @
            X.T`` if precomputed, otherwise ``None``.
    """
    def __init__(
        self, 
        alpha=1, 
        fit_intercept=True,
        prefit_X=None,
        **kwargs,
    ):
        """
        Initializes the Ridge regression model with the specified alpha,
        fit_intercept, and optional X_precompute.
        """
        super(Ridge, self).__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept

        self.iXTXaeXT = self.prefit(prefit_X) if prefit_X is not None else None
        self.prefit_X = True if prefit_X is not None else False

    def prefit(self, X):
        """
        Precomputes ``inv(X.T @ X + alpha*eye(X.shape[1])) @ X.T`` to be used in
        the `fit` method.

        Args:
            X (np.ndarray): 
                The input features matrix.

        Returns:
            np.ndarray: 
                Precomputed inverse of ``inv(X.T @ X + alpha*eye(X.shape[1])) @
                X.T``.
        """
        ns = self.get_backend_namespace(X=X)
        cat, eye, inv, ones = (ns[key] for key in ['cat', 'eye', 'inv', 'ones'])
        if self.fit_intercept:
            X = cat((X, ones((X.shape[0], 1))), axis=1)
                
        inv_XT_X_plus_alpha_eye_XT = inv(X.T @ X + self.alpha*eye(X.shape[1])) @ X.T
        return inv_XT_X_plus_alpha_eye_XT

    def fit(self, X, y):
        """
        Fits the Ridge regression model to the data.

        Args:
            X (np.ndarray): 
                Training data features.
            y (np.ndarray): 
                Target values.

        Returns:
            Ridge: 
                The instance of this Ridge model.
        """
        self.n_features_in_ = X.shape[1]

        ## Give a UserWarning if n_features >= n_samples
        if X.shape[1] >= X.shape[0]:
            import warnings
            warnings.warn('OLS solution is expected to diverge from sklearn solution when n_features >= n_samples')

        ns = self.get_backend_namespace(X=X, y=y)
        zeros = ns['zeros']

        ## Regression algorithm
        inv_XT_X_plus_alpha_eye_XT = self.prefit(X) if self.iXTXaeXT is None else self.iXTXaeXT
        theta = inv_XT_X_plus_alpha_eye_XT @ y

        ### Extract bias terms
        if self.fit_intercept:
            self.intercept_ = theta[-1]
            self.coef_ = theta[:-1]
        else:
            self.intercept_ = zeros((y.shape[1],)) if y.ndim == 2 else 0
            self.coef_ = theta

        return self


class ReducedRankRidgeRegression(LinearRegression_sk):
    """
    Implements a Reduced Rank Ridge Regression model using a closed-form
    solution based on doing SVD on the coefficients of the Ridge regression Beta
    matrix. This model reduces the rank of the coefficient matrix to address
    multicollinearity and reduce model complexity. \n
    NOTE: self.coef_ is stored as the dense beta coefficients, not as a Kruskal
    tensor. \n

    RH 2024
    
    Args:
        rank (int):
            The target rank for the reduced rank regression model. It determines
            the number of singular values to retain in the model. (Default is 3)
        alpha (float):
            The regularization strength which must be a positive float.
            Regularizes the estimate to prevent overfitting by constraining the
            size of the coefficients. Usually ~1e5. (Default is 1)
        fit_intercept (bool):
            Specifies whether to calculate the intercept for this model. If set
            to ``True``, a column of ones is added to the feature matrix \(X\).
            (Default is ``True``)
        prefit_X (Optional[np.ndarray, torch.Tensor]):
            Optional. If the ``X`` array is provided here, then ``inv(X^T X +
            alpha*eye(X.shape[1])) @ X^T`` will be precomputed and stored
            in the model.

    Attributes:
        rank (int):
            The rank specified for reducing the regression model.
        alpha (float):
            Regularization parameter.
        fit_intercept (bool):
            Specifies if the intercept should be calculated.
        iXTXaeXT (Optional[np.ndarray]):
            Stores the precomputed ``inv(X.T @ X + alpha*eye(X.shape[1])) @
            X.T`` if precomputed, otherwise ``None``.
    """

    def __init__(
        self, 
        rank=3, 
        alpha=1,
        fit_intercept=True,
        prefit_X=None,
        **kwargs,
    ):
        """
        Initializes the Reduced Rank Regression model with the specified rank
        and an optional 
        """
        super(ReducedRankRidgeRegression, self).__init__()
        self.fit_intercept = fit_intercept
        self.rank = rank
        self.alpha = alpha

        self.iXTXXT = self.prefit(prefit_X) if prefit_X is not None else None
        self.prefit_X = True if prefit_X is not None else False

    def prefit(self, X):
        """
        Precomputes the \(X^T X + \alpha I\) inverse times \(X^T\) to be used in
        the `fit` method.

        Args:
            X (np.ndarray): 
                The input features matrix.

        Returns:
            np.ndarray: 
                Precomputed inverse of \(X^T X + \alpha I\) times \(X^T\).
        """
        ns = self.get_backend_namespace(X=X)
        cat, eye, inv, ones = (ns[key] for key in ['cat', 'eye', 'inv', 'ones'])
        if self.fit_intercept:
            X = cat((X, ones((X.shape[0], 1))), axis=1)
                
        inv_XT_X_plus_alpha_eye_XT = inv(X.T @ X + self.alpha*eye(X.shape[1])) @ X.T
        return inv_XT_X_plus_alpha_eye_XT

    def fit(self, X, y, rank=None):
        """
        Fits the Reduced Rank Regression model to the data, applying a rank
        reduction on the OLS regression coefficients.

        Args:
            X (np.ndarray): 
                Training data features.
            y (np.ndarray): 
                Target values.
            rank (Optional[int]):
                The target rank for the reduced rank regression model. It
                determines the number of singular values to retain in the model.
                (Default is ``None``)

        Returns:
            ReducedRankRegression: 
                The instance of this Reduced Rank Regression model.
        """
        self.rank = self.rank if rank is None else rank
        self.n_features_in_ = X.shape[1]
        assert self.rank <= X.shape[1], 'Rank must be less than or equal to the number of features'

        ns = self.get_backend_namespace(X=X, y=y)
        svd, zeros = (ns[key] for key in ['svd', 'zeros'])

        ## Calculate OLS solution
        inv_XT_X_plus_alpha_eye_XT = self.prefit(X) if self.iXTXXT is None else self.iXTXXT
        B_r = inv_XT_X_plus_alpha_eye_XT @ y

        ## Calculate SVD of B_r
        U, S, Vt = svd(B_r, full_matrices=False)
        
        ## Truncate to rank
        Vt = Vt[:self.rank]
        B_rrr = B_r @ Vt.T @ Vt

        ### Extract bias terms
        if self.fit_intercept:
            self.intercept_ = B_rrr[-1]
            self.coef_ = B_rrr[:-1]
        else:
            self.intercept_ = zeros((y.shape[1],)) if y.ndim == 2 else 0
            self.coef_ = B_rrr

        return self

