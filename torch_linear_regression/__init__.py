from .linear_regression import LinearRegression_sk, OLS, Ridge, ReducedRankRidgeRegression
from .ridge_mml import RidgeMML
from .utils import check_full_rank
from .lars_en import lars_elastic_net_path, interpolate_lars_path, SingularGramError

__version__ = '0.4.0'