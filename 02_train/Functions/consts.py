from dataclasses import dataclass
import numpy as np

#diff model names
class WindowModelNames:
    SVM = "svm"
    LOGISTIC = "logistic"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"

class SecondModelNames:
    NO_MODEL = "no_model"
    LOGISTIC = "logistic"
    MARKOV = "markov"

#params for wrapper func
n_features_range = [3, 5, 7, 10, 12, 15, 17, 19, 20]
window_models_hps = {
    WindowModelNames.RANDOM_FOREST:{'Random_Forest__class_weight': 'balanced', 'Random_Forest__max_depth': 24, 'Random_Forest__max_samples': np.float64(0.7852979148891981), 'Random_Forest__min_samples_leaf': 10, 'Random_Forest__min_samples_split': 24, 'Random_Forest__n_estimators': 317},
    WindowModelNames.XGBOOST:{'colsample_bytree': np.float64(0.6161734358153725), 'gamma': np.float64(0.21319886690573622), 'learning_rate': np.float64(0.014581398811193679), 'max_depth': 6, 'n_estimators': 254, 'scale_pos_weight': 1, 'subsample': np.float64(0.6125716742746937)},
}
models_hp_for_wrapper = {WindowModelNames.RANDOM_FOREST: [window_models_hps[WindowModelNames.RANDOM_FOREST], n_features_range, WindowModelNames.RANDOM_FOREST],
             WindowModelNames.XGBOOST: [window_models_hps[WindowModelNames.XGBOOST], n_features_range, WindowModelNames.XGBOOST]}

#admin features
admin_features = ['First second of the activity','Last second of the activity','Participant ID','Group number','Recording number','Protocol']

#force recompute full object
@dataclass
class RecomputeFunctionsConfig:
    load_data: bool = True
    segment_signal: bool = True
    extract_features: bool = True
    split_data: bool = True
    vet_features_and_normalize: bool = True
    select_features: bool = True
    choose_hyperparameters: bool = True
    train_window_model: bool = True
    create_test_time_df: bool = True
    prediction_by_second: bool = True