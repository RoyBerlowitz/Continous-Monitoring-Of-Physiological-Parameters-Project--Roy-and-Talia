class ModelNames:
    SVM = "svm"
    LOGISTIC = "logistic"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"

class ModelNamesSecondClassification:
    NO_MODEL = "no_model"
    LOGISTIC = "logistic"

chosen_hp_split1 = {
    ModelNames.XGBOOST: [
        {'run_name':'1','params':{}}
    ],
    ModelNames.RANDOM_FOREST: [
        {'run_name':'1','params':{}}
    ],
    ModelNames.LOGISTIC: [],
    ModelNames.SVM: [],
}

chosen_hp_split2 = {
    ModelNames.XGBOOST: [
        {'run_name':'1','params':{}}
    ],
    ModelNames.RANDOM_FOREST: [
        {'run_name':'1','params':{}}
    ],
    ModelNames.LOGISTIC: [],
    ModelNames.SVM: [],
}