from Functions_part_b import load_cache_or_compute, select_features, train_model, evaluate_model

def run_part_b(save_cache=False, force_recompute_select_features=True, force_recompute_train_model=True, force_recompute_evaluate_model=True):

    ##--------------- Part A: select features ----------------##
    X_selected = load_cache_or_compute(
        "select_features.pkl",
        lambda: select_features([],[]),
        force_recompute=force_recompute_select_features,
        save=save_cache
    )

    print('\033[32mFeature selection completed\033[0m')

    ##--------------- Part B: train model -----------##
    model_A, model_B = load_cache_or_compute(
        "train_model.pkl",
        lambda: train_model(X_selected, []),
        force_recompute=force_recompute_train_model,
        save=save_cache
    )

    print('\033[32mTrain model completed\033[0m')

    #--------------- Part C: evaluate model ---------------##

    D  = load_cache_or_compute(
        "evaluate_model.pkl",
        lambda: evaluate_model(model_A, model_B,[],[]),
        force_recompute=force_recompute_evaluate_model,
        save=save_cache
    )

    print('\033[32mEvaluate model completed\033[0m')

    return


run_part_b()