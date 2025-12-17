import pickle

from Functions_part_b import load_cache_or_compute, select_features, choose_hyperparameters, train_model, evaluate_model, ModelNames

columns_not_to_normalize = ['First second of the activity','Last second of the activity','Participant ID','Group number','Recording number','Protocol']

def run_part_b_specific_dataset(X_train, X_test, y_train, y_test, scaler, split_name, save_cache=False, force_recompute_select_features=True, force_recompute_find_hp=True, force_recompute_train_model=True, force_recompute_evaluate_model=True):

    features = [col for col in X_train.columns if col not in columns_not_to_normalize]
    xtrain = X_train#[features]
    xtest = X_test[features]

    print(f'\033[34mStarting on split {split_name} ==========================================\033[0m')

    ##--------------- Part A: select features ----------------##
    # X_selected = load_cache_or_compute(
    #     f"{split_name}_select_features.pkl",
    #     lambda: select_features([],[]),
    #     force_recompute=force_recompute_select_features,
    #     save=save_cache
    # )

    print('\033[32mFeature selection completed\033[0m')

    ##--------------- Part B: train model -----------##
    trained_models = {}

    # for model_name in [ModelNames.LOGISTIC, ModelNames.XGBOOST, ModelNames.SVM, ModelNames.RANDOM_FOREST]: #all
    for model_name in [ModelNames.SVM, ModelNames.RANDOM_FOREST]: #Roee
    # for model_name in [ModelNames.XGBOOST, ModelNames.LOGISTIC]: #talia
        #get best hyperparameters
        model_best_hp = load_cache_or_compute(
            f"{split_name}_{model_name}_best_hp.pkl",
            # lambda: choose_hyperparameters(X_selected,y_train,model_name,split_name=split_name),
            lambda: choose_hyperparameters(xtrain,y_train,model_name,split_name=split_name),
            force_recompute=force_recompute_find_hp,
            save=save_cache
        )

        #train chosen model
        trained_models[model_name] = load_cache_or_compute(
            f"{split_name}_{model_name}_trained_model.pkl",
            # lambda: train_model(X_selected, []),
            lambda: train_model(xtrain, y_train, model_best_hp, model_name),
            force_recompute=force_recompute_train_model,
            save=save_cache
        )

        print(f'\033[32mFinished training {model_name} {split_name}\033[0m')

    print('\033[32mTrain model completed\033[0m')

    # --------------- Part C: evaluate model ---------------##

    D = load_cache_or_compute(
        f"{split_name}_evaluate_models.pkl",
        lambda: evaluate_model(list(trained_models.values()), list(trained_models.keys()),
                               xtest, y_test,
                               split_name = split_name,
                               save_model_outputs=True),
        force_recompute=force_recompute_evaluate_model,
        save=save_cache
    )

    print('\033[32mEvaluate model completed\033[0m')

    return

def run_part_b(save_cache=False, force_recompute_select_features=True, force_recompute_find_hp=True, force_recompute_train_model=True, force_recompute_evaluate_model=True):

    #load part a
    part_a_res_cache_path = "part_a_final_output.pkl"
    # part_a_res_cache_path = "part_a_final_output_all_data.pkl"
    with open(part_a_res_cache_path, "rb") as f:
        part_a_res = pickle.load(f)

    split1_vet_features, split2_vet_features = part_a_res

    # #split1
    split1_X_train, split1_X_test, split1_y_train, split1_y_test, scalers = split1_vet_features
    run_part_b_specific_dataset(split1_X_train, split1_X_test, split1_y_train, split1_y_test, None, 'split_1', save_cache,
                                force_recompute_select_features, force_recompute_find_hp, force_recompute_train_model, force_recompute_evaluate_model)

    #split2
    X_vetting, X_test_norm, split2_Y_train, split2_Y_test, scaler = split2_vet_features
    run_part_b_specific_dataset(X_vetting, X_test_norm, split2_Y_train, split2_Y_test, scaler, 'split_2',save_cache, force_recompute_select_features, force_recompute_find_hp, force_recompute_train_model, force_recompute_evaluate_model)

    return


run_part_b(save_cache=True)