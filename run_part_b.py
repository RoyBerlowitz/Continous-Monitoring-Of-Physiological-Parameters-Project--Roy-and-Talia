import pickle
import numpy as np

from Functions_part_b import (load_cache_or_compute, select_features, choose_hyperparameters, train_model,
                              evaluate_model, ModelNames, chosen_hp_split1, chosen_hp_split2, wrapper_selection)

admin_features = ['First second of the activity','Last second of the activity','Participant ID','Group number','Recording number','Protocol']
wrapper_params = {
    ModelNames.RANDOM_FOREST:{'Random_Forest__class_weight': 'balanced', 'Random_Forest__max_depth': 37, 'Random_Forest__max_samples': np.float64(0.7253152871382157), 'Random_Forest__min_samples_split': 28, 'Random_Forest__n_estimators': 445},
    ModelNames.XGBOOST:{'colsample_bytree': np.float64(0.6161734358153725), 'gamma': np.float64(0.21319886690573622), 'learning_rate': np.float64(0.014581398811193679), 'max_depth': 6, 'n_estimators': 254, 'scale_pos_weight': 1, 'subsample': np.float64(0.6125716742746937)},
}

def run_part_b_specific_dataset(X_train, X_test, y_train, y_test, scaler, models_to_run, split_name, split_by_group_flag=False, use_wrapper=False, chosen_hp=None, save_cache=False, force_recompute_select_features=True, force_recompute_find_hp=True, force_recompute_train_model=True, force_recompute_evaluate_model=True):

    features = [col for col in X_train.columns if col not in admin_features]
    wrapper_text = "" if not use_wrapper else "_wrapper"

    print(f'\033[34mStarting on split {split_name}{wrapper_text} ==========================================\033[0m')

    ##--------------- Part A: select features ----------------##
    if not use_wrapper:
        selected_feats = load_cache_or_compute(
            f"{split_name}_select_features.pkl",
            lambda: select_features(X_train, y_train, X_test, split_name=split_name),
            force_recompute=force_recompute_select_features,
            save=save_cache
        )

        print('\033[32mFeature selection completed\033[0m')

    ##--------------- Part B: train model -----------##
    trained_models = {}

    for model_name in models_to_run:# talia
        if use_wrapper:
            selected_feats = load_cache_or_compute(
                    f"{split_name}_{model_name}_wrapper_select_features.pkl",
                    lambda: wrapper_selection(X_train[features], y_train, frozen_params=wrapper_params[model_name], model_type=model_name),
                    force_recompute=force_recompute_select_features,
                    save=save_cache
                )
            print('\033[32mFeature selection completed\033[0m')

        X_selected = X_train[selected_feats+admin_features]

        #get best hyperparameters
        model_best_hp = load_cache_or_compute(
            f"{split_name}_{model_name}{wrapper_text}_best_hp.pkl",
            lambda: choose_hyperparameters(X_selected,y_train,model_name,split_name=split_name,split_by_group_flag=split_by_group_flag),
            force_recompute=force_recompute_find_hp,
            save=save_cache
        )

        #train chosen model
        trained_models[model_name] = load_cache_or_compute(
            f"{split_name}_{model_name}{wrapper_text}_trained_model.pkl",
            lambda: train_model(X_selected, y_train, model_best_hp, model_name),
            force_recompute=force_recompute_train_model,
            save=save_cache
        )

        print(f'\033[32mFinished training {model_name} {split_name}{wrapper_text}\033[0m')

    print('\033[32mTrain model completed\033[0m')

    # --------------- Part C: evaluate model ---------------##

    D =  load_cache_or_compute(
        f"{split_name}{wrapper_text}_evaluate_models.pkl",
        lambda: evaluate_model(list(trained_models.values()), list(trained_models.keys()),
                               X_test[selected_feats], y_test,
                               split_name = split_name,
                               save_model_outputs=True),
        force_recompute=force_recompute_evaluate_model,
        save=save_cache
    )

    print('\033[32mEvaluate model completed\033[0m')

    return

def run_part_b(chosen_hp_split1=None, chosen_hp_split2=None, save_cache=False, force_recompute_select_features=True, force_recompute_find_hp=True, force_recompute_train_model=True, force_recompute_evaluate_model=True):

    models = [ModelNames.LOGISTIC, ModelNames.XGBOOST, ModelNames.SVM, ModelNames.RANDOM_FOREST]  #all
    wrapper_models = [ModelNames.XGBOOST, ModelNames.RANDOM_FOREST]
    wrapper_models = [ModelNames.XGBOOST]

    #load part a
    # part_a_res_cache_path = "part_a_final_output.pkl"
    part_a_res_cache_path = "part_a_final_output_all_data.pkl"
    with open(part_a_res_cache_path, "rb") as f:
        part_a_res = pickle.load(f)

    split1_vet_features, split2_vet_features = part_a_res

    # #split1
    split1_X_train, split1_X_test, split1_y_train, split1_y_test, scalers = split1_vet_features
    # run_part_b_specific_dataset(split1_X_train, split1_X_test, split1_y_train, split1_y_test, None, models, 'split_1', save_cache=save_cache,
    #                             force_recompute_select_features=force_recompute_select_features, force_recompute_find_hp=force_recompute_find_hp,
    #                             force_recompute_train_model=force_recompute_train_model, force_recompute_evaluate_model=force_recompute_evaluate_model,
    #                             chosen_hp=chosen_hp_split1)
    # run_part_b_specific_dataset(split1_X_train, split1_X_test, split1_y_train, split1_y_test, None, wrapper_models, 'split_1',
    #                             use_wrapper=True, save_cache=save_cache,
    #                             force_recompute_select_features=force_recompute_select_features,
    #                             force_recompute_find_hp=force_recompute_find_hp,
    #                             force_recompute_train_model=force_recompute_train_model,
    #                             force_recompute_evaluate_model=force_recompute_evaluate_model,
    #                             chosen_hp=chosen_hp_split1)

    #split2
    X_vetting, X_test_norm, split2_Y_train, split2_Y_test, scaler = split2_vet_features
    # run_part_b_specific_dataset(X_vetting, X_test_norm, split2_Y_train, split2_Y_test, scaler, models, 'split_2',True, save_cache=save_cache,
    #                             force_recompute_select_features=force_recompute_select_features, force_recompute_find_hp=force_recompute_find_hp,
    #                             force_recompute_train_model=force_recompute_train_model, force_recompute_evaluate_model=force_recompute_evaluate_model,
    #                             chosen_hp=chosen_hp_split2)
    run_part_b_specific_dataset(X_vetting, X_test_norm, split2_Y_train, split2_Y_test, scaler, wrapper_models, 'split_2', True,
                                use_wrapper=True, save_cache=save_cache,
                                force_recompute_select_features=force_recompute_select_features,
                                force_recompute_find_hp=force_recompute_find_hp,
                                force_recompute_train_model=force_recompute_train_model,
                                force_recompute_evaluate_model=force_recompute_evaluate_model,
                                chosen_hp=chosen_hp_split2)

    return

run_part_b(save_cache=True, force_recompute_select_features=True, force_recompute_find_hp=True,
                                force_recompute_train_model=True, force_recompute_evaluate_model=True,)
# run_part_b(save_cache=True, chosen_hp_split1=chosen_hp_split1, chosen_hp_split2=chosen_hp_split2)
