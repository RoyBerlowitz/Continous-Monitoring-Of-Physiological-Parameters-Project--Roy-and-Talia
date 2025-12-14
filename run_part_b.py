import pickle

from Functions_part_b import load_cache_or_compute, select_features, train_model, evaluate_model

columns_not_to_normalize = ['First second of the activity','Last second of the activity','Participant ID','Group number','Recording number','Protocol']

def run_part_b_specific_dataset(X_train, X_test, y_train, y_test, scaler, split_name, save_cache=False, force_recompute_select_features=True, force_recompute_train_model=True, force_recompute_evaluate_model=True):

    features = [col for col in X_train.columns if col not in columns_not_to_normalize]
    xtrain = X_train[features]
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
    model_A, model_B = load_cache_or_compute(
        f"{split_name}_train_model.pkl",
        # lambda: train_model(X_selected, []),
        lambda: train_model(xtrain, y_train),
        force_recompute=force_recompute_train_model,
        save=save_cache
    )

    print('\033[32mTrain model completed\033[0m')

    # --------------- Part C: evaluate model ---------------##

    D = load_cache_or_compute(
        f"{split_name}_evaluate_model.pkl",
        lambda: evaluate_model([model_A, model_B], ['logistic', 'xgboost'],
                               xtest, y_test,
                               split_name = split_name,
                               save_model_outputs=True),
        force_recompute=force_recompute_evaluate_model,
        save=save_cache
    )

    print('\033[32mEvaluate model completed\033[0m')

    return

def run_part_b(save_cache=False, force_recompute_select_features=True, force_recompute_train_model=True, force_recompute_evaluate_model=True):

    #load part a
    part_a_res_cache_path = "part_a_final_output.pkl"
    with open(part_a_res_cache_path, "rb") as f:
        part_a_res = pickle.load(f)

    split1_vet_features, split2_vet_features = part_a_res

    #split1
    X_trains, X_tests, y_trains, y_tests, scalers = split1_vet_features

    participant_id_column_name = 'Participant ID'
    group_number_column_name = 'Group number'

    all_participants = list(X_trains.groupby([participant_id_column_name, group_number_column_name]).groups.keys())

    for i in range(len(all_participants)):
        (pid, gid) = all_participants[i]

        # Filter only this participant+group rows
        mask_train = ((X_trains[participant_id_column_name] == pid) &
                (X_trains[group_number_column_name] == gid))
        mask_test = ((X_tests[participant_id_column_name] == pid) &
                (X_tests[group_number_column_name] == gid))

        X_train = X_trains[mask_train]
        X_test = X_tests[mask_test]
        y_train = y_trains[mask_train]
        y_test = y_tests[mask_test]
        scaler = scalers[i]

        if sum(y_train==1) > 0 and sum(y_test==1) > 0:
            run_part_b_specific_dataset(X_train, X_test, y_train, y_test, scaler, f'split_1_{pid}_{gid}', save_cache,
                                    force_recompute_select_features, force_recompute_train_model,
                                    force_recompute_evaluate_model)
        else:
            print(f'\033[31mIn split 1 participant {pid} group {gid} y_train has {sum(y_train==1)} label 1, y_test has {sum(y_test==1)} label 1\033[0m')

    #split2
    X_vetting, X_test_norm, split2_Y_train, split2_Y_test, scaler = split2_vet_features
    run_part_b_specific_dataset(X_vetting, X_test_norm, split2_Y_train, split2_Y_test, scaler, 'split_2',save_cache, force_recompute_select_features, force_recompute_train_model, force_recompute_evaluate_model)

    return


run_part_b(save_cache=True)