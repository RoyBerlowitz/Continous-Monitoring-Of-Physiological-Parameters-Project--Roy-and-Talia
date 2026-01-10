from Continous_monitoring.run_part_a import run_part_a
# from Continous_monitoring.run_part_b import run_part_b
from Continous_monitoring.main_02 import run_part_b
from Continous_monitoring.run_part_c import run_part_c

split1_dfs, split2_dfs = run_part_a(data_path, save_cache=True, more_prints=True, force_recompute_load_data=False,
                                    force_recompute_seg=False, force_recompute_features=False,
                                    force_recompute_splits=True, force_recompute_vet_features=True)

run_part_b(chosen_hp_split1, chosen_hp_split2, wrapper_models, save_cache=False, force_recompute_select_features=True, force_recompute_find_hp=True,
                                force_recompute_train_model=True, force_recompute_evaluate_model=True, use_wrapper = True)

res, gs = run_part_c(save_cache=True, force_recompute_load_data=False, force_recompute_select_features=False,
                     force_recompute_find_hp=False, force_recompute_train_model=True,
                     force_recompute_evaluate_model=True)
