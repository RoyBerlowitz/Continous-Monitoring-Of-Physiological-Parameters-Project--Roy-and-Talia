from .window_timing_translator_preprocessing import create_df_for_time_classification

def create_test_time_df(X_test, model, selected_feats):
    df_for_time_classification = create_df_for_time_classification(X_test)
    y_prob = model.predict_proba(X_test[selected_feats])[:, 1]
    df_for_time_classification["window_probability"] = y_prob

    return df_for_time_classification