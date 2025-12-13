from .evaluate_model_functions import evaluate_one_model, plot_ROC, plot_PRC, save_model_outputs_to_xlsx, create_folder_for_saving

def evaluate_model(models, model_names, X_test, y_test, save_model_outputs=False, split_name=None):
    # roc_auc
    # prc_auc
    # sensitivity
    # confusion_matrix

    folder_name = False
    if save_model_outputs:
        folder_name = create_folder_for_saving(split_name)

    model_outputs = []

    for i in range(len(models)):
        model_output = evaluate_one_model(models[i], model_names[i], X_test, y_test)
        model_outputs.append(model_output)

    plot_ROC(model_outputs, folder_name)
    plot_PRC(model_outputs, folder_name)

    if save_model_outputs:
        save_model_outputs_to_xlsx(model_outputs, folder_name)

    return model_outputs