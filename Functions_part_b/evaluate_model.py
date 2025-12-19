from .evaluate_model_functions import evaluate_one_model, plot_ROC, plot_PRC, save_model_outputs_to_xlsx, create_folder_for_saving
import copy as copy
import pandas as pd
def evaluate_model(models, model_names, test_x, y_test, save_model_outputs=False, split_name=None):
    # roc_auc
    # prc_auc
    # sensitivity
    # confusion_matrix

    X_test = copy.deepcopy(test_x)
    administrative_features = ['First second of the activity', 'Last second of the activity', 'Participant ID', 'Group number','Recording number', 'Protocol', "__participant_key__"]
    # we don't want the administrative features to be a part of the model, so we remove them from the hyperparameteres loop.
    X_test = X_test[[col for col in X_test.columns if col not in administrative_features]]

    folder_name = False
    if save_model_outputs:
        folder_name = create_folder_for_saving(split_name)

    all_excel_data = []
    plot_outputs = []

    for i in range(len(models)):
        model_output = evaluate_one_model(models[i], model_names[i], X_test, y_test)
        all_excel_data.extend(model_output['excel_rows'])
        plot_outputs.append(model_output['plot_data'])

    plot_ROC(plot_outputs, folder_name)
    plot_PRC(plot_outputs, folder_name)

    if save_model_outputs:
        df = pd.DataFrame(all_excel_data)
        df.to_excel(f'{folder_name}/model_outputs.xlsx', index=False)

    return all_excel_data

    # model_outputs = []
    #
    # for i in range(len(models)):
    #     model_output = evaluate_one_model(models[i], model_names[i], X_test, y_test)
    #     all_excel_data.extend(model_output['excel_rows'])
    #     plot_outputs.append(model_output['plot_data'])
    #
    # plot_ROC(model_outputs, folder_name)
    # plot_PRC(model_outputs, folder_name)
    #
    # if save_model_outputs:
    #     save_model_outputs_to_xlsx(model_outputs, folder_name)
    #
    # return model_outputs