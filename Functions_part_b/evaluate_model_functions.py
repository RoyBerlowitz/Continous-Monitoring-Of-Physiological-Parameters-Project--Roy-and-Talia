import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    recall_score,
    confusion_matrix,
    cohen_kappa_score,
    accuracy_score,
    precision_score,
    f1_score
)
import pandas as pd
import numpy as np
import os
import re

def evaluate_one_model(model, model_name, X_test, y_test):
    # predict_proba - the prob of each row to be in each class.
    # take the prob of being in class Handwashing
    y_prob = model.predict_proba(X_test)[:, 1]
    y_predicted = model.predict(X_test)

    # ---------- Accuracy ----------
    accuracy = accuracy_score(y_test, y_predicted)

    # ---------- Cohen's Kappa ----------
    # Tries to estimate how bias the model prediction towards the majority group, by dividing  (accuracy minus ration of majority group) / (1 minus ration of majority group)
    cohen_kappa = cohen_kappa_score(y_test, y_predicted)

    # ---------- ROC & AUC ----------
    roc_auc = roc_auc_score(y_test, y_prob) #Area Under the ROC Curve, higher better
    # FPR (False Positive Rate) = Routine misclassified as Handwashing
    # TPR (True Positive Rate / Sensitivity) = Handwashing correctly detected
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    best_roc_point = closest_point_roc(fpr, tpr, roc_thresholds)
    train_optimal_roc_point = get_roc_point_at_threshold(fpr, tpr, roc_thresholds, model.optimal_threshold_ROC_)

    # ---------- Precision-Recall Curve (PRC) ----------
    # Precision - How many predicted washings were correct
    # Recall - How many real washings were detected
    precision, recall, prc_thresholds = precision_recall_curve(y_test, y_prob)
    best_prc_point = closest_point_prc(precision, recall, prc_thresholds)
    prc_auc = auc(recall, precision) # area under curve. higher better detection of label 1
    train_optimal_prc_point = get_prc_point_at_threshold(recall, precision, prc_thresholds, model.optimal_threshold_PRC_)

    # we are now defining 3 working points:
    # default - threshold =0.5
    # optimal point in the AUC-ROC curve
    # optimal point in the PRC curve
    # we will evaluate the model on all three of them for comparison
    # we will use the optimal threshold by PRC and ROC that were extracted during the train, to prevent calculating the threshold based on the validation data
    working_points = [
        {'type': 'Original (0.5)', 'threshold': 0.5},
        {'type': 'Best_ROC_AUC', 'threshold': model.optimal_threshold_ROC_},
        {'type': 'Best_PRC', 'threshold': model.optimal_threshold_PRC_},
        {'type': 'Recall 0.7', 'threshold': model.threshold_70},
    ]


    ## if u want to get metrics at one thresh and print =================================================
    # def print_metrics_table(y_true, y_pred, y_prob=None, title="Metrics"):
    #     """
    #     Print classification metrics including Precision, Recall (Sensitivity), Accuracy,
    #     F1 Score, Cohen's Kappa, PR AUC, and ROC AUC.
    #
    #     Parameters:
    #     - y_true : true labels
    #     - y_pred : predicted labels
    #     - y_prob : predicted probabilities for positive class (required for AUC metrics)
    #     - title  : title for the table
    #     """
    #     from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, cohen_kappa_score
    #     from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
    #
    #     # Basic metrics
    #     p = precision_score(y_true, y_pred, zero_division=0)
    #     s = recall_score(y_true, y_pred, zero_division=0)
    #     a = accuracy_score(y_true, y_pred)
    #     f1 = f1_score(y_true, y_pred, zero_division=0)
    #     kappa = cohen_kappa_score(y_true, y_pred)
    #
    #     # PR AUC
    #     if y_prob is not None:
    #         precision, recall, _ = precision_recall_curve(y_true, y_prob)
    #         pr_auc = auc(recall, precision)
    #         # ROC AUC
    #         roc_auc = roc_auc_score(y_true, y_prob)
    #     else:
    #         pr_auc = None
    #         roc_auc = None
    #
    #     # Print metrics table
    #     print(f"\n--- {title} ---")
    #     print(f"{'Metric':<15} | {'Value':<10}")
    #     print("-" * 28)
    #     print(f"{'Precision':<15} | {p:.4f}")
    #     print(f"{'Sensitivity':<15} | {s:.4f}")
    #     print(f"{'Accuracy':<15} | {a:.4f}")
    #     print(f"{'F1 Score':<15} | {f1:.4f}")
    #     print(f"{'Cohen Kappa':<15} | {kappa:.4f}")
    #     if pr_auc is not None:
    #         print(f"{'PR AUC':<15} | {pr_auc:.4f}")
    #     if roc_auc is not None:
    #         print(f"{'ROC AUC':<15} | {roc_auc:.4f}")
    #
    # y_pred_test = (y_prob >= model.optimal_threshold_ROC_).astype(int)
    #
    # print_metrics_table(y_test,y_pred_test, y_prob, 'Results from test on model trained on train' )
    # ===================================================================================================================================================

    excel_results = []

    for wp in working_points:
        # we pring the prediction
        y_predicted = (y_prob >= wp['threshold']).astype(int)
        print(f"Threshold: {wp['threshold']:.4f} | Counts: {np.bincount(y_predicted)}")

        # # ---------- Predictions ----------
        # # specific predicted label
        # y_predicted = model.predict(X_test)

        # ---------- Accuracy ----------
        accuracy = accuracy_score(y_test, y_predicted)

        # ---------- Cohen's Kappa ----------
        # Tries to estimate how bias the model prediction towards the majority group, by dividing  (accuracy minus ration of majority group) / (1 minus ration of majority group)
        cohen_kappa = cohen_kappa_score(y_test, y_predicted)


        # ---------- Sensitivity (Recall for Class 1) ----------
        # how many real handwashing events were detected
        # Sensitivity = TP / (TP + FN)
        # TN: Routine correctly classified
        # FP: Routine classified as Handwashing(false alarm)
        # FN: Missed handwashing(dangerous)
        # TP: Correct handwashing detection
        sensitivity = recall_score(y_test, y_predicted, pos_label=1)

        # ---------- Confusion Matrices ----------
        confusion_matrx = confusion_matrix(y_test, y_predicted)

        # ---------- Confusion Matrix values ----------
        TN, FP, FN, TP = confusion_matrx.ravel()

        # ---------- Precision ----------
        precision_score_val = precision_score(y_test, y_predicted, pos_label=1)

        # ---------- Specificity (True Negative Rate) ----------
        # Specificity = TN / (TN + FP)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

        # ---------- F1 Score ----------
        f1 = f1_score(y_test, y_predicted, pos_label=1)

        # we add the the results
        excel_results.append( {
            'model_name': model_name + wp['type'],
            'accuracy': accuracy,
            'cohen_kappa': cohen_kappa,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision,
            'recall_curve': recall,
            'prc_auc': prc_auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision_score': precision_score_val,
            'f1_score': f1,
            'confusion_matrix': confusion_matrx,
            'best_roc_point': best_roc_point,
        })

    return {
        'excel_rows': excel_results,
        'plot_data': {
            'model_name': model_name,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'prc_auc': prc_auc,
            'best_roc_point': best_roc_point,
            'best_prc_point': best_prc_point,
            'train_optimal_prc_point': train_optimal_prc_point,
            'train_optimal_roc_point': train_optimal_roc_point
        }
    }


def plot_ROC(model_outputs, folder_name):
    plt.figure()

    for m in model_outputs:
        plt.plot(m['fpr'], m['tpr'], label=f"Model {m['model_name']} (AUC = {m['roc_auc']:.3f})")
        # we mark the optimal point
        best_point = m['best_roc_point']
        plt.scatter(best_point['fpr'], best_point['tpr'],
                    s=100,
                    marker='*',
                    edgecolors='black',
                    zorder=5,
                    label=f"Best Point (T={best_point['threshold']:.2f})")

        train_optimal_roc_point = m['train_optimal_roc_point']
        plt.scatter(train_optimal_roc_point[0], train_optimal_roc_point[1],
                    s=100,
                    marker='*',
                    edgecolors='black',
                    zorder=5,
                    label=f"Train optimal ROC Point (T={train_optimal_roc_point[2]:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve – Handwashing (Class 1)')
    plt.legend()

    if folder_name:
        prc_path = f"{folder_name}/roc_curve.png"
        plt.savefig(prc_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_PRC(model_outputs, folder_name):
    plt.figure()

    for m in model_outputs:
        plt.plot(m['recall'], m['precision'], label=f"Model {m['model_name']} (PRC AUC = {m['prc_auc']:.3f})")
        # we mark the optimal point
        best_point = m['best_prc_point']
        if best_point:
            plt.scatter(best_point['recall'], best_point['precision'],
                        s=100,
                        marker='D',
                        edgecolors='black',
                        zorder=5,
                        label=f"Best PRC Point (T={best_point['threshold']:.2f})")

        train_prc_point = m['train_optimal_prc_point']
        if best_point:
            plt.scatter(train_prc_point[1], train_prc_point[0],
                        s=100,
                        marker='D',
                        edgecolors='black',
                        zorder=5,
                        label=f"Train optimal PRC Point (T={train_prc_point[2]:.2f})")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall Curve – Handwashing (Class 1)')
    plt.legend()

    if folder_name:
        prc_path = f"{folder_name}/prc_curve.png"
        plt.savefig(prc_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def closest_point_roc(fpr, tpr, thresholds):
    distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
    idx = np.argmin(distances)

    return {
        'fpr': fpr[idx],
        'tpr': tpr[idx],
        'threshold': thresholds[idx],
        'distance': distances[idx]
    }

def closest_point_prc(precision, recall, thresholds):
    distances = np.sqrt((recall - 1)**2 + (precision - 1)**2)
    idx = np.argmin(distances)

    return {
        'precision': precision[idx],
        'recall': recall[idx],
        'threshold': thresholds[idx] if idx < len(thresholds) else None,
        'distance': distances[idx]
    }

def get_recall_70(precision, recall, thresholds):
    """
    :return: precision_at_70, recall_at_70, threshold_at_70
    """
    target_recall = 0.7
    idx = np.argmin(np.abs(recall - target_recall))

    precision_at_70 = precision[idx]
    recall_at_70 = recall[idx]
    print("Actual Recall:", recall_at_70)
    print("Actual Precision:", precision_at_70)


    # Threshold (only valid if idx < len(thresholds))
    threshold_at_70 = thresholds[idx] if idx < len(thresholds) else 1.0

    return precision_at_70, recall_at_70, threshold_at_70

def get_prc_point_at_threshold(recall, precision, thresholds, threshold):
    # Find the index of the threshold closest to T
    idx = np.argmin(np.abs(thresholds - threshold))

    # Get corresponding precision and recall
    precision_at_T = precision[idx + 1]
    recall_at_T = recall[idx + 1]
    threshold_at_T = thresholds[idx]

    return precision_at_T, recall_at_T, threshold_at_T

def get_roc_point_at_threshold(fpr, tpr, thresholds, threshold):
    # Find the index of the threshold closest to T
    idx = np.argmin(np.abs(thresholds - threshold))

    # Get corresponding FPR and TPR
    fpr_at_T = fpr[idx]
    tpr_at_T = tpr[idx]
    threshold_at_T = thresholds[idx]

    return fpr_at_T, tpr_at_T, threshold_at_T

def save_model_outputs_to_xlsx(model_outputs, folder_name):
    df = pd.DataFrame(model_outputs)
    columns_to_save = ['model_name', 'roc_auc', 'prc_auc', 'sensitivity', 'confusion_matrix', 'accuracy', 'cohen_kappa', 'best_roc_point', 'best_prc_point']
    df[columns_to_save].to_excel(f'{folder_name}/model_outputs.xlsx')


def create_folder_for_saving(split_name):
    base_dir = "model_outputs"

    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Find existing folders that start with a number_
    existing = os.listdir(base_dir)
    indices = []

    for folder in existing:
        match = re.match(r"(\d+)_", folder)
        if match:
            indices.append(int(match.group(1)))

    # Determine next index
    next_index = max(indices) + 1 if indices else 1

    # Create new folder
    folder_path = os.path.join(base_dir, f"{next_index}_{split_name}")
    os.makedirs(folder_path)

    return folder_path