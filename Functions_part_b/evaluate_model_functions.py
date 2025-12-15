import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    recall_score,
    confusion_matrix
)
import pandas as pd
import os
import re

def evaluate_one_model(model, model_name, X_test, y_test):
    # predict_proba - the prob of each row to be in each class.
    # take the prob of being in class hand washing
    y_prob = model.predict_proba(X_test)[:, 1]

    # ---------- ROC & AUC ----------
    roc_auc = roc_auc_score(y_test, y_prob) #Area Under the ROC Curve, higher better
    # FPR (False Positive Rate) = Routine misclassified as Handwashing
    # TPR (True Positive Rate / Sensitivity) = Handwashing correctly detected
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # ---------- Precision-Recall Curve (PRC) ----------
    # Precision - How many predicted washings were correct
    # Recall - How many real washings were detected
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    prc_auc = auc(recall, precision) # area under curve. higher better detection of label 1

    # ---------- Predictions ----------
    # specific predicted label
    y_pred = model.predict(X_test)

    # ---------- Sensitivity (Recall for Class 1) ----------
    # how many real handwashing events were detected
    # Sensitivity = TP / (TP + FN)
    # TN: Routine correctly classified
    # FP: Routine classified as Handwashing(false alarm)
    # FN: Missed handwashing(dangerous)
    # TP: Correct handwashing detection
    sensitivity = recall_score(y_test, y_pred, pos_label=1)

    # ---------- Confusion Matrices ----------
    #[[TN, FP],
    # [FN, TP]]
    confusion_matrx = confusion_matrix(y_test, y_pred)

    return {'model_name':model_name, 'roc_auc':roc_auc, 'fpr':fpr, 'tpr': tpr, 'precision': precision, 'recall':recall, 'prc_auc':prc_auc, 'sensitivity': sensitivity, 'confusion_matrix':confusion_matrx}

def plot_ROC(model_outputs, folder_name):
    plt.figure()

    for m in model_outputs:
        plt.plot(m['fpr'], m['tpr'], label=f"Model {m['model_name']} (AUC = {m['roc_auc']:.3f})")

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

def save_model_outputs_to_xlsx(model_outputs, folder_name):
    df = pd.DataFrame(model_outputs)
    columns_to_save = ['model_name', 'roc_auc', 'prc_auc', 'sensitivity', 'confusion_matrix']
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