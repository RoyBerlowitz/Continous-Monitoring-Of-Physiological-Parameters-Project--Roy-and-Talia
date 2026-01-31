import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import pickle
import pandas as pd


part_a_res_cache_path = "part_a_final_output.pkl"
with open(part_a_res_cache_path, "rb") as f:
        part_a_res = pickle.load(f)

split1_vet_features, split2_vet_features = part_a_res

X_vetting, X_test_norm, split2_Y_train, split2_Y_test, scaler = split2_vet_features

X_total = pd.concat([X_vetting, X_test_norm], axis=0)
Y_total = pd.concat([split2_Y_train, split2_Y_test], axis=0)
X_total ["recording identifier"] = X_total["Group number"] + X_total["Participant ID"]
groups = X_total["recording identifier"].values


administrative_features = ['recording identifier', 'First second of the activity', 'Last second of the activity', 'Participant ID', 'Group number','Recording number', 'Protocol']
columns_to_keep = [c for c in X_total.columns if c not in administrative_features]
X_total = X_total [columns_to_keep]


def plot_positive_class_distribution(X_df, Y_series, groups_array, output_file="features_distribution_label_1.pdf"):
    """
    יוצר קובץ PDF עם Boxplots שמראים את התפלגות הפיצ'רים בקבוצות השונות,
    אך ורק עבור דגימות של שטיפת ידיים (Label=1).
    """
    # 1. איחוד הכל לטבלה אחת לצורך נוחות
    # מוודאים שהאינדקסים מאופסים כדי למנוע בעיות חיבור
    plot_df = X_df.copy().reset_index(drop=True)
    plot_df['Label'] = Y_series.values
    plot_df['Group'] = groups_array

    # 2. הסינון הקריטי: משאירים רק שטיפות ידיים
    # המטרה: לראות איך נראית "שטיפת ידיים" אצל כל קבוצה בנפרד
    positive_df = plot_df[plot_df['Label'] == 1].copy()

    if len(positive_df) == 0:
        print("Error: No positive samples found!")
        return

    # רשימת הפיצ'רים לציור (בלי עמודות העזר)
    features = [c for c in plot_df.columns if c not in ['Label', 'Group']]

    print(f"Plotting {len(features)} features for Positive Class (Label=1) only...")
    print(f"Total positive samples: {len(positive_df)}")

    # 3. יצירת ה-PDF
    with PdfPages(output_file) as pdf:
        for i, feature in enumerate(features):
            plt.figure(figsize=(14, 6))

            # Boxplot שמראה את הטווח של הערכים עבור שטיפת ידיים בכל קבוצה
            sns.boxplot(x='Group', y=feature, data=positive_df, palette="Set3")

            # הוספת נקודות (Strip plot) יכולה לעזור לראות את כמות הדאטה,
            # אבל אם יש המון נקודות זה יכביד. נשאיר רק Boxplot.

            plt.title(f'Feature: "{feature}" (Label=1 Only)\nComparison across Groups', fontsize=14)
            plt.xlabel('Group Number', fontsize=12)
            plt.ylabel('Feature Value', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.5)

            plt.tight_layout()

            pdf.savefig()
            plt.close()

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(features)}...")

    print(f"Done! Saved to {output_file}")


# --- הפעלה ---
# הנחות:
# X_total ו-Y_total כבר מסודרים ותואמים בשורות.
# groups הוא מערך באותו אורך בדיוק כמו X_total.

plot_positive_class_distribution(X_total, Y_total, groups)