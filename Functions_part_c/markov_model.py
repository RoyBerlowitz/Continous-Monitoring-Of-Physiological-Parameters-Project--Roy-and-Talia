import numpy as np
from hmmlearn import hmm


def prepare_data_for_hmm(seconds_df, target):
    # אנחנו משתמשים בעמודות ההסתברות שיצרנו (prob_1 עד prob_4)
    # ניתן גם להשתמש בממוצע המשוקלל שלהן כפי שדיברנו
    X = seconds_df[["prob_1", "prob_2", "prob_3", "prob_4"]].values
    y = target.values

    # חישוב אורכי ההקלטות - קריטי ל-HMM
    lengths = seconds_df.groupby(['recording_identifier']).size().values

    return X, y, lengths


def train_supervised_hmm(X_train, y_train, lengths):
    # 1. חישוב ממוצעים וסטיות תקן לכל מצב (0 ו-1) מהטריין
    means = np.array([X_train[y_train == i].mean(axis=0) for i in [0, 1]])
    # שימוש ב-diag covariance לשיפור היציבות
    covars = np.array([np.var(X_train[y_train == i], axis=0) for i in [0, 1]])

    # 2. אתחול מטריצת מעברים (ניתן גם לחשב מהלייבלים)
    # כאן הגדרנו 98% סיכוי להישאר באותו מצב (החלקה חזקה)
    transmat = np.array([[0.98, 0.02],
                         [0.05, 0.95]])

    model = hmm.GaussianHMM(n_components=2, covariance_type="diag")

    # הזרקת הפרמטרים
    model.startprob_ = np.array([0.9, 0.1])  # נניח שמתחילים בשגרה
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars

    # מניעת אתחול רנדומלי - השתמש במה שנתנו
    model.init_params = ""

    # אימון (עדכון עדין של הפרמטרים)
    model.fit(X_train, lengths)
    return model