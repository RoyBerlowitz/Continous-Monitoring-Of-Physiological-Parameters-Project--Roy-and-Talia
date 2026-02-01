import numpy as np
from hmmlearn import hmm


def prepare_data_for_hmm(seconds_df, target):
    # This function is meant to create the suitable data for the markov model
    # we keep the non-admnistrative data for the model, as we want it to calculate only based on the probabilities.
    X = seconds_df[["prob_1", "prob_2", "prob_3", "prob_4"]].values
    # we define the target
    y = target.values

    # we calculaye the lenghts of recording, as it is important for the viterbi algorithm so it can stop when obtaining the right number of seconds
    lengths = seconds_df.groupby(['recording_identifier']).size().values

    return X, y, lengths


def calculate_real_transitions(y_train, lengths):
    # this function is meant to find the best transition matrix, based on the transition probabilities in the dataset

    current_idx = 0
    # the number of 0->1 transitions
    off_to_on = 0
    # the number of 1->0 transitions
    on_to_off = 0
    # the number of 0 and 1 in total
    total_0 = 0
    total_1 = 0

    # we go over each recording separately
    for length  in lengths:
        # we get the labels of the recording until its end
        y_seg = y_train[current_idx: current_idx + length]
        #  we sum the number of states changes
        off_to_on += np.sum((y_seg[:-1] == 0) & (y_seg[1:] == 1))
        on_to_off += np.sum((y_seg[:-1] == 1) & (y_seg[1:] == 0))
        # we get the total number of zeros and ones
        total_0 += np.sum(y_seg[:-1] == 0)
        total_1 += np.sum(y_seg[:-1] == 1)
        current_idx += length

    # here we compute the transition probabilities
    # transition from non-handwashing to handwashing - the relative number of non-handwashing points that were followed by handwashing points
    p_01 = off_to_on / max(total_0, 1)
    # transition from handwashing to non-handwashing - the relative number of handwashing points that were followed by non-handwashing points
    p_10 = on_to_off / max(total_1, 1)
    # transition from non-handwashing to non-handwashing
    p_00 = 1 - p_01
    # transition from handwashing to handwashing
    p_11 = 1 - p_10

    return np.array([[p_00, p_01], [p_10, p_11]])
# def calculate_real_transitions(y_train):
#
#     # we count the transition between the different states:
#     # 0->0, 0->1, 1->0, 1->1
#     total_0 = np.sum(y_train == 0)
#     total_1 = np.sum(y_train == 1)
#
#     #  we sum the number of states changes
#     off_to_on = np.sum((y_train[:-1] == 0) & (y_train[1:] == 1))
#     on_to_off = np.sum((y_train[:-1] == 1) & (y_train[1:] == 0))
#
#
#     p_01 = off_to_on / total_0
#     # transition from non-handwashing to non-handwashing
#     p_00 = 1 - p_01
#     # transition from handwashing to non-handwashing - the relative number of handwashing points that were followed by non-handwashing points
#     p_10 = on_to_off / total_1
#     # transition from handwashing to handwashing
#     p_11 = 1 - p_10
#     return np.array([[p_00, p_01], [p_10, p_11]])

def train_supervised_hmm(X_train, y_train, lengths):
    # We use Markov Model to try and classify the point
    # The idea is simple - try and predict the next point based on the former point and the probabilites for moving.
    # prob_1, prob_2, prob_3, prob_4 - the probability of the different windows. Those are our observations

    # we start by calculating the mean for each label
    means = np.array([X_train[y_train == i].mean(axis=0) for i in [0, 1]])
    # we also find the covariance matrix
    # the reason for the 0.001 addition is to make a threshold for the covariance which enhances stabillity and make the model less strict
    covariances = np.array([np.var(X_train[y_train == i], axis=0) + 1e-3 for i in [0, 1]])

    # we find the transition matrix based on the statistic of the data
    transmat = calculate_real_transitions(y_train, lengths)

    # we define the markov model, we have two classes so we have 2 components, and the covariance is diagonal by how we defined
    # model = hmm.GMMHMM(n_components=2, covariance_type="diag", n_mix = 1)
    model = hmm.GaussianHMM(n_components=2, covariance_type="diag")

    # we start always without handwashing
    model.startprob_ = np.array([1.0, 0.0])
    model.transmat_ = transmat
    # we use the means and covariances to identify and distinguish between events - as we have the learning.
    # The algorithm is viterbi algorithm that tries to find the most probable way.
    # the mean and covariance helps him to identify each seconds and to determine what is more probable.
    model.means_ = means
    model.covars_ = covariances

    # we force the model to use our parameters
    model.init_params = ""

    # # we fit the model to find the most probable track
    # model.fit(X_train, lengths)
    return model