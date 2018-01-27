
"""
To deploy your own serialized model for deployment to a stream of live data
and to calculate the score of your model on the hidden data implement
the functions:

row_prediction which takes a list as input in the order:
[Vol, S1, S2, ..., S27, S28];

vectorized_prediction which takes a np.ndarry as input with features as
columns in the same order and entries as rows.
"""
# these imports are needed for the endpoints which you do not need to change
from runtime.framework import (endpoint, argument, returns, visualization)
from runtime.schema import fields
import numpy as np
from sklearn.metrics import mean_squared_error
from amld_functions import (get_data, get_score, plot_confmat)

# model dependent libraries
from keras.models import load_model
import keras.backend as K
from sklearn.externals import joblib


# global variables can be loaded into memory at deployment to improve efficiency
# be careful if using tensorflow globally
g_scaler = joblib.load('scaler.model')
g_threshold99 = 3.8257


def row_prediction(row):
    """
    This function should return the prediction of your model on a single row
    of data.
    Inputs:
    param row: np.array shape (, 29) with feature order
               [Vol, S1, S2, ..., S27, S28]
    Returns:
    <bool>: True for predicted anomaly, False otherwise
    """
    # apply the globally loaded feature scaler
    scaled = g_scaler.transform(np.atleast_2d(row))
    # load the trained autoencoder model
    ae_model = load_model('model.h5')
    # get predicted autoencoder output for the scaled input row
    predicted = ae_model.predict(scaled)
    # tidy up the Keras backend for this call
    K.clear_session()
    # calculate the mean squared error for this row
    mse = mean_squared_error(predicted.T, scaled.T)
    # return True if the mse is over the 99% threshold for non-anomalous data
    return (mse > g_threshold99)


def vectorized_prediction(matrix):
    """
    This function should return a vector of predictions of your model on an
    input matrix of batch data with m examples.
    Inputs:
    param matrix: np.ndarray shape (m, 29) with feature order
                  [Vol, S1, S2, ..., S27, S28]
    Returns:
    <np.ndarray>: shape (m, 1) with m_i True for predicted anomaly and False
                  otherwise
    """
    # apply the globally loaded feature scaler
    scaled = g_scaler.transform(matrix)
    # load the trained autoencoder model
    ae_model = load_model('model.h5')
    # get predicted autoencoder output for the scaled input batch data
    predicted = ae_model.predict(scaled)
    # calculate the mean squared error for all examples
    mse = mean_squared_error(predicted.T, scaled.T, multioutput='raw_values')
    # return vector with True if the mse is over the 99% threshold
    return (mse > g_threshold99)


###
# You do not need to edit below this line but are encouraged to understand
# the implementation of the endpoints, just ask if you want more information
# about endpoint implementation and functionality!
###


@endpoint()
@argument("Vol", type=float, description="Value of input sensor Vol")
@argument("S1", type=float, description="Value of input sensor S1")
@argument("S2", type=float, description="Value of input sensor S2")
@argument("S3", type=float, description="Value of input sensor S3")
@argument("S4", type=float, description="Value of input sensor S4")
@argument("S5", type=float, description="Value of input sensor S5")
@argument("S6", type=float, description="Value of input sensor S6")
@argument("S7", type=float, description="Value of input sensor S7")
@argument("S8", type=float, description="Value of input sensor S8")
@argument("S9", type=float, description="Value of input sensor S9")
@argument("S10", type=float, description="Value of input sensor S10")
@argument("S11", type=float, description="Value of input sensor S11")
@argument("S12", type=float, description="Value of input sensor S12")
@argument("S13", type=float, description="Value of input sensor S13")
@argument("S14", type=float, description="Value of input sensor S14")
@argument("S15", type=float, description="Value of input sensor S15")
@argument("S16", type=float, description="Value of input sensor S16")
@argument("S17", type=float, description="Value of input sensor S17")
@argument("S18", type=float, description="Value of input sensor S18")
@argument("S19", type=float, description="Value of input sensor S19")
@argument("S20", type=float, description="Value of input sensor S20")
@argument("S21", type=float, description="Value of input sensor S21")
@argument("S22", type=float, description="Value of input sensor S22")
@argument("S23", type=float, description="Value of input sensor S23")
@argument("S24", type=float, description="Value of input sensor S24")
@argument("S25", type=float, description="Value of input sensor S25")
@argument("S26", type=float, description="Value of input sensor S26")
@argument("S27", type=float, description="Value of input sensor S27")
@argument("S28", type=float, description="Value of input sensor S28")
@returns("predicted_class", type=float,
         description="Prediction as a float: 1.0 if anomaly, 0.0 otherwise")
def stream_prediction(
    Vol,
    S1,  S2,  S3,  S4,  S5,  S6,  S7,  S8,  S9,  S10,
    S11, S12, S13, S14, S15, S16, S17, S18, S19, S20,
    S21, S22, S23, S24, S25, S26, S27, S28
):
    """
    This endpoint is used to connect a live stream of data to a model and
    return predictions in real time for each new feature row as it arrives
    from a remote location.
    """
    # take the inputs and put them into and ordered np.array shape (, 29)
    feature_list = np.array([
        Vol,
        S1,  S2,  S3,  S4,  S5,  S6,  S7,  S8,  S9,  S10,
        S11, S12, S13, S14, S15, S16, S17, S18, S19, S20,
        S21, S22, S23, S24, S25, S26, S27, S28
    ])
    # return the predicted output from the function row_prediction
    return float(row_prediction(feature_list))


@endpoint()
@argument("sas_token", type=str,
          description="To score your model with the hidden data enter the "
                      "sas_token provided by workshop conveners, not needed "
                      "for local test", default="")
@returns("score", type=float, description="The score of the deployed model "
                                           "on the hidden data")
@visualization(tab='confusion_matrix.html')
def calculate_score(sas_token):
    """
    This endpoint is used to load some hidden batch data from the cloud and
    calculate the score of the deployed model. It will also display the
    confusion matrix of the model in a tab on the model execution page.
    """
    # get the input data to calculate the score
    X, y = get_data(sas_token)
    # get the prediction vector from the function vectorized_prediction
    prediction = vectorized_prediction(X)
    # convert the predicted data to the same format as the hidden targets
    y_pred = ['normal' if not ipred else 'fraud' for ipred in prediction]
    # plot and save the confusion matrix for display in the model page
    plot_confmat(y, y_pred)
    # return the model's score and the name of the saved plot to render
    return get_score(y, y_pred)