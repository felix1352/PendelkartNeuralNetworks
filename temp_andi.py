"""
This file is used by Andreas to try some stuff.

It may contain duplicated code and only exists for the reason that I do not want to modify your code.
"""

from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint
from dynamics import get_dxdt
from scipy.integrate import solve_ivp
import numpy as np
from pathlib import Path
from keras.models import load_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

STATE_NAMES = ['phi', 'dphi', 's', 'ds']

STATES_MEASURABLE = True
HISTORY_SIZE = 5  # number of past angles that are considered

TS = 0.01
CART_POLE_PARAMETER = {
    'g': 9.81,
    'M': 0.7,
    'm': 0.221,
    'l': 0.5,
    'b': 0.02,
    'c': 1
}
NUMBER_OF_TRAINING_SIMULATIONS = 200
RANDOM_INITIAL_STATE = True
EPOCHS = 30

DX_DT = get_dxdt()


def rhs_cartpole(t: int, x: np.ndarray, F) -> np.ndarray:
    return DX_DT(t, x, parameter=CART_POLE_PARAMETER, F=F)


def revert_modulo(angles, threshold=np.pi):
    full_turns = 0
    new_angles = [angles[0]]
    for i in range(1, len(angles)):
        if abs(angles[i - 1] - angles[i]) > threshold:
            turns_in_positive_direction = angles[i] - angles[i - 1] > 0
            if turns_in_positive_direction:
                full_turns -= 1
            else:
                full_turns += 1
        new_angles.append(angles[i] + 2 * np.pi * full_turns)

    return np.array(new_angles)


def get_observation_and_target(state_sequence, force_sequence):
    if STATES_MEASURABLE:
        return get_observation_and_target_state_measurable(state_sequence, force_sequence)
    else:
        return get_observation_and_target_history_angles(state_sequence, force_sequence)


def get_input_and_output_size():
    if STATES_MEASURABLE:
        return 6, 5
    else:
        return 3 * HISTORY_SIZE + 1, 3


def get_observation_and_target_history_angles(state_sequence, force_sequence):
    X = []
    for i in range(HISTORY_SIZE - 1, -1, -1):
        X.append(np.sin(state_sequence[i:-HISTORY_SIZE + i, 0]))
    for i in range(HISTORY_SIZE - 1, -1, -1):
        X.append(np.cos(state_sequence[i:-HISTORY_SIZE + i, 0]))
    for i in range(HISTORY_SIZE - 1, -1, -1):
        X.append(state_sequence[i:-HISTORY_SIZE + i, 2])
    X.append(force_sequence[HISTORY_SIZE - 1:])
    X = np.array(X).transpose()  # cos(phi_k), cos(phi_k-1), ..., cos(phi_k-History-1), sin(phi_k), ....

    Y = np.array([
        np.sin(state_sequence[HISTORY_SIZE:, 0]),
        np.cos(state_sequence[HISTORY_SIZE:, 0]),
        state_sequence[HISTORY_SIZE:, 2],
    ]).transpose()
    Y = Y - X[:, [HISTORY_SIZE * i for i in range(3)]]  # only learn the change of the state
    return X, Y


def get_observation_and_target_state_measurable(state_sequence, force_sequence):
    X_part = np.array([
        np.sin(state_sequence[:-1, 0]),  # sin(phi)
        np.cos(state_sequence[:-1, 0]),  # cos(phi)
        state_sequence[:-1, 1],  # dphi
        state_sequence[:-1, 2],  # s
        state_sequence[:-1, 3],  # ds
        force_sequence  # F
    ]).transpose()
    Y_part = np.array([
        np.sin(state_sequence[1:, 0]),  # sin(phi)
        np.cos(state_sequence[1:, 0]),  # cos(phi)
        state_sequence[1:, 1],  # dphi
        state_sequence[1:, 2],  # s
        state_sequence[1:, 3],  # ds
    ]).transpose()
    Y_part = Y_part - X_part[:, :-1]  # only learn the change of the state
    return X_part, Y_part


def forward_prediction_FFNN(model, force, forward_simulation_result, normalisation_factors):
    mean_X, std_X, mean_Y, std_Y = normalisation_factors

    if STATES_MEASURABLE:
        forward_prediction = [forward_simulation_result[0]]
        for i in range(len(forward_simulation_result) - 1):
            current_input = get_observation_from_state(forward_prediction[-1], force[i])
            current_input_norm = (current_input - mean_X) / std_X
            Y_pred_norm = model(current_input_norm).numpy().flatten()
            Y_pred = Y_pred_norm * std_Y + mean_Y
            forward_prediction.append(get_state_from_prediction(current_input, Y_pred))
    else:
        forward_prediction = []
        for i in range(HISTORY_SIZE):
            forward_prediction.append(forward_simulation_result[i])
        for i in range(len(forward_simulation_result) - HISTORY_SIZE):
            last_state_sequence = np.array(forward_prediction[i:i + HISTORY_SIZE])
            last_state_sequence = np.vstack((last_state_sequence, np.zeros((1, 4)) * np.nan))
            current_input, _ = get_observation_and_target_history_angles(last_state_sequence,
                                                                         force[i:i + HISTORY_SIZE])
            current_input_norm = (current_input - mean_X) / std_X
            Y_pred_norm = model(current_input_norm).numpy().flatten()
            Y_pred = Y_pred_norm * std_Y + mean_Y
            Y = (Y_pred + current_input[:, [HISTORY_SIZE * i for i in range(3)]]).flatten()
            angle = np.arctan2(Y[0], Y[1])
            predicted_state = np.array([angle, np.nan, Y[2], np.nan])
            forward_prediction.append(predicted_state)

    forward_prediction = np.array(forward_prediction)
    forward_prediction[:, 0] = revert_modulo(forward_prediction[:, 0])
    return forward_prediction


def get_observation_from_state(state, force):
    return np.array([
        np.sin(state[0]),  # sin(phi)
        np.cos(state[0]),  # cos(phi)
        state[1],  # dphi
        state[2],  # s
        state[3],  # ds
        force  # F
    ]).reshape(1, 6)


def get_state_from_prediction(X, Y_pred):
    assert X.shape == (1, 6)
    Y = X[0, :-1] + Y_pred
    angle = np.arctan2(Y[0], Y[1])
    return np.array([angle,
                     Y[2],
                     Y[3],
                     Y[4]]).reshape((4,))


def test_model(model, normalization_values):
    fig, axs = plt.subplots(4 + 1)
    fig.suptitle('Forward Simulation and Network prediction')
    for j in range(1):
        if j == 0:
            x0 = np.array([0.4, 0, 0, 0])
            force_vector = np.sin(np.arange(int(20 / TS)) * TS)
            label1 = 'true_test'
            label2 = 'pred_test'
        else:
            force_vector = generate_random_force_vector(1000, 1).flatten()
            x0 = np.random.uniform(low=[0, -5, -0.5, -0.3],
                                   high=[2 * np.pi, 5, 0.5, 0.3])
            label1 = 'true_train_distribution'
            label2 = 'pred_train_distribution'
        state_sequence = simulate_cart_pole(x0, force_vector)
        forward_prediction = forward_prediction_FFNN(model, force_vector, state_sequence, normalization_values)

        n = len(STATE_NAMES)
        t_eval = np.arange(len(force_vector) + 1) * TS
        for i in range(n):
            axs[i].plot(t_eval, state_sequence[:, i], label=label1)
            axs[i].plot(t_eval, forward_prediction[:, i], label=label2)
            axs[i].set_ylabel(STATE_NAMES[i])
        # axs[i].set_ylim([state_sequence[:, i].min() - 0.5, state_sequence[:, i].max() + 0.5])
        axs[-1].plot(t_eval[:-1], force_vector)
        axs[-1].set_ylabel('force')
        axs[0].legend()
        # fig.tight_layout()

    fig2, axs2 = plt.subplots(n)
    fig2.suptitle('Forward Simulation prediction error')
    for i in range(n):
        axs2[i].plot(t_eval, state_sequence[:, i] - forward_prediction[:, i], label='true-pred')
        axs2[i].set_ylabel(STATE_NAMES[i])
        axs2[i].set_ylim([-0.5, 0.5])
    axs2[0].legend()
    fig2.tight_layout()


def generate_training_data():
    input_dim, output_dim = get_input_and_output_size()
    X = np.empty((0, input_dim))
    Y = np.empty((0, output_dim))

    force_vectors = generate_random_force_vector(1000, NUMBER_OF_TRAINING_SIMULATIONS)
    for i in range(NUMBER_OF_TRAINING_SIMULATIONS):
        if RANDOM_INITIAL_STATE:
            x0 = np.random.uniform(low=[0, -5, -0.5, -0.3],
                                   high=[2 * np.pi, 5, 0.5, 0.3])
        else:
            x0 = np.zeros(4)

        force_vector = force_vectors[i]
        state_sequence = simulate_cart_pole(x0, force_vector)

        X_part, Y_part = get_observation_and_target(state_sequence, force_vector)
        X = np.vstack((X, X_part))
        Y = np.vstack((Y, Y_part))

    return X, Y


def simulate_cart_pole(x0, force_vector):
    states = [x0]
    for i in range(len(force_vector)):
        result_time_step = solve_ivp(rhs_cartpole, t_span=[i * TS, (i + 1) * TS], y0=states[-1],
                                     args=(force_vector[i],))
        states.append(result_time_step.y[:, 1])

    return np.array(states)


def generate_random_force_vector(vector_length, number_of_vectors):
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
    gpr = GaussianProcessRegressor(kernel=kernel)

    x = np.linspace(0, 10, vector_length)
    X = x.reshape(-1, 1)

    y_samples = gpr.sample_y(X, number_of_vectors, random_state=np.random.randint(0, 2 ** 32 - 1))
    return y_samples


def main():
    input_dim, output_dim = get_input_and_output_size()

    model = Sequential()
    model.add(Dense(units=200, input_shape=(input_dim,), activation='relu'))
    model.add(Dense(units=200, activation='relu'))
    model.add(Dense(units=200, activation='relu'))
    model.add(Dense(units=output_dim))

    model.compile(optimizer='adam', loss='mse')

    X, Y = generate_training_data()
    mean_X, std_X = X.mean(axis=0), X.std(axis=0) + 1e-8  # add jitter to ensure std is not zero
    mean_Y, std_Y = Y.mean(axis=0), Y.std(axis=0) + 1e-8

    X = (X - mean_X) / std_X
    Y = (Y - mean_Y) / std_Y

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    filepath = Path() / 'ModelsFFnn' / f'cartpole_NN.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False,
                                 mode='min')
    history = model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=512, callbacks=[checkpoint],
                        validation_data=(X_val, Y_val), verbose=2)
    model = load_model(filepath)

    Y_val_pred = model(X_val).numpy()
    mse_val = (1 / Y_val.shape[1] * np.sum((Y_val_pred - Y_val) ** 2, axis=1)).mean()
    print(f'mse: {mse_val}')

    test_model(model, (mean_X, std_X, mean_Y, std_Y))

    plt.figure()
    plt.semilogy(history.history['loss'])
    plt.semilogy(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

    plt.show()


if __name__ == '__main__':
    main()
