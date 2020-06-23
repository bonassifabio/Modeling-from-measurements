import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spint
import scipy.io as sio
from os.path import isfile, isdir

def lorenz(t, xyzr, s=10, b=8/3):
    x, y, z, r = xyzr
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    r_dot = 0   # rho is constant
    return x_dot, y_dot, z_dot, r_dot

dt = 0.01
T_end = 10

# Time vector
try:
    data = sio.loadmat('data/lorenz_data.mat')
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_val = data['X_val']
    Y_val = data['Y_val']
    t = data['t']
    print('Lorenz trajectories imported from file')

except FileNotFoundError:
    N_exp = 2500
    N_val = 500
    t = np.arange(0, T_end, dt)
    X_train = np.empty((4, 0))
    Y_train = np.empty((3, 0))
    X_val = np.empty((4, 0))
    Y_val = np.empty((3, 0))

    for i in range(0, N_exp + N_val):
        rho_val = np.random.choice([10, 28, 40], p=[0.2, 0.4, 0.4])
        y0 = np.append(30 * (np.random.random((3,)) - 0.5), rho_val)

        sol = spint.solve_ivp(lorenz, y0=y0, t_span=[0, T_end], t_eval=t, atol=1e-10, rtol=1e-9)

        if i < N_exp:
            X_train = np.concatenate((X_train, sol.y[:, 0:-1]), axis=1)
            Y_train = np.concatenate((Y_train, sol.y[:-1, 1:]), axis=1)
        else:
            X_val = np.concatenate((X_val, sol.y[:, 0:-1]), axis=1)
            Y_val = np.concatenate((Y_val, sol.y[:-1, 1:]), axis=1)

    sio.savemat('data/lorenz_data.mat',
                {'X_train': X_train, 'Y_train': Y_train, 'X_val': X_val, 'Y_val': Y_val, 't': t},
                do_compression = True)
    print('Lorenz trajectories dumped to file')

#%%
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, LeakyReLU, Add
from tensorflow.keras.callbacks import EarlyStopping
import os, signal

norm_mean = np.mean(X_train, axis=1).reshape((4, 1))
norm_std = np.std(X_train, axis=1).reshape((4, 1))

X_train_norm = (X_train - norm_mean) / norm_std
Y_train_norm = (Y_train - norm_mean[:-1,:]) / norm_std[:-1,:]
X_val_norm = (X_val - norm_mean) / norm_std
Y_val_norm = (Y_val - norm_mean[:-1,:]) / norm_std[:-1,:]

tf.enable_eager_execution()

if isfile('saved/trained_network_part3a') or isdir('saved/trained_network_part3a'):
    nn = tf.keras.models.load_model('saved/trained_network_part3a')
    print('Pre-loaded NN model imported')

else:
    class CatchUserInterruptCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if isfile('STOP.txt'):
                print('Stopping on user request...')
                self.stopped_epoch = epoch
                self.model.stop_training = True
                os.remove('STOP.txt')

    x0 = Input(shape=(4,))
    x1 = Dense(units=30, activation='tanh')(x0)
    x2 = LeakyReLU(alpha=0.05)(Dense(units=30)(x1))
    x3 = Dense(units=30, activation='tanh')(x2)
    x4 = Add()([x1, x3])
    x5 = LeakyReLU(alpha=0.05)(Dense(units=30)(x4))
    x6 = Dense(units=30, activation='tanh')(x5)
    x7 = Add()([x4, x6])
    x8 = Dense(units=3, activation='linear')(x7)

    nn = Model(x0, x8)
    nn.compile(tf.keras.optimizers.Adam(4e-4), loss='mse')
    esr = EarlyStopping(monitor='val_loss', verbose=1, restore_best_weights=True, patience=100)
    nn.summary()
    nn.fit(X_train_norm.T, Y_train_norm.T, validation_data=(X_val_norm.T, Y_val_norm.T),
           epochs=2000, batch_size=1250, shuffle=True, callbacks=[CatchUserInterruptCallback(), esr])

    nn_json = nn.to_json()
    nn.save('saved/trained_network_part3a')
    print("Neural network trained and dumped to file")

#%%
y0 = np.append(30 * (np.random.random((3,)) - 0.5), 28) # Rho is managed as an initial state
sol = spint.solve_ivp(lorenz, y0=y0, t_span=[0, T_end], t_eval=t, atol=1e-10, rtol=1e-9)
sol_true = sol.y[0:3, :]
sol = spint.solve_ivp(lorenz, y0=y0, t_span=[0, T_end], t_eval=t, atol=1e-5, rtol=1e-4)
sol_app = sol.y[0:3, :]

x0 = (y0.reshape((4,1)) - norm_mean) / norm_std
sol_nn = np.zeros(sol_true.shape)
for i in range(0, sol_nn.shape[1]):
    x_next = nn.predict(x0.T).T
    sol_nn[:, i] = (x_next * norm_std[:-1,:] + norm_mean[:-1,:]).reshape((3,))
    x0 = np.append(x_next, x0[-1].reshape(1, 1), axis=0)

t_sol = np.linspace(0, T_end, sol_nn.shape[1])

fig = plt.figure(3)
ax3 = plt.axes(projection='3d')
ax3.plot3D(sol_true[0, :], sol_true[1, :], sol_true[2, :], 'b:')
ax3.plot3D(sol_app[0, :], sol_app[1, :], sol_app[2, :], 'g')
ax3.plot3D(sol_nn[0, :], sol_nn[1, :], sol_nn[2, :], 'r--')
ax3.scatter3D(sol_true[0, 0], sol_true[1, 0], sol_true[2, 0])
ax3.scatter3D(sol_nn[0, 0], sol_nn[1, 0], sol_nn[2, 0])
plt.legend(['Real', 'ODE', 'NN'])
plt.show()

fig2 = plt.figure(4)
plt.suptitle("Lorenz system")
plt.subplot(3, 1, 1)
plt.plot(t_sol, sol_true[0, :], 'b:')
plt.plot(t_sol, sol_app[0, :], 'g')
plt.plot(t_sol, sol_nn[0, :], 'r--')
plt.legend(["Real", "ODE", "NN"])
plt.grid()
plt.xlim((0, T_end))
plt.xlabel("t [s]")
plt.ylabel("x")

plt.subplot(3, 1, 2)
plt.plot(t_sol, sol_true[1, :], 'b:')
plt.plot(t_sol, sol_app[1, :], 'g')
plt.plot(t_sol, sol_nn[1, :], 'r--')
plt.legend(["Real", "ODE", "NN"])
plt.grid()
plt.xlim((0, T_end))
plt.xlabel("t [s]")
plt.ylabel("y")

plt.subplot(3, 1, 3)
plt.plot(t_sol, sol_true[2, :], 'b:')
plt.plot(t_sol, sol_app[2, :], 'g')
plt.plot(t_sol, sol_nn[2, :], 'r--')
plt.legend(["Real", "ODE", "NN"])
plt.grid()
plt.xlim((0, T_end))
plt.xlabel("t [s]")
plt.ylabel("z")

plt.show()