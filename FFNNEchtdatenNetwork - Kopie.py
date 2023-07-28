import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint
import optuna
#Das ist ein Feed Forward Neural Network in dem als Input die drei letzten Positionen eingegeben werden und als output die nächste Position bestimmt werden soll

#Modell definieren
model = Sequential()
model.add(Dense(units = 200, input_shape=(16,), activation='relu'))   #Eingangschicht
model.add(Dense(units = 200, activation='relu'))                #Mittelschichten
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=3))                                   #Ausgangsschicht

#Modell kompilieren
model.compile(optimizer='adam', loss='mse')

#Trainingsdaten einlesen und form anpassen und aufteilen in Trainings- und Testdaten
train_inputs = np.genfromtxt('inputdataechtdaten.csv',delimiter=',')
train_outputs = np.genfromtxt('outputdataechtdaten.csv',delimiter=',')
train_inputs, validate_inputs, train_outputs, validate_outputs = train_test_split(train_inputs, train_outputs, test_size=0.2, random_state=42)

#Hyperparameteroptimisierung
def create_model(trial):
    model = Sequential()
    model.add(Dense(units=trial.suggest_int('num_neurons', 50, 200), input_shape=(16,), activation='relu'))  # Eingangsschicht
    num_hidden_layers = trial.suggest_int('num_layers', 1, 3)  # Anzahl der Mittelschichten
    for _ in range(num_hidden_layers):
        model.add(Dense(units=trial.suggest_int('num_neurons', 50, 200), activation='relu'))  # Mittelschichten
    model.add(Dense(units=3))  # Ausgangsschicht
    model.compile(optimizer='adam', loss='mse')
    return model

def objective(trial):
    # Modell erstellen
    model = create_model(trial)

    # Callback für das Speichern des besten Modells
    checkpoint = ModelCheckpoint('best_model.hdf5', monitor='val_loss', verbose=0, save_best_only=True, mode='min')

    # Modell trainieren
    history = model.fit(train_inputs.reshape(-1, 4), train_outputs.reshape(-1, 1),
                        epochs=5, batch_size=1, callbacks=[checkpoint],
                        validation_data=(validate_inputs, validate_outputs), verbose=2)

    # Rückgabe des validierungsfehlers als Zielfunktion für die Optimierung
    return history.history['val_loss'][-1]

# Hyperparameteroptimierung mit Optuna
#study = optuna.create_study(direction='minimize')
#study.optimize(objective, n_trials=10)

# Ausgabe der besten Hyperparameter
#best_params = study.best_params
#print('Beste Hyperparameter:', best_params)

#Modell trainieren
filepath = "\\Users\\Felix Koch\\Documents\\BachelorArbeit\\ModelsFFnn\\weights-{epoch:02d}-{val_loss:.7f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
callbacks_list = [checkpoint]
#history = model.fit(train_inputs, train_outputs, epochs=20, batch_size=16, callbacks=callbacks_list, validation_data=(validate_inputs,validate_outputs), verbose=2)
model.load_weights("\\Users\\Felix Koch\\Documents\\BachelorArbeit\\ModelsFFnn\\weights-18-0.0000023.hdf5")

model.save('FFNN_EinfachpendelEchtdaten.keras')

#Plotten von Trainingsloss und Validationloss
#plt.figure()
#plt.plot(history.epoch,history.history['loss'], label='loss')
#plt.plot(history.epoch,history.history['val_loss'], label='val_loss')
#plt.legend()
#plt.show()