import numpy as np
from dynamics import get_dxdt
from scipy.integrate import solve_ivp
import csv

dx_dt = get_dxdt()

#Modell definieren

#Daten einlesen

dt = 0.01
t_start = 0.0
t_end = 10

def F_func(t):
    return np.sin(t)

default_parameter = {
        'g': 9.81,
        'M': 0.7,
        'm': 0.221,
        'l': 0.5,
        'b': 0.02,
        'c': 1
    }

t_eval = np.arange(t_start, t_end, dt) #Vektor der für gleichmäßige Zeitabstände beim Lösen sorgt, später wichtig wegen realer Abtastung

def berechne_verlauf(x0):
    # x0 muss nicht mehr modifiziert werden
    counter = 0
    x0 = [x0[0], 0, x0[1], 0]
    res = solve_ivp(lambda t, x: dx_dt(t, x, default_parameter, F=F_func(t)), [t_start, t_end], x0, t_eval=t_eval)
    erg = np.vstack([res.y[0], res.y[2]])
    return erg




#Trainingsdaten generieren
num_samples = 1;
train_inputs = np.empty((1, 7))
train_outputs = np.empty((1, 2))
#startwerte = np.ones((num_samples, 2), dtype=int)  #auswenig lernen
#startwerte = np.random.uniform(2, 5, size=(num_samples, 2))
startwerte = np.zeros((num_samples, 2), dtype=int)
a = 0
for x0 in startwerte:
        # Generiere Datenverlauf für den aktuellen Startwert
    winkelverlauf = berechne_verlauf(x0)
    for i in range(2, len(winkelverlauf[0])-2):
        input = np.array([winkelverlauf[0][i], winkelverlauf[0][i-1], winkelverlauf[0][i-2], winkelverlauf[1][i], winkelverlauf[1][i-1], winkelverlauf[1][i-2], F_func(i*dt)])
        output = np.array([winkelverlauf[0][i+1], winkelverlauf[1][i+1]])
        if a == 0:
            train_inputs = input
            train_outputs = output
            a = 1
        else:
            train_inputs = np.vstack((train_inputs, input))
            train_outputs = np.vstack((train_outputs, output))

with open('inputdata.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in train_inputs:
        writer.writerow(row)

with open('outputdata.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in train_outputs:
        writer.writerow(row)