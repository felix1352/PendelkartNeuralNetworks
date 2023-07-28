import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from load_pendel_data import LoadHdf5Mat
import csv

plt.close('all')
#Daten einlesen
#Gesamte Messdaten einlesen
path = Path('verlade_bruecke.mat')
data_upswing = LoadHdf5Mat(path)
time = data_upswing.get('X').get('Data')
data_names = data_upswing.get('Y').get('Name')
force = data_upswing.get('Y').get('Data')[0]
pos = data_upswing.get('Y').get('Data')[1]
angle = data_upswing.get('Y').get('Data')[3]
angle = angle*np.pi/180


angle_train = angle[5000:50000]
time_train = time[5000:50000]
force_train = force[5000:50000]
pos_train = pos[5000:50000]

plt.subplot(1,3,1)
plt.plot(time_train, angle_train)
plt.subplot(1,3,2)
plt.plot(time_train, force_train)
plt.subplot(1,3,3)
plt.plot(time_train, pos_train)
#plt.show()

angle1_train = np.sin(angle_train)
angle2_train = np.cos(angle_train)

train_inputs = np.empty((1, 16))
train_outputs = np.empty((1, 3))
for i in range(4, len(time_train)-2):
    input = np.array([angle1_train[i], angle1_train[i-1], angle1_train[i-2], angle1_train[i-3], angle1_train[i-4], angle2_train[i], angle2_train[i-1], angle2_train[i-2], angle2_train[i-3], angle2_train[i-4], pos_train[i], pos_train[i-1], pos_train[i-2], pos_train[i-3], pos_train[i-4], force_train[i]])
    output = np.array([angle1_train[i+1], angle2_train[i+1], pos_train[i+1]])
    if i == 4:
        train_inputs = input
        train_outputs = output
    else:
        train_inputs = np.vstack((train_inputs, input))
        train_outputs = np.vstack((train_outputs, output))

with open('inputdataechtdaten.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in train_inputs:
        writer.writerow(row)

with open('outputdataechtdaten.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in train_outputs:
        writer.writerow(row)

