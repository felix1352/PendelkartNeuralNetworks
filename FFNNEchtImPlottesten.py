import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from load_pendel_data import LoadHdf5Mat
from keras.models import load_model

#Daten einlesen
#Gesamte Messdaten einlesen
path = Path('verlade_bruecke.mat')
data_upswing = LoadHdf5Mat(path)
time = data_upswing.get('X').get('Data')
data_names = data_upswing.get('Y').get('Name')
force = data_upswing.get('Y').get('Data')[0]
pos = data_upswing.get('Y').get('Data')[1]
angle = data_upswing.get('Y').get('Data')[3]
angle = angle*np.pi/180 #in rad umrechnen

angle_test = angle[10000:10100]
angle_test = angle_test % (2*np.pi)
angle1_test = np.sin(angle_test)
angle2_test = np.cos(angle_test)
force_test = force[10000:10100]
pos_test = pos[10000:10100]
time_test = time[10000:10100]

model = load_model('FFNN_EinfachpendelEchtdaten.keras')

angle_sin = angle1_test[4]
angle_sin_m1 = angle1_test[3]
angle_sin_m2 = angle1_test[2]
angle_sin_m3 = angle1_test[1]
angle_sin_m4 = angle1_test[0]
angle_cos = angle2_test[4]
angle_cos_m1 = angle2_test[3]
angle_cos_m2 = angle2_test[2]
angle_cos_m3 = angle2_test[1]
angle_cos_m4 = angle2_test[0]
x0 = pos_test[4]
xm1 = pos_test[3]
xm2 = pos_test[2]
xm3 = pos_test[1]
xm4 = pos_test[0]
tffnn = np.array([])
phi_sin_ffnn = np.array([])
phi_cos_ffnn = np.array([])
xffnn = np.array([])
current_step = 4
while current_step < len(time_test)-2:
    xffnn = np.append(xffnn, x0)
    tffnn = np.append(tffnn, time_test[current_step])
    phi_sin_ffnn = np.append(phi_sin_ffnn, angle_sin)
    phi_cos_ffnn = np.append(phi_cos_ffnn, angle_cos)
    input_data = np.array([[angle_sin, angle_sin_m1, angle_sin_m2, angle_sin_m3, angle_sin_m4, angle_cos, angle_cos_m1, angle_cos_m2, angle_cos_m3, angle_cos_m4, x0, xm1, xm2, xm3, xm4, force_test[current_step]]])
    ergplus1 = model.predict(input_data)   #Vorhersage schon beim ersten schritt sehr schlecht. keine ahnung warum
    angle_sin, angle_sin_m1, angle_sin_m2, angle_sin_m3, angle_sin_m4, angle_cos, angle_cos_m1, angle_cos_m2, angle_cos_m3, angle_cos_m4, x0, xm1, xm2, xm3, xm4 = ergplus1[0][0], angle_sin, angle_sin_m1, angle_sin_m2, angle_sin_m3, ergplus1[0][1], angle_cos, angle_cos_m1, angle_cos_m2, angle_cos_m3, ergplus1[0][2], x0, xm1, xm2, xm3
    current_step = current_step+1

print(0)
angle_pred = phi_sin_ffnn
plt.figure()
plt.subplot(1,2,1)
plt.title('Sinus von phi')
plt.plot(tffnn, phi_sin_ffnn, label='predicted')
plt.plot(time_test, angle1_test, label='echt')
plt.legend()
plt.subplot(1,2,2)
plt.title('Cosinus von phi')
plt.plot(tffnn, phi_cos_ffnn, label='predicted')
plt.plot(time_test, angle2_test, label='echt')
plt.legend()
plt.show()

for i in range(0, len(tffnn)):
    angle_pred[i] = np.arctan2(phi_sin_ffnn[i], phi_cos_ffnn[i])
    if angle_pred[i] < 0:
        angle_pred[i] += 2 * np.pi


plt.figure()
plt.subplot(3, 1, 1)
plt.plot(time_test, angle_test, label='gemessener Verlauf phi')
plt.plot(tffnn, angle_pred, label='vorhergesagter Verlauf phi')
plt.legend(loc='lower left')
plt.xlabel('Zeit in s')
plt.ylabel('Auslenkung in rad')
plt.subplot(3, 1, 2)
plt.plot(time_test, pos_test, label='gemessener Verlauf x')
plt.plot(tffnn, xffnn, label='vorhergesagter Verlauf x')
plt.legend(loc='lower left')
plt.xlabel('Zeit in s')
plt.ylabel('Auslenkung in m')
plt.subplot(3, 1, 3)
plt.plot(time_test, force_test)
plt.xlabel('Zeit in s')
plt.ylabel('Kraft')
plt.show()
