import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from dynamics import get_dxdt
from scipy.integrate import solve_ivp
from keras.models import load_model

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

def berechne_verlauf(x0, F):
    # x0 muss nicht mehr modifiziert werden
    counter = 0
    x0 = [x0[0], 0, x0[1], 0]
    res = solve_ivp(lambda t, x: dx_dt(t, x, default_parameter, F=F_func(t)), [t_start, t_end], x0, t_eval=t_eval)
    erg = np.vstack([res.y[0], res.y[2]])
    return erg

x0 = [0, 0]
erg = berechne_verlauf(x0, 0)
phidata = erg[0]
xdata = erg[1]


#Verlauf mit FFNN vorhersagen

model = load_model('FFNN_Einfachpendel.keras')

phi0 = phidata[2]
phim1 = phidata[1]
phim2 = phidata[0]
x0 = xdata[2]
xm1 = xdata[1]
xm2 = xdata[0]
tffnn = np.array([])
phiffnn = np.array([])
xffnn = np.array([])
i = 2*dt
while i<t_end-dt:
    xffnn = np.append(xffnn, x0)
    tffnn = np.append(tffnn, i)
    phiffnn = np.append(phiffnn, phi0)
    input_data = np.array([[phi0, phim1, phim2, x0, xm1, xm2, F_func(i)]])
    ergplus1 = model.predict(input_data)
    phi0, phim1, phim2, x0, xm1, xm2 = ergplus1[0][0], phi0, phim1, ergplus1[0][1], x0, xm1
    i = i+dt

plt.subplot(1, 2, 1)
plt.plot(t_eval, phidata, label='echter Verlauf Winkel')
plt.plot(tffnn, phiffnn, label='vorhergesagter Verlauf Winkel')
plt.legend(loc='lower right')
plt.xlabel('Zeit in s')
plt.ylabel('Auslenkung in rad')

plt.subplot(1, 2, 2)
plt.plot(t_eval, xdata, label='echter Verlauf Wagenposition')
plt.plot(tffnn, xffnn, label='vorhergesagter Verlauf Wagen')
plt.legend(loc='lower right')
plt.xlabel('Zeit in s')
plt.ylabel('Position in m')


plt.show()

correlationphi, _ = pearsonr(phiffnn, phidata[2:])
print('Korrelation Winkel: ', correlationphi)

correlationx, _ = pearsonr(xffnn, xdata[2:])
print('Korrelation Position: ', correlationx)