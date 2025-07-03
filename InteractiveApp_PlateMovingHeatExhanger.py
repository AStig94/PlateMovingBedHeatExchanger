import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Plate Moving Bed Heat Exchanger", layout="wide")

st.title("Plate Moving Bed Heat Exchanger Simulator")
st.markdown("Adjust the parameters below to simulate the heat exchanger behavior.")

# Sidebar input parameters
st.sidebar.header("Input Parameters")

# Geometry
L = st.sidebar.number_input("Length of plates (m)", min_value=0.1, value=1.0)
B = st.sidebar.number_input("Distance between plates (m)", min_value=0.01, value=0.05)
N = st.sidebar.number_input("Number of plates", min_value=1, step=1, value=10)
H = st.sidebar.number_input("Height of plates (m)", min_value=0.1, value=1.25)

# Material properties
m_s = st.sidebar.number_input("Solid mass flow rate (kg/s)", min_value=0.001, value=250/(60*60))
Cp_s = st.sidebar.number_input("Solid specific heat capacity (J/kg·K)", min_value=100.0, value=1200.0)
m_f = st.sidebar.number_input("Fluid mass flow rate (kg/s)", min_value=0.001, value=(10/(60*60))*1000)
Cp_f = st.sidebar.number_input("Fluid specific heat capacity (J/kg·K)", min_value=500.0, value=4184.0)

T_fout = st.sidebar.number_input("Fluid outlet temperature (°C)", value=22.0)
T0 = st.sidebar.number_input("Solid inlet temperature (°C)", value=350.0)

pb = 600
dx = 0.001
dz = 0.01
C = (m_s * Cp_s) / (m_f * Cp_f)
T_fout += 273.15  # Convert to Kelvin
T0 += 273.15      # Convert to Kelvin

u = m_s / (pb * L * B * N)

# Heat transfer calculations
k_s = 0.2
k_g = 0.0313
beta = 1
gamma = 1
eps_s = 0.4
K_f = k_s / k_g
phi = (1/4) * ((((K_f - 1)/K_f)**2) / (np.log(K_f) - (K_f - 1)/K_f)) - 1 / (3*K_f)
k_s_eff = k_g * beta * (1 - eps_s) / (gamma * k_g / k_s + phi)

a = np.inf
d_p = 0.25e-3
Y = d_p / (2 * a)
eps_w = 1 - (1 - eps_s) * (0.7293 + 0.5139 * Y) / (1 + Y)
k_s_nw_eff = k_g * (eps_w + (1 - eps_w) / (2 * phi + (2/3) * (k_g / k_s)))
R_c = d_p / (2 * k_s_nw_eff)

nu_w = 1.0533e-6
pb_w = 1000
t_pp = 2e-3
u_w = m_f / (pb_w * L * t_pp * N)
d_hd = 2 * (L * t_pp) / (L + t_pp)
Re_hd = u_w * d_hd / nu_w
mu_w = pb_w * nu_w
k_w = 0.598
Pr_w = Cp_f * mu_w / k_w

T_m = 21.06 + 273.15
T_w = 27.46 + 273.15

h_w = (0.0214 * (Re_hd**0.8 - 100) * Pr_w**0.4 *
       (1 + (d_hd / L))**(2/3) * (T_m / T_w)**0.48 * k_w) / d_hd

t_dw = 2e-3
k_dw = 16.3
R = R_c + (t_dw / k_dw) + 1 / h_w
h = 1 / R

# Matrix initialization
lamb = (k_s_eff * dz) / (pb * Cp_s * u * dx**2)
Ix = int(B / dx)
Iz = int(H / dz)

a_arr = np.full((Ix-1), -lamb, dtype=float)
a_arr[-1] = -k_s_eff / (h * dx)

b_arr = np.full((Ix), 1 + 2 * lamb, dtype=float)
b_arr[0] = b_arr[-1] = 1 + k_s_eff / (h * dx)

c_arr = np.full((Ix-1), -lamb, dtype=float)
c_arr[0] = -k_s_eff / (h * dx)

d_arr = np.full((Ix), T0, dtype=float)
d_arr[0] = d_arr[-1] = T_fout

T = np.full((Ix), T0, dtype=float)
T_store = np.full((Ix, Iz), T0, dtype=float)

for im in range(1, Iz):
    a = a_arr.copy()
    b = b_arr.copy()
    c = c_arr.copy()
    d = d_arr.copy()

    for it in range(1, Ix):
        m = a[it-1] / b[it-1]
        b[it] -= m * c[it-1]
        d[it] -= m * d[it-1]

    T[-1] = d[-1] / b[-1]
    for il in range(Ix-2, -1, -1):
        T[il] = (d[il] - c[il] * T[il+1]) / b[il]

    T_store[:, im] = T
    d_arr = T[:]
    d_arr[0] = d_arr[-1] = T_fout + C * (np.mean(T) - T0)

# Plotting
xaxis = np.linspace(-B/2, B/2, Ix)
z_heights = [0, 0.25, 0.5, 0.75, 1.0]
z_indices = [min(int(Iz * z), Iz - 1) for z in z_heights]

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 6), dpi=80)
    for frac, z_idx in zip(z_heights, z_indices):
        ax1.plot(xaxis, T_store[:, z_idx], label=f'H = {H*frac:.2f} m')
    ax1.set_title("Temperature across plate width")
    ax1.set_xlabel("Distance from center (m)")
    ax1.set_ylabel("Temperature (K)")
    ax1.grid()
    ax1.legend()
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=80)
    xaxis_z = np.linspace(0, H, Iz)
    TF = T_fout + C * (np.mean(T_store, axis=0) - T0)
    ax2.plot(xaxis_z, np.mean(T_store, axis=0), label="Solid")
    ax2.plot(xaxis_z, TF, label="Fluid")
    ax2.set_title("Mean temperature along height")
    ax2.set_xlabel("Height (m)")
    ax2.set_ylabel("Temperature (K)")
    ax2.grid()
    ax2.legend()
    st.pyplot(fig2)

# Calculations
T_m_update = (TF[0] + TF[-1]) / 2
T_w_update = np.mean(T_store[0])
Q = m_s * Cp_s * (np.mean(T_store[:, 0]) - np.mean(T_store[:, -1]))

DT_max = np.mean(T_store[:, 0] - TF[0])
DT_min = np.mean(T_store[:, -1] - TF[-1])
DT_log = (DT_max - DT_min) / np.log(DT_max / DT_min)
A = 2 * N * L * H
U_tot = Q / (A * DT_log)
h_sw = 1 / ((1 / U_tot) - (1 / h_w + t_dw / k_dw))

st.markdown("### Output Metrics")
st.write(f"Mean mixed fluid temperature `T_m` = {round(T_m_update - 273.15, 2)} °C")
st.write(f"Mean solid wall temperature `T_w` = {round(T_w_update - 273.15, 2)} °C")
st.write(f"Heat transfer rate `Q` = {round(Q / 1000, 2)} kW")
st.write(f"Overall heat transfer coefficient `U` = {round(U_tot, 2)} W/(m²·K)")
st.write(f"Solid-wall heat transfer coefficient `h_sw` = {round(h_sw, 2)} W/(m²·K)")
