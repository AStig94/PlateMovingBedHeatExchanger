import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Solid Heat Exchanger Simulation", layout="wide")
st.title("🔁 Solid Heat Exchanger Simulation")

# Sidebar inputs
st.sidebar.header("Input Parameters")

# Geometry
L = st.sidebar.number_input("Plate Length L [m]", value=1.0)
B = st.sidebar.slider("Distance Between Plates B [m]", 0.01, 0.2, 0.05)
N = st.sidebar.slider("Number of Plates N", 1, 100, 10)
H = st.sidebar.number_input("Plate Height H [m]", value=1.25)
dx = 0.001
dz = 0.01

# Solid properties
m_s = st.sidebar.number_input("Mass Flow (Solid) [kg/h]", value=250.0) / 3600
Cp_s = st.sidebar.number_input("Specific Heat (Solid) [J/kg·K]", value=1200.0)
T0 = st.sidebar.number_input("Solid Inlet Temperature T0 [°C]", value=350.0)
pb = st.sidebar.number_input("Bulk Density [kg/m³]", value=600.0)

# Fluid properties
m_f = st.sidebar.number_input("Mass Flow (Fluid) [L/h]", value=10.0) * 1000 / 3600
Cp_f = 4184
T_fout = st.sidebar.number_input("Fluid Outlet Temperature [°C]", value=22.0)

# Material parameters
eps_s = st.sidebar.slider("Void Fraction (ε)", 0.1, 0.9, 0.4)
k_s = 0.2
k_g = 0.0313
a = np.inf
d_p = 0.25e-3

# Derived values
u = m_s / (pb * L * B * N)
C = (m_s * Cp_s) / (m_f * Cp_f)

# Effective thermal conductivity
K_f = k_s / k_g
phi = (1/4) * (((K_f-1)/K_f)**2 / (np.log(K_f)-(K_f-1)/K_f)) - 1/(3*K_f)
k_s_eff = k_g * (1-eps_s) / (1/k_s + phi)

# Wall resistance
Y = d_p/(2*a)
eps_w = 1 - (1-eps_s)*(0.7293+0.5139*Y)/(1+Y)
k_s_nw_eff = k_g * (eps_w + (1-eps_w)/(2*phi+(2/3)*(k_g/k_s)))
R_c = d_p / (2 * k_s_nw_eff)

# Fluid heat transfer (water)
nu_w = 1.0533e-6
pb_w = 1000
t_pp = 2e-3
u_w = m_f / (pb_w * L * t_pp * N)
d_hd = 2 * (L * t_pp) / (L + t_pp)
Re_hd = u_w * d_hd / nu_w
mu_w = pb_w * nu_w
k_w = 0.598
Pr_w = Cp_f * mu_w / k_w

T_m = 21.06
T_w = 27.46
h_w = (0.0214*(Re_hd**0.8 - 100)*(Pr_w**0.4) *
       ((1+(d_hd/L))**(2/3)) * ((T_m/T_w)**0.48)*k_w) / d_hd

t_dw = 2e-3
k_dw = 16.3
R = R_c + (t_dw/k_dw) + 1/h_w
h = 1/R

# Numerical setup
lamb = (k_s_eff * dz) / (pb * Cp_s * u * dx**2)
Ix = int(B/dx)
Iz = int(H/dz)

a_arr = np.full(Ix-1, -lamb)
a_arr[-1] = -k_s_eff / (h * dx)

b_arr = np.full(Ix, 1 + 2 * lamb)
b_arr[0] = 1 + k_s_eff / (h * dx)
b_arr[-1] = 1 + k_s_eff / (h * dx)

c_arr = np.full(Ix-1, -lamb)
c_arr[0] = -k_s_eff / (h * dx)

d_arr = np.full(Ix, T0)
d_arr[0] = T_fout + C * (T0 - T0)
d_arr[-1] = T_fout + C * (T0 - T0)

T = np.full(Ix, T0)
T_store = np.full((Ix, Iz), T0)

# Thomas algorithm
for im in range(1, Iz):
    a = a_arr.copy()
    b = b_arr.copy()
    c = c_arr.copy()
    d = T.copy()
    d[0] = T_fout + C * (np.mean(T) - T0)
    d[-1] = T_fout + C * (np.mean(T) - T0)

    for it in range(1, Ix):
        m = a[it-1] / b[it-1]
        b[it] -= m * c[it-1]
        d[it] -= m * d[it-1]

    T[-1] = d[-1] / b[-1]
    for il in range(Ix-2, -1, -1):
        T[il] = (d[il] - c[il] * T[il+1]) / b[il]

    T_store[:, im] = T

# Plot temperature profile at various heights
xaxis = np.linspace(-B/2, B/2, Ix)
z_heights = [0, 0.25, 0.5, 0.75, 1.0]
z_indices = [int(Iz * z) for z in z_heights]

fig1, ax1 = plt.subplots()
for z_idx, frac in zip(z_indices, z_heights):
    ax1.plot(xaxis, T_store[:, z_idx], label=f'H = {H*frac:.2f} m')

ax1.set_title("Temperature Distribution Between Plates")
ax1.set_xlabel("Distance from Center [m]")
ax1.set_ylabel("Temperature [°C]")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

# Plot average solid/fluid temp
TF = T_fout + C * (np.mean(T_store, 0) - T0)
xaxis_z = np.linspace(0, H, Iz)

fig2, ax2 = plt.subplots()
ax2.plot(xaxis_z, np.mean(T_store, 0), label="Solid")
ax2.plot(xaxis_z, TF, label="Fluid")
ax2.set_title("Mean Temperature of Solid and Fluid")
ax2.set_xlabel("Height [m]")
ax2.set_ylabel("Temperature [°C]")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

# Output section
T_m_update = (TF[0] + TF[-1]) / 2
T_w_update = np.mean(T_store[0])
Q = m_s * Cp_s * (np.mean(T_store[:, 0]) - np.mean(T_store[:, -1]))

DT_max = np.mean(T_store[:, 0] - TF[0])
DT_min = np.mean(T_store[:, -1] - TF[-1])
DT_log = (DT_max - DT_min) / np.log(DT_max / DT_min)
A = 2 * N * L * H
U_tot = Q / (A * DT_log)
h_sw = 1 / ((1 / U_tot) - (1 / h_w + t_dw / k_dw))

st.subheader("📊 Outputs")
st.markdown(f"""
- **Updated T<sub>m</sub>**: `{T_m_update:.2f} °C`  
- **Updated T<sub>w</sub>**: `{T_w_update:.2f} °C`  
- **Heat Transfer (Q)**: `{Q/1000:.2f} kW`  
- **Overall U-value**: `{U_tot:.2f} W/(m²·K)`  
- **Solid-Wall h-value**: `{h_sw:.2f} W/(m²·K)`
""", unsafe_allow_html=True)
