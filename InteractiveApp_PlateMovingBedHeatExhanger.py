import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ------------------- How to -------------------
# If using VSC (else just use the streamlit app:    https://platemovingbedheatexchanger.streamlit.app/
# Make sure you are in the right folder. "cd .." goes one folder up, "cd+space+shift" autofills destination.
# in terminal write:    streamlit run InteractiveApp_PlateMovingBedHeatExchanger.py
# ctrl+c to abort terminal..

st.set_page_config(page_title="Plate Moving Bed Heat Exchanger", layout="wide")

st.title("Plate Moving Bed Heat Exchanger")
st.markdown("Adjust the parameters to simulate the heat exchanger behavior.")

# ------------------- Flowability Check -------------------
st.sidebar.header("Assumptions")
is_free_flowing = st.sidebar.checkbox("Material is free flowing", value=True)

if not is_free_flowing:
    st.error("❌ This model requires the material to be free flowing.\nPlease ensure proper handling or select another approach.")
    st.stop()

# Sidebar inputs with headers
st.sidebar.header("Geometry")
L = st.sidebar.number_input("Length of plates (m)", min_value=0.1, value=1.0)
B_mm = st.sidebar.slider("Distance between plates (mm)", min_value=2, max_value=100, value=50)
B = B_mm / 1000  # convert mm to meters
N = st.sidebar.number_input("Number of plates", min_value=1, step=1, value=10)
H = st.sidebar.number_input("Height of plates (m)", min_value=0.1, value=1.25)

st.sidebar.header("Solid Properties and Flow")
m_s_kgph = st.sidebar.number_input("Solid mass flow rate  (kg/hr)", min_value=0.1, value=240.0, step=10.0)
pb = st.sidebar.number_input("Solid density (kg/m³)", min_value=0.1, value=600.0, step=10.0)
T0 = st.sidebar.number_input("Solid inlet temperature (°C)", value=350.0, step=10.0)
m_s = m_s_kgph / 3600  # convert to kg/s
k_s = st.sidebar.number_input("Solid thermal conductivity (W/m·K)", min_value=0.01, value=0.12)
Cp_s = st.sidebar.number_input("Solid specific heat capacity (J/kg·K)", min_value=100.0, value=1200.0, step=10.0)
eps_s = st.sidebar.slider("Voidage (ε)", min_value=0.01, max_value=0.99, value=0.75, step=0.01)

st.sidebar.header("Fluid Properties and Flow")
u_fluid = 1.0  # fixed fluid velocity in m/s
Cp_f = 4184.0  # constant fluid specific heat capacity (water)
T_fin = st.sidebar.number_input("Fluid inlet temperature (°C)", value=22.0, step=1.0)
st.sidebar.markdown(f"**Fluid velocity between plates:** {u_fluid} m/s")

# ------------------- Acknowledgement -------------------
with st.sidebar.expander("📚 Acknowledgement"):
    st.markdown("""
This application is based on methodologies and insights from the following sources:

1. "Design of Bulk Solids Moving Bed Heat Exchangers" – *Chemical Engineering Online*.  
Available at: [https://www.chemengonline.com/design-bulk-solids-moving-bed-heat-exchangers/?printmode=1](https://www.chemengonline.com/design-bulk-solids-moving-bed-heat-exchangers/?printmode=1)

2. D. Müller, M. van der Meer, and E. Huenges (2019).  
**Heat transfer model of a particle energy storage based moving packed bed heat exchanger**.  
*ResearchGate*.  
Available at: [https://www.researchgate.net/publication/337455917](https://www.researchgate.net/publication/337455917_Heat_transfer_model_of_a_particle_energy_storage_based_moving_packed_bed_heat_exchanger)

3. US Department of Energy (2018).  
**High-temperature heat exchanger for thermal energy storage systems**.  
*OSTI Technical Report No. 1427340*.  
Available at: [https://www.osti.gov/servlets/purl/1427340](https://www.osti.gov/servlets/purl/1427340)

4. M. Meier et al. (2018).  
**Modeling of a moving packed bed heat exchanger for solid particles**.  
*AIP Conference Proceedings*, 2033, 030021.  
DOI: [10.1063/1.5067039](https://aip.scitation.org/doi/pdf/10.1063/1.5067039)

5. Isaza, P. A. (2016).  
**Mathematical modeling and analysis of thermal energy storage systems based on moving packed beds**.  
PhD Thesis, University of Toronto.  
Available at: [https://tspace.library.utoronto.ca/handle/1807/76458](https://tspace.library.utoronto.ca/bitstream/1807/76458/3/Isaza_Pedro_A_201611_PhD_thesis.pdf)
""")

# Calculate fluid mass flow rate based on velocity and geometry
pb_w = 1000  # fluid density (kg/m3) - assuming water
t_pp = 2e-3  # plate thickness (m)
m_f = pb_w * u_fluid * L * t_pp * N  # fluid mass flow rate (kg/s)

def run_simulation(L, B, N, H, m_s, pb, Cp_s, k_s, eps_s, m_f, Cp_f, T_fin, T0):
    dx = 0.001
    dz = 0.01
    C = (m_s * Cp_s) / (m_f * Cp_f)
    T_fin_K = T_fin + 273.15  # Fluid inlet temperature in K
    T0_K = T0 + 273.15        # Solid inlet temperature in K

    # Velocity of solid particles (m/s)
    pb = 600  # solid density (kg/m3)
    u = m_s / (pb * L * B * N)

    # Material and heat transfer properties (constants as before)
    k_g = 0.03742  # W/m·K for N2 at ~200°C. The gas in the void of the particles. Could be adjusted for actual mean temperature.
    beta = 1    # β accounts for a characteristic length of conduction through the gas (e.g., mean free path or interstitial path length). Set to 1 in ref 2.    
    gamma = 1 # γ accounts for a characteristic length of conduction through the solid particle matrix. Set to 1 in ref 2.
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
    t_pp = 2e-3 # Internal distance between welded plates for water to flow in, assumed 2mm.
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

    t_dw = 2e-3 # 2mm wall thickness of plates
    k_dw = 16.3 # Plate material AISI 316L
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

    d_arr = np.full((Ix), T0_K, dtype=float)
    d_arr[0] = d_arr[-1] = T_fin_K

    T = np.full((Ix), T0_K, dtype=float)
    T_store = np.full((Ix, Iz), T0_K, dtype=float)

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
        d_arr[0] = d_arr[-1] = T_fin_K + C * (np.mean(T) - T0_K)

    # Prepare plots - distances in mm
    xaxis_mm = np.linspace(-B_mm/2, B_mm/2, Ix)
    z_heights = [0, 0.25, 0.5, 0.75, 1.0]
    z_indices = [min(int(Iz * z), Iz - 1) for z in z_heights]

    # First plot: temperature across plate width (°C)
    fig1, ax1 = plt.subplots(figsize=(6, 5), dpi=80)
    for frac, z_idx in zip(z_heights, z_indices):
        ax1.plot(xaxis_mm, T_store[:, z_idx] - 273.15, label=f'H = {H*frac:.2f} m')
    ax1.set_title("Temperature across plate width")
    ax1.set_xlabel("Distance from center (mm)")
    ax1.set_ylabel("Temperature (°C)")
    ax1.grid()
    ax1.legend()

    # Second plot: mean temperature along height (°C)
    fig2, ax2 = plt.subplots(figsize=(6, 5), dpi=80)
    xaxis_z = np.linspace(0, H, Iz)
    TF = T_fin_K + C * (np.mean(T_store, axis=0) - T0_K)
    ax2.plot(xaxis_z, np.mean(T_store, axis=0) - 273.15, label="Solid")
    ax2.plot(xaxis_z, TF - 273.15, label="Fluid")
    ax2.set_title("Mean temperature along height")
    ax2.set_xlabel("Height (m)")
    ax2.set_ylabel("Temperature (°C)")
    ax2.grid()
    ax2.legend()

    # Output metrics calculations
    # Convert fluid inlet temperature to K for consistency
    T_fluid_out = T_fin_K + (m_s * Cp_s * (np.mean(T_store[:, 0]) - np.mean(T_store[:, -1]))) / (m_f * Cp_f)

    max_solid_temp_outlet = np.max(T_store[:, -1]) - 273.15
    mean_solid_temp_outlet = np.mean(T_store[:, -1]) - 273.15

    Q = m_s * Cp_s * (np.mean(T_store[:, 0]) - np.mean(T_store[:, -1]))

    DT_max = np.mean(T_store[:, 0] - TF[0])
    DT_min = np.mean(T_store[:, -1] - TF[-1])
    DT_log = (DT_max - DT_min) / np.log(DT_max / DT_min)
    A = 2 * N * L * H
    U_tot = Q / (A * DT_log)
    h_sw = 1 / ((1 / U_tot) - (1 / h_w + t_dw / k_dw))

    return fig1, fig2, T_fluid_out, max_solid_temp_outlet, mean_solid_temp_outlet, Q, U_tot, h_sw

fig1, fig2, T_fluid_out, max_solid_temp_outlet, mean_solid_temp_outlet, Q, U_tot, h_sw = run_simulation(
    L, B, N, H, m_s, pb, Cp_s, k_s, eps_s, m_f, Cp_f, T_fin, T0
)

col1, col2, col3 = st.columns([3, 3, 2])  # Adjust widths as needed
with col1:
    st.pyplot(fig1, use_container_width=False)
with col2:
    st.pyplot(fig2, use_container_width=False)
with col3:
    st.markdown("### Output Metrics")
    st.write(f"Max solid temperature at outlet: {round(max_solid_temp_outlet, 2)} °C")    # Already °C after subtraction
    st.write(f"Mean solid temperature at outlet: {round(mean_solid_temp_outlet, 2)} °C")  # Already °C after subtraction
    st.write(f"Mean fluid temperature at outlet: {round(T_fluid_out - 273.15, 2)} °C")  # Convert K to °C here
    st.write(f"Heat transfer rate Q: {round(Q / 1000, 2)} kW")
    st.write(f"Overall heat transfer coefficient U: {round(U_tot, 2)} W/(m²·K)")
