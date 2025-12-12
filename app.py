# app.py — GravCore Streamlit (Nagy upgraded)
import streamlit as st
import pandas as pd
import numpy as np
from math import sqrt
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import io, time

# -----------------------
# UTM conversion (Redfearn)
# -----------------------
def latlon_to_utm_redfearn(lat, lon):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    a = 6378137.0
    f = 1 / 298.257223563
    k0 = 0.9996
    b = a * (1 - f)
    e = sqrt(1 - (b / a) ** 2)
    e2 = e * e
    zone = np.floor((lon + 180) / 6) + 1
    lon0 = (zone - 1) * 6 - 180 + 3
    lon0_rad = np.radians(lon0)
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)
    T = np.tan(lat_rad) ** 2
    C = (e2 / (1 - e2)) * np.cos(lat_rad) ** 2
    A = np.cos(lat_rad) * (lon_rad - lon0_rad)
    M = (a * ((1 - e2/4 - 3*e2*e2/64 - 5*e2**3/256) * lat_rad
         - (3*e2/8 + 3*e2*e2/32 + 45*e2**3/1024) * np.sin(2*lat_rad)
         + (15*e2*e2/256 + 45*e2**3/1024) * np.sin(4*lat_rad)
         - (35*e2**3/3072) * np.sin(6*lat_rad)))
    easting = k0 * N * (A + (1 - T + C) * A**3 / 6 + (5 - 18*T + T*T + 72*C - 58*e2) * A**5 / 120) + 500000
    northing = k0 * (M + N * np.tan(lat_rad) * (A**2/2 + (5 - T + 9*C + 4*C*C) * A**4/24
                   + (61 - 58*T + T*T + 600*C - 330*e2) * A**6 / 720))
    hemi = np.where(lat >= 0, "north", "south")
    northing = np.where(hemi == "south", northing + 10000000, northing)
    return easting, northing, zone, hemi

# -----------------------
# Drift solver
# -----------------------
def compute_drift(df, G_base):
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="raise")
    df["G_read (mGal)"] = pd.to_numeric(df["G_read (mGal)"], errors="coerce")
    names = df["Nama"].astype(str).tolist()
    unique_st = list(dict.fromkeys(names))
    base = unique_st[0]
    unknown = [s for s in unique_st if s != base]
    N = len(unknown) + 1
    A = []
    b = []
    frac = (df["Time"].dt.hour*3600 + df["Time"].dt.minute*60 + df["Time"].dt.second) / 86400.0
    for i in range(len(df)-1):
        row = np.zeros(N)
        dG = df["G_read (mGal)"].iloc[i+1] - df["G_read (mGal)"].iloc[i]
        dt = frac.iloc[i+1] - frac.iloc[i]
        const = dG
        si = names[i]; sj = names[i+1]
        if sj != base:
            row[unknown.index(sj)] = 1
        else:
            const -= G_base
        if si != base:
            row[unknown.index(si)] = -1
        else:
            const += G_base
        row[-1] = dt
        A.append(row); b.append(const)
    A = np.array(A, dtype=float); b = np.array(b, dtype=float)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    D = float(x[-1])
    Gmap = {base: float(G_base)}
    for idx, st in enumerate(unknown):
        Gmap[st] = float(x[idx])
    return Gmap, D

# -----------------------
# Basic corrections
# -----------------------
def latitude_correction(lat):
    phi = np.radians(lat)
    s = np.sin(phi); s2 = np.sin(2*phi)
    return 978032.67715 * (1 + 0.0053024 * s*s - 0.0000059 * s2*s2)

def free_air(elev):
    return 0.3086 * elev

# -----------------------
# Hammer (unchanged)
# -----------------------
HAMMER_R = np.array([25,100,200,500,2000,5000])
HAMMER_F = np.array([0.035,0.03,0.025,0.02,0.015,0.01])
def hammer_tc(e0, n0, z0, dem_df):
    dx = dem_df["Easting"].to_numpy() - float(e0)
    dy = dem_df["Northing"].to_numpy() - float(n0)
    dist = np.sqrt(dx*dx + dy*dy)
    z = dem_df["Elev"].to_numpy()
    tc = 0.0; inner = 0.0
    for i, outer in enumerate(HAMMER_R):
        mask = (dist >= inner) & (dist < outer)
        if mask.sum() > 0:
            dh = z[mask].mean() - float(z0)
            tc += HAMMER_F[i] * dh
        inner = outer
    return float(tc)

# -----------------------
# Nagy prism — improved
# -----------------------
G_SI = 6.67430e-11
M2MGAL = 1e5

def prism_g_term(xi, yj, zk):
    # helper for single corner terms with safe logs
    R = np.sqrt(xi*xi + yj*yj + zk*zk) + 1e-20
    return xi*np.log(abs(yj + R)) + yj*np.log(abs(xi + R)) - zk*np.arctan2(xi*yj, zk*R + 1e-20)

def prism_vertical_attraction(x1, x2, y1, y2, z1, z2, px, py, pz):
    # Full analytic vertical attraction term for rectangular prism
    # returns sum_{corners} (-1)^{i+j+k} * f(xi,yj,zk)
    X = [x1 - px, x2 - px]
    Y = [y1 - py, y2 - py]
    Z = [z1 - pz, z2 - pz]
    ssum = 0.0
    for i in range(2):
        xi = X[i]
        for j in range(2):
            yj = Y[j]
            for k in range(2):
                zk = Z[k]
                sign = (-1) ** (i + j + k)
                ssum += sign * prism_g_term(xi, yj, zk)
    return ssum

def compute_nagy_tc(e0, n0, z0, dem_df, density, max_radius=10000.0, cell_size=None, z_ref=0.0):
    """
    Compute Nagy prism-based terrain correction for station (e0,n0,z0).
    dem_df must contain columns 'Easting','Northing','Elev' (units: meters).
    density in kg/m^3; max_radius in meters; cell_size optional (m).
    z_ref: bottom reference for prisms (common is 0 = sea level). Prism extends z_ref..ztop.
    """
    # select DEM points within max_radius
    dx = dem_df["Easting"].to_numpy() - float(e0)
    dy = dem_df["Northing"].to_numpy() - float(n0)
    r = np.sqrt(dx*dx + dy*dy)
    dem_sel = dem_df.loc[r <= max_radius].copy()
    if dem_sel.empty:
        return 0.0

    # infer cell size if not given
    if cell_size is None:
        xs = np.sort(np.unique(dem_sel["Easting"].to_numpy()))
        ys = np.sort(np.unique(dem_sel["Northing"].to_numpy()))
        if len(xs) > 1:
            dx_med = np.median(np.diff(xs))
        else:
            dx_med = 25.0
        if len(ys) > 1:
            dy_med = np.median(np.diff(ys))
        else:
            dy_med = dx_med
        cell = max(dx_med, dy_med)
    else:
        cell = float(cell_size)

    # define grid extents for binning
    minx = dem_sel["Easting"].min() - 0.5*cell
    miny = dem_sel["Northing"].min() - 0.5*cell

    dem_sel["ix"] = ((dem_sel["Easting"] - minx) / cell).astype(int)
    dem_sel["iy"] = ((dem_sel["Northing"] - miny) / cell).astype(int)
    # mean elevation per cell
    grouped = dem_sel.groupby(["ix","iy"])["Elev"].mean().reset_index()
    if grouped.empty:
        return 0.0

    # accumulate prism analytic term
    gz_sum = 0.0
    # bottom reference z1 is z_ref (common constant), top is ztop (cell mean)
    for _, row in grouped.iterrows():
        ix = int(row["ix"]); iy = int(row["iy"])
        ztop = float(row["Elev"])
        x1 = minx + ix*cell
        x2 = x1 + cell
        y1 = miny + iy*cell
        y2 = y1 + cell
        # vertical bounds: z_ref .. ztop
        gz_sum += prism_vertical_attraction(x1, x2, y1, y2, float(z_ref), ztop, float(e0), float(n0), float(z0))

    # multiply by G and density to get SI m/s^2, convert to mGal
    gz_si = G_SI * float(density) * gz_sum
    gz_mgal = gz_si * M2MGAL
    return float(gz_mgal)

# -----------------------
# DEM loader robust
# -----------------------
def load_dem(filelike):
    try:
        df = pd.read_csv(filelike)
        if df.shape[1] == 1:
            raise
    except Exception:
        filelike.seek(0)
        df = pd.read_csv(filelike, sep=r"\s+", engine="python", header=None)
    if df.shape[1] < 3:
        raise ValueError("DEM must have ≥ 3 columns (Lon,Lat,Elev or E,N,Elev)")
    df = df.iloc[:, :3].copy()
    df.columns = ["Lon","Lat","Elev"]
    df["Lon"] = pd.to_numeric(df["Lon"], errors="coerce")
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["Elev"] = pd.to_numeric(df["Elev"], errors="coerce")
    df.dropna(inplace=True)
    # convert lon/lat -> UTM
    E,N,_,_ = latlon_to_utm_redfearn(df["Lat"].to_numpy(), df["Lon"].to_numpy())
    return pd.DataFrame({"Easting":E,"Northing":N,"Elev":df["Elev"].to_numpy()})

# -----------------------
# UI
# -----------------------
LOGO_ESDM = "https://raw.githubusercontent.com/dzakyw/AutoGrav/main/logo esdm.png"

# Header kiri
col1, col2 = st.columns([1, 5])
with col1:
    st.image(LOGO_ESDM, width=600)
with col2:
    st.markdown("""
        <h2 style='margin-bottom:0;'>Auto Gravity Processing</h2>
        <p style='margin-top:0;'>Modul Pengolahan Gravity</p>
    """, unsafe_allow_html=True)

st.markdown("---")  # garis pemisah
# ===========================
# SIMPLE LOGIN MODULE
# ===========================

# Daftar username + password
USER_CREDENTIALS = {
    "admin": "12345",
    "user": "password"
}

def check_login():
    # Bila belum login → tampilkan form login
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.title("Login Required")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password")

        st.stop()

# Panggil login check
check_login()

# =============================
# JIKA LOGIN SUKSES, LANJUT APP
# =============================
st.success(f"Welcome, {st.session_state.username}!")

st.title("AutoGrav - Lebih Cepat Lebih Baik")
st.sidebar.header("Inputs")
grav = st.sidebar.file_uploader("Input Gravity Multi-Sheets (.xlsx)", type=["xlsx"])
demf = st.sidebar.file_uploader("DEM (format Lon,Lat,Elev) optional", type=["csv","txt","xyz","xlsx"])
kmf = st.sidebar.file_uploader("Koreksi Medan manual (optional)", type=["csv","xlsx"])
G_base = st.sidebar.number_input("G Absolute di Base", value=0.0)
method = st.sidebar.selectbox("Metode Pengukuran Terrain", ["NAGY (prism)","HAMMER"])
density = st.sidebar.number_input("Densitas Koreksi Medan (kg/m³)", value=2670.0, step=10.0, format="%.1f")
max_radius = st.sidebar.number_input("Jarak Maksimum (m) untuk Nagy", value=10000, step=1000)
z_ref = st.sidebar.number_input("z_ref (bottom prism reference, m)", value=0.0)
run = st.sidebar.button("Run")

st.sidebar.write("Notes: densitas biasanya 2670 kg/m³; adjust naik jarak radius untuk mendapatkan pengukuran medan yang jauh (biasanya 5-10 km)")
st.sidebar.subheader("Contoh File Input") 
st.sidebar.write("[Contoh Data Input Gravity](https://github.com/dzakyw/AutoGrav/raw/9bb43e1559c823350f2371360309d84eaab5ea38/sample_gravity.xlsx)") 
st.sidebar.write("[Contoh DEM](https://github.com/dzakyw/AutoGrav/raw/9bb43e1559c823350f2371360309d84eaab5ea38/sample_dem.csv)") 
st.sidebar.write("[Contoh Koreksi Medan](https://github.com/dzakyw/AutoGrav/raw/9bb43e1559c823350f2371360309d84eaab5ea38/sample_koreksi_medan.csv)")
# validation
if run:
    if method.startswith("NAGY") and (density is None or density <= 0.0):
        st.error("Untuk metode NAGY, isi density > 0 (kg/m³). Mis. 2670.")
        st.stop()

    if grav is None:
        st.error("Upload file gravity .xlsx (multi-sheet).")
        st.stop()

    dem = None
    if demf:
        try:
            dem = load_dem(demf)
            st.success(f"DEM loaded: {len(dem)} points")
        except Exception as e:
            st.error(f"Gagal load DEM: {e}")
            st.stop()

    # load manual koreksi if provided
    km_map = None
    if kmf:
        try:
            km = pd.read_csv(kmf)
        except:
            km = pd.read_excel(kmf)
        if {"Nama","Koreksi_Medan"}.issubset(km.columns):
            km["Koreksi_Medan"] = pd.to_numeric(km["Koreksi_Medan"], errors="coerce")
            km_map = km.set_index("Nama")["Koreksi_Medan"].to_dict()
        else:
            st.warning("File koreksi medan manual harus kolom: Nama, Koreksi_Medan. Ignored.")

    if (km_map is None) and (dem is None):
        st.error("Anda harus upload DEM atau file koreksi medan manual.")
        st.stop()

    # read excel
    try:
        xls = pd.ExcelFile(grav)
    except Exception as e:
        st.error(f"Gagal baca Excel gravitasi: {e}")
        st.stop()

    all_dfs = []
    t0 = time.time()
    total_stations = 0
    for sh in xls.sheet_names:
        df = pd.read_excel(grav, sheet_name=sh)
        required = {"Nama","Time","G_read (mGal)","Lat","Lon","Elev"}
        if not required.issubset(set(df.columns)):
            st.warning(f"Sheet {sh} dilewati (kolom tidak lengkap).")
            continue

        # UTM conversion for stations
        E,N,_,_ = latlon_to_utm_redfearn(df["Lat"].to_numpy(), df["Lon"].to_numpy())
        df["Easting"] = E; df["Northing"] = N

        # drift
        Gmap, D = compute_drift(df, G_base)
        df["G_read (mGal)"] = df["Nama"].map(Gmap)

        # basic
        df["Koreksi Lintang"] = latitude_correction(df["Lat"])
        df["Free Air Correction"] = free_air(df["Elev"])
        df["FAA"] = df["G_read (mGal)"] - df["Koreksi Lintang"] + df["Free Air Correction"]

        # terrain correction selection
        if km_map is not None:
            df["Koreksi Medan"] = df["Nama"].map(km_map)
            missing = df["Koreksi Medan"].isna().sum()
            if missing > 0:
                st.warning(f"{missing} station(s) in sheet {sh} not in manual koreksi_medan.")
        else:
            # compute Nagy or Hammer from DEM
            tc_list = []
            nstations = len(df)
            total_stations += nstations
            # show a progress bar
            progress_text = st.empty()
            pbar = st.progress(0)
            for i in range(nstations):
                e0 = float(df.iloc[i]["Easting"]); n0 = float(df.iloc[i]["Northing"]); z0 = float(df.iloc[i]["Elev"])
                if method.startswith("NAGY"):
                    tc_val = compute_nagy_tc(e0, n0, z0, dem_df=dem if dem is not None else None,
                                              density=density, max_radius=float(max_radius), cell_size=None, z_ref=float(z_ref))
                else:
                    tc_val = hammer_tc(e0, n0, z0, dem)
                tc_list.append(tc_val)
                pbar.progress(int((i+1)/nstations*100))
                progress_text.text(f"Processing sheet {sh}: station {i+1}/{nstations}")
            pbar.empty(); progress_text.empty()
            df["Koreksi Medan"] = tc_list

        df["X-Parasnis"] = 0.04192 * df["Elev"] - df["Koreksi Medan"]
        df["Y-Parasnis"] = df["Free Air Correction"]
        df["Hari"] = sh
        all_dfs.append(df)

    if len(all_dfs) == 0:
        st.error("No valid sheets processed.")
        st.stop()

    df_all = pd.concat(all_dfs, ignore_index=True)
    elapsed = time.time() - t0
    st.write(f"Processed {len(df_all)} rows in {elapsed:.1f} s")

    # slope and Bouguer
    mask = df_all[["X-Parasnis","Y-Parasnis"]].notnull().all(axis=1)
    if mask.sum() >= 2:
        slope, intercept = np.polyfit(df_all.loc[mask,"X-Parasnis"], df_all.loc[mask,"Y-Parasnis"], 1)
    else:
        slope = np.nan
    df_all["Bouger Correction"] = 0.04192 * slope * df_all["Elev"]
    df_all["Simple Bouger Anomaly"] = df_all["FAA"] - df_all["Bouger Correction"]
    df_all["Complete Bouger Correction"] = df_all["Simple Bouger Anomaly"] + df_all["Koreksi Medan"]

    st.success("Done")
    st.dataframe(df_all.head(20))

    # contour plots
    x = df_all["Easting"]; y = df_all["Northing"]
    gx = np.linspace(x.min(), x.max(), 200); gy = np.linspace(y.min(), y.max(), 200)
    GX, GY = np.meshgrid(gx, gy)
    def plot_cont(z, title):
        Z = griddata((x,y), z, (GX,GY), method="cubic")
        fig, ax = plt.subplots(figsize=(8,6))
        cf = ax.contourf(GX, GY, Z, 40, cmap="jet")
        ax.scatter(x, y, c=z, cmap="jet", s=12, edgecolor="k")
        ax.set_title(title)
        fig.colorbar(cf, ax=ax)
        st.pyplot(fig)
    plot_cont(df_all["Complete Bouger Correction"], "CBA")
    plot_cont(df_all["Simple Bouger Anomaly"], "SBA")
    plot_cont(df_all["Elev"], "Elevation")

    # download
    st.download_button("Download CSV", df_all.to_csv(index=False).encode("utf-8"), "gravcore_output.csv")














