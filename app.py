
import streamlit as st
import pandas as pd
import numpy as np
from math import sqrt
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import io, time
from PIL import Image
import os
import streamlit as st
import hashlib

# ---------------------------------------------
# 1. HASH FUNCTION
# ---------------------------------------------
def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

# ---------------------------------------------
# 2. USER DATABASE (EDIT SESUAI KEBUTUHAN)
# ---------------------------------------------
USER_DB = {
    "admin": hash_password("admin"),     # ubah sesuai kebutuhan
    "user":  hash_password("12345"),      # ubah sesuai kebutuhan
}

# ROLE OPSIONAL
USER_ROLES = {
    "admin": "admin",
    "user": "viewer",
}

# ---------------------------------------------
# 3. AUTHENTICATION CHECK
# ---------------------------------------------
def authenticate(username, password):
    if username in USER_DB:
        return USER_DB[username] == hash_password(password)
    return False

# ---------------------------------------------
# 4. LOGIN PAGE (BLOCKING)
# ---------------------------------------------
def login_page():
    st.title("Welcome to Auto Grav")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    login_btn = st.button("Login")

    if login_btn:
        if authenticate(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = USER_ROLES.get(username, "viewer")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()   # hanya menampilkan login page

# ---------------------------------------------
# 5. LOGOUT BUTTON
# ---------------------------------------------
def logout_button():
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

# ---------------------------------------------
# 6. LOGIN WRAPPER
# ---------------------------------------------
def require_login():
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        login_page()

    # Jika sudah login, tampilkan status & tombol logout
    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    logout_button()
require_login()

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

def load_geotiff_without_tfw(file):
    """
    Load GeoTIFF using only TIFF metadata (GeoKeys).
    Works if TIFF contains:
      - ModelPixelScaleTag (33550)
      - ModelTiepointTag   (33922)
    """

    img = Image.open(file)
    arr = np.array(img, dtype=float)

    meta = img.tag_v2

    if 33550 not in meta or 33922 not in meta:
        raise ValueError("GeoTIFF metadata not found. TIFF requires TFW or manual bounding box.")

    # Metadata
    scaleX, scaleY, _ = meta[33550]        # pixel size
    tiepoint = meta[33922]                 # tiepoint structure

    # According to GeoTIFF specs:
    X0 = tiepoint[3]                       # model_space_X of UL corner
    Y0 = tiepoint[4]                       # model_space_Y of UL corner

    rows, cols = arr.shape

    # Build coordinate grid
    X = X0 + np.arange(cols) * scaleX
    Y = Y0 - np.arange(rows) * abs(scaleY)

    XX, YY = np.meshgrid(X, Y)

    df = pd.DataFrame({
        "Easting": XX.ravel(),
        "Northing": YY.ravel(),
        "Elev": arr.ravel()
    })

    return df
# -------------------------------------------------------------
# UNIVERSAL DEM LOADER (CSV / TXT / XYZ / TIFF / TIFF+TFW)
# -------------------------------------------------------------
def load_dem(file):
    name = file.name.lower()

    # CSV / TXT / XYZ
    if name.endswith((".csv", ".txt", ".xyz")):
        try:
            df = pd.read_csv(file)
        except:
            file.seek(0)
            df = pd.read_csv(file, sep=r"\s+", engine="python")

        df.columns = ["Lon","Lat","Elev"][:df.shape[1]]
        df = df.iloc[:, :3]

        df["Lon"] = pd.to_numeric(df["Lon"], errors="coerce")
        df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
        df["Elev"] = pd.to_numeric(df["Elev"], errors="coerce")
        df.dropna(inplace=True)

        E, N, _, _ = latlon_to_utm_manual(df["Lat"], df["Lon"])
        return pd.DataFrame({"Easting": E, "Northing": N, "Elev": df["Elev"]})

    # TIFF → GeoTIFF metadata
    if name.endswith((".tif", ".tiff")):
        return load_geotiff_without_tfw(file)

    raise ValueError("DEM format unsupported. Use: CSV, XYZ, TXT, TIFF")

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
HAMMER_R = np.array([2, 6, 18, 54, 162, 486, 1458, 4374])
HAMMER_F = np.array([0.00027, 0.00019, 0.00013, 0.00009, 0.00006, 0.00004, 0.000025, 0.000015])

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

# ============================================================
# OSS TERRAIN CORRECTION — DEM MASS INTEGRATION (PAPER STYLE)
# ============================================================

G = 6.67430e-11       # m³ kg⁻¹ s⁻²
SI2MGAL = 1e5         # m/s² → mGal
def oss_dem_mass_tc(
    e0, n0, z0,
    dem_df,
    density,
    r_max=30000.0,
    dtheta=1.0,
    dr=50.0
):
    """
    Optimally Selected Sectors (OSS) terrain correction
    DEM mass integration (following Studia Geophysica et Geodaetica, 2020)

    dem_df columns: Easting, Northing, Elev (meters)
    density: kg/m³
    returns: terrain correction in mGal
    """

    # station-centered coordinates
    dx = dem_df["Easting"].values - e0
    dy = dem_df["Northing"].values - n0
    dz = dem_df["Elev"].values - z0

    r = np.sqrt(dx**2 + dy**2)
    theta = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

    mask_global = r <= r_max
    dx = dx[mask_global]
    dy = dy[mask_global]
    dz = dz[mask_global]
    r = r[mask_global]
    theta = theta[mask_global]

    # === estimate DEM cell area ===
    xs = np.unique(dem_df["Easting"].values)
    ys = np.unique(dem_df["Northing"].values)
    dx_cell = np.median(np.diff(xs)) if len(xs) > 1 else 30.0
    dy_cell = np.median(np.diff(ys)) if len(ys) > 1 else dx_cell
    cell_area = dx_cell * dy_cell

    terrain_total = 0.0

    # ===============================
    # ANGULAR OSS (Hammer-style)
    # ===============================
    for th in np.arange(0, 360, dtheta):
        m = (theta >= th) & (theta < th + dtheta)
        if m.sum() == 0:
            continue

        ri = r[m]
        dzi = dz[m]

        # DEM mass integration
        denom = (ri**2 + dzi**2)**1.5
        denom[denom == 0] = np.inf

        dG = G * density * np.sum(dzi * cell_area / denom)
        terrain_total += dG

    return terrain_total * SI2MGAL

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

st.markdown(
    f"""
    <div style="display:flex; align-items:center;">
        <img src="https://raw.githubusercontent.com/dzakyw/AutoGrav/main/logo esdm.png" style="width:200px; margin-right:5px;">
        <div>
            <h2 style="margin-bottom:0;">Auto Grav - Semua Terasa Cepat</h2>
        </div>
    </div>
    <hr>
    """,
    unsafe_allow_html=True
)
st.sidebar.header("Input Files")
grav = st.sidebar.file_uploader("Input Gravity Multi-Sheets (.xlsx)", type=["xlsx"])
demf = st.sidebar.file_uploader("Upload DEM (CSV/XYZ/TIFF)", type=["csv","txt","xyz","tif","tiff"])
kmf = st.sidebar.file_uploader("Koreksi Medan manual (optional jika punya)", type=["csv","xlsx"])
G_base = st.sidebar.number_input("G Absolute di Base", value=0.0)

method = st.sidebar.selectbox("Metode Pengukuran Terrain", ["NAGY (prism)","HAMMER"])
density = st.sidebar.number_input(
    "Density (kg/m³)",
    value=2670.0,
    step=10.0
)

max_radius = st.sidebar.number_input(
    "Max terrain radius (m)",
    value=30000,
    step=5000
)

z_ref = st.sidebar.number_input("z_ref (bottom prism reference, m)", value=0.0)
run = st.sidebar.button("Run")

st.sidebar.write("Notes: densitas biasanya 2670 kg/m³; adjust naik jarak radius untuk mendapatkan pengukuran medan yang jauh (biasanya 5-10 km)")
st.sidebar.subheader("Contoh File Input") 
st.sidebar.write("[Contoh Data Input Gravity](https://github.com/dzakyw/AutoGrav/raw/9bb43e1559c823350f2371360309d84eaab5ea38/sample_gravity.xlsx)") 
st.sidebar.write("[Contoh DEM dengan format .txt](https://github.com/dzakyw/AutoGrav/raw/9bb43e1559c823350f2371360309d84eaab5ea38/sample_dem.csv)") 
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
            st.success(f"DEM loaded: {len(dem)} points.")
        except Exception as e:
            st.error(f"DEM load failed: {e}")
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
        # -----------------------------------------
        # TERRAIN CORRECTION — OSS DEM MASS
        # -----------------------------------------
        
        tc_list = []
        nstations = len(df)
        
        for i in range(nstations):
            e0 = float(df.iloc[i]["Easting"])
            n0 = float(df.iloc[i]["Northing"])
            z0 = float(df.iloc[i]["Elev"])
        
            if method.startswith("NAGY"):
                tc_val = oss_dem_mass_tc(
                    e0, n0, z0,
                    dem_df=dem,
                    density=density,
                    r_max=float(max_radius),
                    dtheta=1.0,   # sesuai permintaan Anda
                    dr=50.0
                )
            else:
                tc_val = hammer_tc(e0, n0, z0, dem)
        
            tc_list.append(tc_val)
        
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
    # ============================================================
    # PARASNIS PLOT (X vs Y)
    # ============================================================
    st.subheader("Plot Parasnis X–Y")
    
    # Only valid rows from ALL data
    mask = df_all[["X-Parasnis", "Y-Parasnis"]].notnull().all(axis=1)
    df_parasnis = df_all.loc[mask].copy()
    if len(df_parasnis) < 2:
        st.warning("Data tidak cukup untuk regresi Parasnis.")
    else:
        # Extract X and Y
        X = df_parasnis["X-Parasnis"].values
        Y = df_parasnis["Y-Parasnis"].values
    
        # Regression
        slope, intercept = np.polyfit(X, Y, 1)
    
        # Generate regression line
        X_line = np.linspace(min(X), max(X), 100)
        Y_line = slope * X_line + intercept
    
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X, Y, s=25, color="blue", label="Data Parasnis", alpha=0.7)
        ax.plot(X_line, Y_line, color="red", linewidth=2, label=f"Regresi: Y = {slope:.5f} X + {intercept:.5f}")
    
        ax.set_xlabel("X-Parasnis (mGal)")
        ax.set_ylabel("Y-Parasnis (mGal)")
        ax.set_title("Diagram Parasnis (X vs Y)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
    
        st.pyplot(fig)
    
        st.success(f"Slope (K) = {slope:.5f}")
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
    st.download_button("Download CSV", df_all.to_csv(index=False).encode("utf-8"), "Hasil Perhitungan.csv")
   










































