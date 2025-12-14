import streamlit as st
import pandas as pd
import numpy as np
from math import sqrt
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import io, time
from PIL import Image
import os
import hashlib

# ---------------------------------------------
# LOGIN SYSTEM (UNCHANGED)
# ---------------------------------------------
def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

USER_DB = {
    "admin": hash_password("admin"),
    "user":  hash_password("12345"),
}

USER_ROLES = {
    "admin": "admin",
    "user": "viewer",
}

def authenticate(username, password):
    if username in USER_DB:
        return USER_DB[username] == hash_password(password)
    return False

def login_page():
    st.title("Welcome to Auto Grav")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = USER_ROLES.get(username, "viewer")
            st.rerun()
        else:
            st.error("Invalid username or password")
    st.stop()

def logout_button():
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

def require_login():
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        login_page()
    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    logout_button()

require_login()

# ---------------------------------------------
# UTM CONVERSION (REDFEARN) — UNCHANGED
# ---------------------------------------------
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

# ---------------------------------------------
# DEM LOADER (UNCHANGED)
# ---------------------------------------------
def load_dem(file):
    df = pd.read_csv(file)
    df = df.iloc[:, :3]
    df.columns = ["Lon","Lat","Elev"]
    df["Lon"] = pd.to_numeric(df["Lon"], errors="coerce")
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["Elev"] = pd.to_numeric(df["Elev"], errors="coerce")
    df.dropna(inplace=True)
    E, N, _, _ = latlon_to_utm_redfearn(df["Lat"], df["Lon"])
    return pd.DataFrame({"Easting":E,"Northing":N,"Elev":df["Elev"]})

# ---------------------------------------------
# BASIC CORRECTIONS (UNCHANGED)
# ---------------------------------------------
def latitude_correction(lat):
    phi = np.radians(lat)
    return 978032.67715 * (1 + 0.0053024*np.sin(phi)**2 - 0.0000059*np.sin(2*phi)**2)

def free_air(elev):
    return 0.3086 * elev

# ---------------------------------------------
# DRIFT SOLVER (UNCHANGED)
# ---------------------------------------------
def compute_drift(df, G_base):
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S")
    frac = (df["Time"].dt.hour*3600 + df["Time"].dt.minute*60 + df["Time"].dt.second)/86400
    names = df["Nama"].astype(str).tolist()
    uniq = list(dict.fromkeys(names))
    base = uniq[0]
    unknown = [s for s in uniq if s != base]
    A=[]; b=[]
    for i in range(len(df)-1):
        row = np.zeros(len(unknown)+1)
        dG = df.iloc[i+1]["G_read (mGal)"] - df.iloc[i]["G_read (mGal)"]
        dt = frac.iloc[i+1] - frac.iloc[i]
        c = dG
        if names[i+1] != base: row[unknown.index(names[i+1])] = 1
        else: c -= G_base
        if names[i] != base: row[unknown.index(names[i])] = -1
        else: c += G_base
        row[-1] = dt
        A.append(row); b.append(c)
    x,*_ = np.linalg.lstsq(np.array(A),np.array(b),rcond=None)
    Gmap = {base:G_base}
    for i,s in enumerate(unknown): Gmap[s] = x[i]
    return Gmap

# ============================================================
# OSS TERRAIN CORRECTION — DIPERBAIKI (INI INTINYA)
# ============================================================
G = 6.67430e-11
SI2MGAL = 1e5
TWO_PI = 2*np.pi

def oss_dem_mass_tc(e0, n0, z0, dem_df, density, r_max=30000.0, dtheta=1.0):
    dx = dem_df["Easting"].values - e0
    dy = dem_df["Northing"].values - n0
    r = np.sqrt(dx*dx + dy*dy)
    mask = r <= r_max

    dx = dx[mask]
    dy = dy[mask]
    r  = r[mask]
    z  = dem_df["Elev"].values[mask]

    theta = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

    xs = np.unique(dem_df["Easting"].values)
    ys = np.unique(dem_df["Northing"].values)
    dx_cell = np.median(np.diff(xs))
    dy_cell = np.median(np.diff(ys))
    cell_area = dx_cell * dy_cell

    terrain = 0.0

    for th in np.arange(0, 360, dtheta):
        m = (theta >= th) & (theta < th + dtheta)
        if m.sum() < 5:
            continue

        ri = r[m]
        zi = z[m]

        w = 1.0 / (ri**3 + 1e-20)
        z_m = np.sum(w * zi) / np.sum(w)

        hi = zi - z_m
        denom = (ri**2 + hi**2)**1.5
        denom[denom == 0] = np.inf

        dg = TWO_PI * G * density * np.sum(hi * cell_area / denom)
        terrain += dg

    return terrain * SI2MGAL

# ---------------------------------------------
# UI — UNCHANGED
# ---------------------------------------------
st.markdown("""
<h2>AutoGrav — OSS Terrain Correction</h2>
""", unsafe_allow_html=True)

st.sidebar.header("Input Files")
grav = st.sidebar.file_uploader("Gravity Excel (.xlsx)", type=["xlsx"])
demf = st.sidebar.file_uploader("DEM CSV (Lon,Lat,Elev)", type=["csv"])
G_base = st.sidebar.number_input("G Absolute Base", value=0.0)
density = st.sidebar.number_input("Density (kg/m³)", value=2670.0)
max_radius = st.sidebar.number_input("Terrain Radius (m)", value=30000)
run = st.sidebar.button("Run")

st.sidebar.subheader("Contoh File")
st.sidebar.write("[Gravity Example](https://github.com/dzakyw/AutoGrav/raw/main/sample_gravity.xlsx)")
st.sidebar.write("[DEM Example](https://github.com/dzakyw/AutoGrav/raw/main/sample_dem.csv)")

# ---------------------------------------------
# PROCESSING
# ---------------------------------------------
if run:
    dem = load_dem(demf)
    xls = pd.ExcelFile(grav)
    all_df = []

    for sh in xls.sheet_names:
        df = pd.read_excel(grav, sheet_name=sh)
        E,N,_,_ = latlon_to_utm_redfearn(df["Lat"], df["Lon"])
        df["Easting"]=E; df["Northing"]=N

        Gmap = compute_drift(df, G_base)
        df["G_read (mGal)"] = df["Nama"].map(Gmap)

        df["Koreksi Lintang"] = latitude_correction(df["Lat"])
        df["Free Air Correction"] = free_air(df["Elev"])
        df["FAA"] = df["G_read (mGal)"] - df["Koreksi Lintang"] + df["Free Air Correction"]

        tc=[]
        for i in range(len(df)):
            tc.append(
                oss_dem_mass_tc(
                    df.iloc[i]["Easting"],
                    df.iloc[i]["Northing"],
                    df.iloc[i]["Elev"],
                    dem,
                    density,
                    max_radius
                )
            )
        df["Koreksi Medan"]=tc
        df["X-Parasnis"]=0.04192*df["Elev"]-df["Koreksi Medan"]
        df["Y-Parasnis"]=df["Free Air Correction"]
        df["Hari"]=sh
        all_df.append(df)

    out = pd.concat(all_df, ignore_index=True)

    k,_ = np.polyfit(out["X-Parasnis"], out["Y-Parasnis"], 1)
    out["Bouger Correction"]=0.04192*k*out["Elev"]
    out["SBA"]=out["FAA"]-out["Bouger Correction"]
    out["CBA"]=out["SBA"]+out["Koreksi Medan"]

    st.success("Selesai")
    st.dataframe(out.head())

    st.download_button(
        "Download CSV",
        out.to_csv(index=False).encode(),
        "autograv_output.csv"
    )
