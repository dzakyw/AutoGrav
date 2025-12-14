# =============================================================
# AutoGrav – Streamlit Application
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
from math import sqrt
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from PIL import Image
import hashlib
import time

# =============================================================
# AUTHENTICATION
# =============================================================

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

USER_DB = {
    "admin": hash_password("admin"),
    "user":  hash_password("12345"),
}

USER_ROLES = {
    "admin": "admin",
    "user": "viewer",
}

def authenticate(username: str, password: str) -> bool:
    return username in USER_DB and USER_DB[username] == hash_password(password)

def login_page():
    st.title("Welcome to AutoGrav")

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
    if not st.session_state.get("logged_in", False):
        login_page()

    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    logout_button()

require_login()

# =============================================================
# COORDINATE TRANSFORM – UTM REDFEARN
# =============================================================

def latlon_to_utm_redfearn(lat, lon):
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)

    a = 6378137.0
    f = 1 / 298.257223563
    k0 = 0.9996

    b = a * (1 - f)
    e = sqrt(1 - (b / a) ** 2)
    e2 = e ** 2

    zone = np.floor((lon + 180) / 6) + 1
    lon0 = (zone - 1) * 6 - 180 + 3
    lon0 = np.radians(lon0)

    lat = np.radians(lat)
    lon = np.radians(lon)

    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    T = np.tan(lat) ** 2
    C = (e2 / (1 - e2)) * np.cos(lat) ** 2
    A = np.cos(lat) * (lon - lon0)

    M = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * np.sin(2 * lat)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * np.sin(4 * lat)
        - (35 * e2**3 / 3072) * np.sin(6 * lat)
    )

    easting = (
        k0 * N * (A + (1 - T + C) * A**3 / 6 +
        (5 - 18 * T + T**2 + 72 * C - 58 * e2) * A**5 / 120)
        + 500000
    )

    northing = k0 * (
        M + N * np.tan(lat) * (
            A**2 / 2 +
            (5 - T + 9 * C + 4 * C**2) * A**4 / 24 +
            (61 - 58 * T + T**2 + 600 * C - 330 * e2) * A**6 / 720
        )
    )

    northing = np.where(lat < 0, northing + 1e7, northing)

    return easting, northing, zone, None

# =============================================================
# DEM LOADER
# =============================================================

def load_geotiff_without_tfw(file):
    img = Image.open(file)
    arr = np.array(img, dtype=float)
    meta = img.tag_v2

    if 33550 not in meta or 33922 not in meta:
        raise ValueError("GeoTIFF metadata missing")

    scale_x, scale_y, _ = meta[33550]
    tie = meta[33922]

    X0, Y0 = tie[3], tie[4]
    rows, cols = arr.shape

    X = X0 + np.arange(cols) * scale_x
    Y = Y0 - np.arange(rows) * abs(scale_y)

    XX, YY = np.meshgrid(X, Y)

    return pd.DataFrame({
        "Easting": XX.ravel(),
        "Northing": YY.ravel(),
        "Elev": arr.ravel()
    })

def load_dem(file):
    name = file.name.lower()

    if name.endswith((".csv", ".txt", ".xyz")):
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, sep=r"\s+", engine="python")

        df = df.iloc[:, :3]
        df.columns = ["Lon", "Lat", "Elev"]
        df = df.apply(pd.to_numeric, errors="coerce").dropna()

        E, N, _, _ = latlon_to_utm_redfearn(df["Lat"], df["Lon"])
        return pd.DataFrame({"Easting": E, "Northing": N, "Elev": df["Elev"]})

    if name.endswith((".tif", ".tiff")):
        return load_geotiff_without_tfw(file)

    raise ValueError("Unsupported DEM format")

# =============================================================
# GRAVITY CORRECTIONS
# =============================================================

def compute_drift(df, G_base):
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S")
    df["G_read (mGal)"] = pd.to_numeric(df["G_read (mGal)"])

    names = df["Nama"].tolist()
    stations = list(dict.fromkeys(names))
    base = stations[0]
    unknown = stations[1:]

    N = len(unknown) + 1
    A, b = [], []

    tfrac = (
        df["Time"].dt.hour * 3600 +
        df["Time"].dt.minute * 60 +
        df["Time"].dt.second
    ) / 86400

    for i in range(len(df) - 1):
        row = np.zeros(N)
        dG = df.iloc[i + 1]["G_read (mGal)"] - df.iloc[i]["G_read (mGal)"]
        dt = tfrac.iloc[i + 1] - tfrac.iloc[i]

        c = dG
        if names[i + 1] != base:
            row[unknown.index(names[i + 1])] = 1
        else:
            c -= G_base

        if names[i] != base:
            row[unknown.index(names[i])] = -1
        else:
            c += G_base

        row[-1] = dt
        A.append(row)
        b.append(c)

    x, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)

    Gmap = {base: G_base}
    for i, stn in enumerate(unknown):
        Gmap[stn] = x[i]

    return Gmap, x[-1]

def latitude_correction(lat):
    phi = np.radians(lat)
    return 978032.67715 * (
        1 + 0.0053024 * np.sin(phi)**2 -
        0.0000059 * np.sin(2 * phi)**2
    )

def free_air(elev):
    return 0.3086 * elev

# =============================================================
# HAMMER TERRAIN CORRECTION
# =============================================================

HAMMER_R = np.array([2, 6, 18, 54, 162, 486, 1458, 4374])
HAMMER_F = np.array([0.00027, 0.00019, 0.00013, 0.00009,
                     0.00006, 0.00004, 0.000025, 0.000015])

def hammer_tc(e0, n0, z0, dem):
    dx = dem["Easting"] - e0
    dy = dem["Northing"] - n0
    dist = np.sqrt(dx**2 + dy**2)

    tc = 0.0
    inner = 0.0

    for r, f in zip(HAMMER_R, HAMMER_F):
        m = (dist >= inner) & (dist < r)
        if m.any():
            tc += f * (dem.loc[m, "Elev"].mean() - z0)
        inner = r

    return tc

# =============================================================
# UI
# =============================================================

st.markdown("""
<div style="display:flex;align-items:center">
<img src="https://raw.githubusercontent.com/dzakyw/AutoGrav/main/logo esdm.png" width="180">
<h2 style="margin-left:10px">AutoGrav – Semua Terasa Cepat</h2>
</div><hr>
""", unsafe_allow_html=True)

st.sidebar.header("Input Files")
grav = st.sidebar.file_uploader("Gravity Excel (.xlsx)", ["xlsx"])
demf = st.sidebar.file_uploader("DEM (CSV / XYZ / TIFF)", ["csv", "txt", "xyz", "tif", "tiff"])
G_base = st.sidebar.number_input("G Absolute Base (mGal)", value=0.0)
method = st.sidebar.selectbox("Terrain Method", ["HAMMER"])
run = st.sidebar.button("Run")
