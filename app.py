import streamlit as st
import pandas as pd
import numpy as np
from math import sqrt
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import hashlib
import time

# ============================================================
# LOGIN SYSTEM
# ============================================================

def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

USER_DB = {
    "admin": hash_password("admin"),
    "user": hash_password("12345"),
}

def login():
    if "logged" not in st.session_state:
        st.title("AutoGrav Login")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if u in USER_DB and USER_DB[u] == hash_password(p):
                st.session_state.logged = True
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Login gagal")
        st.stop()

    st.sidebar.success(f"Login: {st.session_state.user}")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

login()

# ============================================================
# UTM CONVERSION (REDFEARN)
# ============================================================

def latlon_to_utm(lat, lon):
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)

    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2*f - f*f
    k0 = 0.9996

    zone = np.floor((lon + 180)/6) + 1
    lon0 = (zone - 1)*6 - 180 + 3
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    lon0_r = np.radians(lon0)

    N = a / np.sqrt(1 - e2*np.sin(lat_r)**2)
    T = np.tan(lat_r)**2
    C = e2/(1-e2)*np.cos(lat_r)**2
    A = np.cos(lat_r)*(lon_r - lon0_r)

    M = a*((1-e2/4-3*e2**2/64-5*e2**3/256)*lat_r
         -(3*e2/8+3*e2**2/32+45*e2**3/1024)*np.sin(2*lat_r)
         +(15*e2**2/256+45*e2**3/1024)*np.sin(4*lat_r))

    E = k0*N*(A+(1-T+C)*A**3/6)+500000
    Nn = k0*(M+N*np.tan(lat_r)*(A**2/2))

    Nn = np.where(lat<0, Nn+10000000, Nn)
    return E, Nn

# ============================================================
# OSS TERRAIN CORRECTION — DEM MASS (PAPER STYLE)
# ============================================================

G = 6.67430e-11
SI2MGAL = 1e5
TWO_PI = 2*np.pi

def oss_dem_mass_tc(e0, n0, z0, dem, density, r_max=30000, dtheta=1.0):
    dx = dem["Easting"].values - e0
    dy = dem["Northing"].values - n0
    dz = dem["Elev"].values - z0

    r = np.sqrt(dx*dx + dy*dy)
    theta = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

    mask = r <= r_max
    r = r[mask]
    dz = dz[mask]
    theta = theta[mask]

    # cell area estimation
    xs = np.unique(dem["Easting"].values)
    ys = np.unique(dem["Northing"].values)
    dx_cell = np.median(np.diff(xs))
    dy_cell = np.median(np.diff(ys))
    cell_area = dx_cell * dy_cell

    terrain = 0.0

    for th in np.arange(0, 360, dtheta):
        m = (theta >= th) & (theta < th + dtheta)
        if m.sum() == 0:
            continue

        ri = r[m]
        hi = dz[m]

        denom = (ri**2 + hi**2)**1.5
        denom[denom == 0] = np.inf

        dg = TWO_PI * G * density * np.sum(hi * cell_area / denom)
        terrain += dg

    return terrain * SI2MGAL

# ============================================================
# BASIC GRAVITY CORRECTIONS
# ============================================================

def lat_corr(lat):
    p = np.radians(lat)
    return 978032.67715*(1+0.0053024*np.sin(p)**2-0.0000059*np.sin(2*p)**2)

def free_air(h):
    return 0.3086*h

# ============================================================
# DRIFT SOLVER
# ============================================================

def solve_drift(df, G_base):
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S")
    frac = (df["Time"].dt.hour*3600 +
            df["Time"].dt.minute*60 +
            df["Time"].dt.second)/86400

    names = df["Nama"].astype(str).tolist()
    uniq = list(dict.fromkeys(names))
    base = uniq[0]
    unk = [s for s in uniq if s != base]

    A=[]; b=[]
    for i in range(len(df)-1):
        row = np.zeros(len(unk)+1)
        dG = df.iloc[i+1]["G_read (mGal)"] - df.iloc[i]["G_read (mGal)"]
        dt = frac.iloc[i+1]-frac.iloc[i]

        c = dG
        if names[i+1]!=base: row[unk.index(names[i+1])] = 1
        else: c -= G_base
        if names[i]!=base: row[unk.index(names[i])] = -1
        else: c += G_base

        row[-1]=dt
        A.append(row); b.append(c)

    x,*_ = np.linalg.lstsq(np.array(A),np.array(b),rcond=None)
    Gmap={base:G_base}
    for i,s in enumerate(unk): Gmap[s]=x[i]
    return Gmap

# ============================================================
# UI
# ============================================================

st.title("AutoGrav – OSS Terrain (DEM Mass Integration)")

grav = st.sidebar.file_uploader("Gravity Excel (.xlsx)", type=["xlsx"])
demf = st.sidebar.file_uploader("DEM CSV (Lon,Lat,Elev)", type=["csv"])
density = st.sidebar.number_input("Density (kg/m³)", value=2670.0)
rmax = st.sidebar.number_input("Radius OSS (m)", value=30000)
G_base = st.sidebar.number_input("G_base (mGal)", value=0.0)

run = st.sidebar.button("Run")

# ============================================================
# PROCESS
# ============================================================

if run:

    dem_raw = pd.read_csv(demf)
    dem_raw.columns=["Lon","Lat","Elev"]
    E,N = latlon_to_utm(dem_raw["Lat"], dem_raw["Lon"])
    dem = pd.DataFrame({"Easting":E,"Northing":N,"Elev":dem_raw["Elev"]})

    xls = pd.ExcelFile(grav)
    dfs=[]

    for sh in xls.sheet_names:
        df = pd.read_excel(grav, sheet_name=sh)

        E,N = latlon_to_utm(df["Lat"], df["Lon"])
        df["Easting"]=E; df["Northing"]=N

        Gmap = solve_drift(df, G_base)
        df["G_read (mGal)"] = df["Nama"].map(Gmap)

        df["Koreksi Lintang"] = lat_corr(df["Lat"])
        df["FAC"] = free_air(df["Elev"])
        df["FAA"] = df["G_read (mGal)"] - df["Koreksi Lintang"] + df["FAC"]

        tc=[]
        for i in range(len(df)):
            tc.append(
                oss_dem_mass_tc(
                    df.iloc[i]["Easting"],
                    df.iloc[i]["Northing"],
                    df.iloc[i]["Elev"],
                    dem,
                    density,
                    rmax
                )
            )

        df["Koreksi Medan"]=tc
        df["Hari"]=sh
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    # Parasnis
    out["X-Parasnis"]=0.04192*out["Elev"]-out["Koreksi Medan"]
    out["Y-Parasnis"]=out["FAC"]
    k,_=np.polyfit(out["X-Parasnis"],out["Y-Parasnis"],1)

    out["Bouger Corr"]=0.04192*k*out["Elev"]
    out["SBA"]=out["FAA"]-out["Bouger Corr"]
    out["CBA"]=out["SBA"]+out["Koreksi Medan"]

    st.success("Selesai")
    st.dataframe(out.head())

    st.download_button("Download CSV",
        out.to_csv(index=False).encode(),
        "autograv_output.csv"
    )

