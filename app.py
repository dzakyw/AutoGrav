# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 07:28:38 2025

@author: asus
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from pyproj import Transformer
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg

_LOCK = RendererAgg.lock

# ======================================
# SETTINGS
# ======================================
use_nagy = True           # True = prism method, False = Hammer
max_radius = 10000        # radius for terrain correction
density = 2670            # density for Nagy
cell_size_override = None # auto DEM grid size


# ======================================
# DRIFT CORRECTION
# ======================================
def compute_drift(df, G_base):
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="raise")
    df["G_read (mGal)"] = pd.to_numeric(df["G_read (mGal)"], errors="coerce")

    names = df["Nama"].astype(str).tolist()
    uniq = list(dict.fromkeys(names))
    base = uniq[0]
    unknown = [s for s in uniq if s != base]

    N = len(unknown) + 1
    A = []
    b = []

    frac = (df["Time"].dt.hour*3600 + df["Time"].dt.minute*60 + df["Time"].dt.second)/86400

    for i in range(len(df)-1):
        row = np.zeros(N)
        dG = df["G_read (mGal)"].iloc[i+1] - df["G_read (mGal)"].iloc[i]
        dt = frac.iloc[i+1] - frac.iloc[i]

        const = dG
        s1 = names[i]
        s2 = names[i+1]

        if s2 != base:
            row[unknown.index(s2)] = 1
        else:
            const -= G_base

        if s1 != base:
            row[unknown.index(s1)] = -1
        else:
            const += G_base

        row[-1] = dt
        A.append(row)
        b.append(const)

    A = np.array(A, float)
    b = np.array(b, float)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    D = x[-1]

    Gmap = {base: G_base}
    for i, s in enumerate(unknown):
        Gmap[s] = x[i]

    return Gmap, D


# ======================================
# LATITUDE & FREE-AIR
# ======================================
def latitude_correction(lat):
    phi = np.radians(lat)
    return 978032.67715 * (1 + 0.0053024*np.sin(phi)**2 - 0.0000059*np.sin(2*phi)**2)

def free_air(elev):
    return 0.3086 * elev


# ======================================
# HAMMER METHOD
# ======================================
HAMMER_R = np.array([25,100,200,500,2000,5000])
HAMMER_F = np.array([0.035,0.03,0.025,0.02,0.015,0.01])

def hammer_tc(e0,n0,z0,dem_df):
    dx = dem_df["Easting"].to_numpy() - e0
    dy = dem_df["Northing"].to_numpy() - n0
    dist = np.sqrt(dx*dx + dy*dy)
    Z = dem_df["Elev"].to_numpy()

    tc=0
    inner=0
    for i,outer in enumerate(HAMMER_R):
        mask = (dist>=inner)&(dist<outer)
        if mask.sum()>0:
            dh = Z[mask].mean() - z0
            tc += HAMMER_F[i]*dh
        inner = outer
    return tc


# ======================================
# NAGY / PRISM
# ======================================
G_SI = 6.67430e-11
M2MGAL = 1e5

def prism_term(x1,x2,y1,y2,z1,z2,px,py,pz):
    X=[x1-px, x2-px]
    Y=[y1-py, y2-py]
    Z=[z1-pz, z2-pz]
    g=0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                xi, yj, zk = X[i], Y[j], Z[k]
                R = np.sqrt(xi*xi + yj*yj + zk*zk) + 1e-20
                s = (-1)**(i+j+k)
                g += s * (
                    xi*np.log(yj+R) +
                    yj*np.log(xi+R) -
                    zk*np.arctan2(xi*yj, zk*R)
                )
    return g


def nagy_tc(e0,n0,z0, dem_df, maxr, rho, cell):
    dx = dem_df["Easting"] - e0
    dy = dem_df["Northing"] - n0
    r = np.sqrt(dx*dx + dy*dy)
    block = dem_df[r<=maxr].copy()
    if block.empty:
        return 0.0

    if cell is None:
        xs = np.sort(block["Easting"].unique())
        ys = np.sort(block["Northing"].unique())
        if len(xs)>1:
            dxm = np.median(np.diff(xs))
        else:
            dxm=25
        if len(ys)>1:
            dym = np.median(np.diff(ys))
        else:
            dym=dxm
        cell = max(dxm, dym)

    minx = block["Easting"].min()
    maxx = block["Easting"].max()
    miny = block["Northing"].min()
    maxy = block["Northing"].max()

    block["ix"] = ((block["Easting"]-minx)/cell).astype(int)
    block["iy"] = ((block["Northing"]-miny)/cell).astype(int)
    grouped = block.groupby(["ix","iy"])["Elev"].mean().reset_index()

    z_bottom = dem_df["Elev"].min() - 1000
    gz_sum=0
    for _,r in grouped.iterrows():
        ix,iy = int(r["ix"]), int(r["iy"])
        ztop = r["Elev"]
        x1 = minx + ix*cell
        x2 = x1 + cell
        y1 = miny + iy*cell
        y2 = y1 + cell
        gz_sum += prism_term(x1,x2,y1,y2, z_bottom, ztop, e0,n0,z0)

    gz_si = G_SI * rho * gz_sum
    return gz_si*M2MGAL


# ======================================
# DEM Loader (robust)
# ======================================
def load_dem(file):
    df = None
    try:
        df = pd.read_csv(file)
        if df.shape[1] == 1:
            raise
    except:
        file.seek(0)
        try:
            df = pd.read_csv(file, sep=r"\s+", engine="python", header=None)
        except:
            file.seek(0)
            df = pd.read_excel(file)

    if df.shape[1] == 1:
        df = df.iloc[:,0].str.split(r"[\s,]+", expand=True)

    if df.shape[1] < 3:
        raise ValueError("DEM must have ≥ 3 columns")

    # assume Lon,Lat,Elev
    df = df.iloc[:, :3]
    df.columns = ["Lon","Lat","Elev"]
    df["Lon"] = pd.to_numeric(df["Lon"], errors="coerce")
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["Elev"] = pd.to_numeric(df["Elev"], errors="coerce")
    df.dropna(inplace=True)

    # convert to UTM
    meanlon = df["Lon"].mean()
    meanlat = df["Lat"].mean()
    zone = int((meanlon+180)/6)+1
    hemi = "north" if meanlat>=0 else "south"
    proj = f"+proj=utm +zone={zone} +{hemi} +ellps=WGS84 +units=m"
    tr = Transformer.from_crs("epsg:4326", proj, always_xy=True)
    E,N = tr.transform(df["Lon"], df["Lat"])

    return pd.DataFrame({"Easting":E, "Northing":N, "Elev":df["Elev"]})


# ======================================
# STREAMLIT UI
# ======================================
st.title("GravCore – Streamlit Edition")
st.markdown("Aplikasi Pengolahan Data Gravitasi (Drift, FAA, Bouguer, Terrain Correction Hammer/Nagy)")

st.sidebar.header("Input Files")

grav_file = st.sidebar.file_uploader("Upload Gravity Excel Multi-hari", type=["xlsx"])
km_file   = st.sidebar.file_uploader("Upload Koreksi Medan Manual (opsional)", type=["csv","xlsx"])
dem_file  = st.sidebar.file_uploader("Upload DEM (Lon,Lat,Elev)", type=["csv","txt","xyz","xlsx"])

G_base = st.sidebar.number_input("G Base nilai absolut (mGal)", value=0.0)
method = st.sidebar.selectbox("Terrain Method", ["NAGY (High Accuracy)","HAMMER (Fast)"])
use_nagy_method = (method.startswith("NAGY"))

process_button = st.sidebar.button("Proses Data")

st.sidebar.markdown("### Download Contoh File")
st.sidebar.write("[Contoh Gravity Excel](https://files.catbox.moe/5t5ez6.xlsx)")
st.sidebar.write("[Contoh DEM](https://files.catbox.moe/83a7y7.csv)")
st.sidebar.write("[Contoh Koreksi Medan](https://files.catbox.moe/1zz2z1.csv)")


# ======================================
# MAIN PROCESS
# ======================================
if process_button:
    if grav_file is None:
        st.error("Upload file gravity dahulu.")
        st.stop()

    # read excel
    xls = pd.ExcelFile(grav_file)
    days = []

    # load dem if exists
    if dem_file:
        dem_df = load_dem(dem_file)
        st.success(f"DEM loaded: {len(dem_df)} points")
    else:
        dem_df = None

    # load km if exists
    if km_file:
        try:
            km_df = pd.read_csv(km_file)
        except:
            km_file.seek(0)
            km_df = pd.read_excel(km_file)
        km_df["Koreksi_Medan"] = pd.to_numeric(km_df["Koreksi_Medan"], errors="coerce")
    else:
        km_df = None

    for sh in xls.sheet_names:
        df = pd.read_excel(grav_file, sheet_name=sh)
        req = {"Nama","Time","G_read (mGal)","Lat","Lon","Elev"}
        if not req.issubset(df.columns):
            st.warning(f"Sheet {sh} dilewati (kolom tidak lengkap)")
            continue

        # convert to UTM
        meanlon = df["Lon"].mean()
        zone = int((meanlon+180)/6)+1
        hemi = "north" if df["Lat"].mean()>=0 else "south"
        proj = f"+proj=utm +zone={zone} +{hemi} +ellps=WGS84 +units=m"
        tr = Transformer.from_crs("epsg:4326", proj, always_xy=True)
        E,N = tr.transform(df["Lon"], df["Lat"])
        df["Easting"] = E
        df["Northing"] = N

        # compute drift
        Gmap, drift = compute_drift(df, G_base)
        df["G_corrected"] = df["Nama"].map(Gmap)
        df["G_read (mGal)"] = df["G_corrected"]

        # basic corrections
        df["Koreksi Lintang"] = latitude_correction(df["Lat"])
        df["Free Air Correction"] = free_air(df["Elev"])
        df["FAA"] = df["G_read (mGal)"] - df["Koreksi Lintang"] + df["Free Air Correction"]

        # terrain
        if dem_df is not None:
            tlist=[]
            for i in range(len(df)):
                e0=df.iloc[i]["Easting"]
                n0=df.iloc[i]["Northing"]
                z0=df.iloc[i]["Elev"]
                if use_nagy_method:
                    tc_val = nagy_tc(e0,n0,z0, dem_df, max_radius, density, cell_size_override)
                else:
                    tc_val = hammer_tc(e0,n0,z0, dem_df)
                tlist.append(tc_val)
            df["Koreksi Medan"] = tlist
        else:
            if km_df is None:
                st.error("Tidak ada DEM atau file koreksi medan manual.")
                st.stop()
            df["Koreksi Medan"] = df["Nama"].map(km_df.set_index("Nama")["Koreksi_Medan"])

        # parasnis
        df["X-Parasnis"] = 0.04192*df["Elev"] - df["Koreksi Medan"]
        df["Y-Parasnis"] = df["Free Air Correction"]
        df["Hari"] = sh
        days.append(df)

    if len(days)==0:
        st.error("Tidak ada sheet valid.")
        st.stop()

    df_total = pd.concat(days, ignore_index=True)

    # compute slope K
    mask = df_total[["X-Parasnis","Y-Parasnis"]].notnull().all(axis=1)
    if mask.sum()>=2:
        a,b = np.polyfit(df_total.loc[mask,"X-Parasnis"], df_total.loc[mask,"Y-Parasnis"],1)
        K=a
    else:
        K=np.nan

    df_total["Bouger Correction"] = 0.04192*K*df_total["Elev"]
    df_total["Simple Bouger Anomaly"] = df_total["FAA"] - df_total["Bouger Correction"]
    df_total["Complete Bouger Correction"] = df_total["Simple Bouger Anomaly"] + df_total["Koreksi Medan"]

    st.success("Proses selesai!")

    st.subheader("Preview Hasil")
    st.dataframe(df_total.head(20))

    # ======================================
    # CONTOUR PLOT
    # ======================================
    x = df_total["Easting"]
    y = df_total["Northing"]
    gx = np.linspace(x.min(), x.max(), 200)
    gy = np.linspace(y.min(), y.max(), 200)
    GX,GY = np.meshgrid(gx,gy)

    def plot_contour(z, title):
        with _LOCK:
            Z = griddata((x,y), z, (GX,GY), method="cubic")
            fig, ax = plt.subplots(figsize=(8,6))
            c=ax.contourf(GX,GY,Z,40,cmap="jet")
            ax.scatter(x,y,c=z,cmap="jet",edgecolor="k",s=15)
            ax.set_title(title)
            fig.colorbar(c, ax=ax)
            st.pyplot(fig)

    st.subheader("Peta Kontur")
    plot_contour(df_total["Complete Bouger Correction"], "Complete Bouguer Anomaly (CBA)")
    plot_contour(df_total["Simple Bouger Anomaly"], "Simple Bouguer Anomaly (SBA)")
    plot_contour(df_total["Elev"], "Elevasi (m)")

    # DOWNLOAD
    st.subheader("Download Hasil")
    csv = df_total.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "gravcore_output.csv")
