import streamlit as st
import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# ============================================================
# 1) UTM CONVERSION WITHOUT PYPROJ (REDFEARN / KRÜGER)
# ============================================================

def latlon_to_utm_redfearn(lat, lon):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    # WGS84
    a = 6378137.0
    f = 1 / 298.257223563
    k0 = 0.9996

    b = a * (1 - f)
    e = sqrt(1 - (b/a)**2)
    e2 = e*e

    zone = np.floor((lon + 180) / 6) + 1
    lon0 = (zone - 1)*6 - 180 + 3  # central meridian
    lon0_rad = np.radians(lon0)

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    T = np.tan(lat_rad)**2
    C = (e2 / (1 - e2)) * np.cos(lat_rad)**2
    A = np.cos(lat_rad) * (lon_rad - lon0_rad)

    # Meridional Arc
    M = (a * ((1 - e2/4 - 3*e2**2/64 - 5*e2**3/256) * lat_rad
        - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024) * np.sin(2 * lat_rad)
        + (15*e2**2/256 + 45*e2**3/1024) * np.sin(4 * lat_rad)
        - (35*e2**3/3072) * np.sin(6 * lat_rad)))

    easting = k0 * N * (
        A + (1 - T + C)*A**3/6 + (5 - 18*T + T*T + 72*C - 58*(e2))*A**5/120
    ) + 500000

    northing = k0 * (M + N*np.tan(lat_rad)*(
        A*A/2 + (5 - T + 9*C + 4*C*C)*A**4/24 +
        (61 - 58*T + T*T + 600*C - 330*(e2))*A**6/720
    ))

    hemisphere = np.where(lat >= 0, "north", "south")
    northing = np.where(hemisphere == "south", northing + 10000000, northing)

    return easting, northing, zone, hemisphere


# ============================================================
# 2) DRIFT CORRECTION
# ============================================================

def compute_drift(df, Gbase):
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="raise")
    df["G_read (mGal)"] = pd.to_numeric(df["G_read (mGal)"], errors="coerce")
    
    names = df["Nama"].tolist()
    uniq = list(dict.fromkeys(names))
    
    base = uniq[0]
    unknown = [s for s in uniq if s != base]

    N = len(unknown) + 1
    A = []
    b = []

    frac = (df["Time"].dt.hour*3600 + df["Time"].dt.minute*60 + df["Time"].dt.second) / 86400

    for i in range(len(df) - 1):
        row = np.zeros(N)
        dG = df["G_read (mGal)"].iloc[i+1] - df["G_read (mGal)"].iloc[i]
        dt = frac.iloc[i+1] - frac.iloc[i]

        const = dG
        s1 = names[i]
        s2 = names[i+1]

        # Station j
        if s2 != base:
            row[unknown.index(s2)] = 1
        else:
            const -= Gbase

        # Station i
        if s1 != base:
            row[unknown.index(s1)] = -1
        else:
            const += Gbase

        # Drift coefficient
        row[-1] = dt
        A.append(row)
        b.append(const)

    A = np.array(A, float)
    b = np.array(b, float)

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    drift = x[-1]

    sol = {base: Gbase}
    for i, st in enumerate(unknown):
        sol[st] = x[i]

    return sol, drift


# ============================================================
# 3) BASIC CORRECTIONS
# ============================================================

def latitude_correction(lat):
    phi = np.radians(lat)
    return 978032.67715 * (1 + 0.0053024 * np.sin(phi)**2 - 0.0000059 * np.sin(2*phi)**2)

def free_air(elev):
    return 0.3086 * elev


# ============================================================
# 4) HAMMER TC
# ============================================================

HAMMER_R = np.array([25,100,200,500,2000,5000])
HAMMER_F = np.array([0.035,0.030,0.025,0.020,0.015,0.010])

def hammer_tc(e0, n0, z0, dem):
    dx = dem["Easting"] - e0
    dy = dem["Northing"] - n0
    dist = np.sqrt(dx*dx + dy*dy)
    Z = dem["Elev"]

    tc=0
    inner=0
    for i, outer in enumerate(HAMMER_R):
        mask = (dist>=inner) & (dist<outer)
        if mask.sum()>0:
            dh = Z[mask].mean() - z0
            tc += HAMMER_F[i] * dh
        inner = outer

    return tc


# ============================================================
# 5) NAGY PRISM
# ============================================================

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
                R = sqrt(xi*xi + yj*yj + zk*zk) + 1e-12
                s = (-1)**(i+j+k)
                g += s*( xi*np.log(abs(yj+R)) +
                         yj*np.log(abs(xi+R)) -
                         zk*np.arctan2(xi*yj, zk*R) )
    return g

def nagy_tc(e0,n0,z0,dem,maxr=10000,density=2670):
    dx = dem["Easting"]-e0
    dy = dem["Northing"]-n0
    r = np.sqrt(dx*dx + dy*dy)
    block = dem[r<=maxr].copy()

    if block.empty:
        return 0.0

    xs = np.sort(block["Easting"].unique())
    ys = np.sort(block["Northing"].unique())
    if len(xs)>1:
        cell = np.median(np.diff(xs))
    else:
        cell = 25
    if len(ys)>1:
        cell = max(cell, np.median(np.diff(ys)))

    minx = block["Easting"].min()
    miny = block["Northing"].min()
    zbottom = dem["Elev"].min() - 1000

    block["ix"] = ((block["Easting"]-minx)/cell).astype(int)
    block["iy"] = ((block["Northing"]-miny)/cell).astype(int)

    grouped = block.groupby(["ix","iy"])["Elev"].mean().reset_index()

    gz_sum=0
    for _, row in grouped.iterrows():
        ix,iy = int(row["ix"]), int(row["iy"])
        ztop  = row["Elev"]
        x1 = minx + ix*cell
        x2 = x1 + cell
        y1 = miny + iy*cell
        y2 = y1 + cell
        gz_sum += prism_term(x1,x2,y1,y2,zbottom,ztop,e0,n0,z0)

    gz_si = G_SI * density * gz_sum
    return gz_si*M2MGAL


# ============================================================
# 6) DEM LOADER (Lon,Lat,Elev → UTM)
# ============================================================


def load_dem(file):
    try:
        df = pd.read_csv(file)
        if df.shape[1] == 1:
            raise
    except:
        file.seek(0)
        df = pd.read_csv(file, sep=r"\s+", engine="python", header=None)

    if df.shape[1] < 3:
        raise ValueError("DEM must have ≥ 3 columns")

    df = df.iloc[:,:3]
    df.columns=["Lon","Lat","Elev"]
    df["Lon"] = pd.to_numeric(df["Lon"], errors="coerce")
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["Elev"] = pd.to_numeric(df["Elev"], errors="coerce")
    df.dropna(inplace=True)

    E,N,_,_ = latlon_to_utm_redfearn(df["Lat"], df["Lon"])
    return pd.DataFrame({"Easting":E,"Northing":N,"Elev":df["Elev"]})


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("GravCore – Streamlit Flexible Terrain Correction")
st.caption("Drift • FAA • Bouguer • Hammer/Nagy • Contour • DEM or Manual Terrain")

st.sidebar.header("Input File")
grav = st.sidebar.file_uploader("Gravity Excel Multi-Hari", type=["xlsx"])
kmf  = st.sidebar.file_uploader("Koreksi Medan Manual (Opsional)", type=["csv","xlsx"])
demf = st.sidebar.file_uploader("File DEM (Opsional)", type=["csv","txt","xyz","xlsx"])

G_base = st.sidebar.number_input("G_base (mGal)", value=0.0)
method = st.sidebar.selectbox("Terrain Method (untuk DEM)", ["NAGY (High Accuracy)", "HAMMER (Fast)"])
process = st.sidebar.button("PROSES")

st.sidebar.subheader("Contoh File:")
st.sidebar.write("[Gravity Sample](https://files.catbox.moe/5t5ez6.xlsx)")
st.sidebar.write("[DEM Sample](https://files.catbox.moe/83a7y7.csv)")
st.sidebar.write("[Koreksi Medan Sample](https://files.catbox.moe/1zz2z1.csv)")

# ============================================================
# PROCESS
# ============================================================

if process:
    if grav is None:
        st.error("Upload file gravity terlebih dahulu.")
        st.stop()

    # Load manual terrain correction
    if kmf:
        try:
            km = pd.read_csv(kmf)
        except:
            km = pd.read_excel(kmf)
        km["Koreksi_Medan"] = pd.to_numeric(km["Koreksi_Medan"], errors="coerce")
        km_map = km.set_index("Nama")["Koreksi_Medan"].to_dict()
    else:
        km_map = None

    # Load DEM
    if demf:
        dem = load_dem(demf)
        st.success(f"DEM loaded: {len(dem)} grid points")
    else:
        dem = None

    # Final logic: At least one terrain source must exist
    if (km_map is None) and (dem is None):
        st.error("Anda harus menyediakan DEM atau koreksi medan manual.")
        st.stop()

    xls = pd.ExcelFile(grav)
    all_data=[]

    for sh in xls.sheet_names:
        df = pd.read_excel(grav, sheet_name=sh)

        required={"Nama","Time","G_read (mGal)","Lat","Lon","Elev"}
        if not required.issubset(df.columns):
            st.warning(f"Sheet {sh} dilewati (kolom tidak lengkap)")
            continue

        # Convert to UTM
        E,N,_,_ = latlon_to_utm_redfearn(df["Lat"], df["Lon"])
        df["Easting"], df["Northing"] = E, N

        # Drift correction
        Gsol, drift = compute_drift(df, G_base)
        df["G_read (mGal)"] = df["Nama"].map(Gsol)

        # Basic corrections
        df["Koreksi Lintang"] = latitude_correction(df["Lat"])
        df["Free Air Correction"] = free_air(df["Elev"])
        df["FAA"] = df["G_read (mGal)"] - df["Koreksi Lintang"] + df["Free Air Correction"]

        # ============================================================
        # FLEXIBLE TERRAIN CORRECTION
        # ============================================================

        if km_map is not None:
            # PRIORITAS: gunakan MANUAL
            df["Koreksi Medan"] = df["Nama"].map(km_map)

            missing = df["Koreksi Medan"].isna().sum()
            if missing > 0:
                st.warning(f"{missing} stasiun tidak ada pada koreksi medan manual.")

        elif dem is not None:
            # fallback: pakai DEM
            tc = []
            for i in range(len(df)):
                e0,n0,z0 = df.iloc[i][["Easting","Northing","Elev"]]
                if method.startswith("NAGY"):
                    tc_val = nagy_tc(e0,n0,z0, dem)
                else:
                    tc_val = hammer_tc(e0,n0,z0, dem)
                tc.append(tc_val)
            df["Koreksi Medan"] = tc

        else:
            st.error("Tidak ada sumber koreksi medan.")
            st.stop()

        # Parasnis
        df["X-Parasnis"] = 0.04192*df["Elev"] - df["Koreksi Medan"]
        df["Y-Parasnis"] = df["Free Air Correction"]
        df["Hari"] = sh

        all_data.append(df)

    if not all_data:
        st.error("Tidak ada sheet valid.")
        st.stop()

    df = pd.concat(all_data, ignore_index=True)

    # Slope K
    mask = df[["X-Parasnis","Y-Parasnis"]].notnull().all(axis=1)
    if mask.sum()>=2:
        a,b = np.polyfit(df.loc[mask,"X-Parasnis"], df.loc[mask,"Y-Parasnis"],1)
        K=a
    else:
        K=np.nan

    df["Bouger Correction"] = 0.04192 * K * df["Elev"]
    df["Simple Bouger Anomaly"] = df["FAA"] - df["Bouger Correction"]
    df["Complete Bouger Correction"] = df["Simple Bouger Anomaly"] + df["Koreksi Medan"]

    st.success("Proses selesai!")
    st.dataframe(df.head())

    # ============================================================
    # CONTOUR MAPS
    # ============================================================

    st.subheader("Peta Kontur")

    x = df["Easting"]
    y = df["Northing"]
    gx = np.linspace(x.min(), x.max(), 200)
    gy = np.linspace(y.min(), y.max(), 200)
    GX,GY = np.meshgrid(gx,gy)

    def plot_cont(z, title):
        Z = griddata((x,y), z, (GX,GY), method="cubic")
        fig,ax = plt.subplots(figsize=(8,6))
        c=ax.contourf(GX,GY,Z,40,cmap="jet")
        ax.scatter(x,y,c=z,cmap="jet",s=12,edgecolor="k")
        ax.set_title(title)
        fig.colorbar(c, ax=ax)
        st.pyplot(fig)

    plot_cont(df["Complete Bouger Correction"], "CBA")
    plot_cont(df["Simple Bouger Anomaly"], "SBA")
    plot_cont(df["Elev"], "Elevation")

    # ============================================================
    # DOWNLOAD RESULT
    # ============================================================

    st.subheader("Download Hasil")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "gravcore_output.csv")


