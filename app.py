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

# - Import Plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly tidak terinstall. Interactive plots tidak akan tersedia.")

# Jika ingin mencoba Folium untuk peta interaktif:
try:
    import folium
    from folium.plugins import MarkerCluster, HeatMap
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
   
#Fungsi Hash Untuk Login
def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

#-- Database username & password

USER_DB = {
    "admin": hash_password("admin"),
    "user": hash_password("12345"),
}

# ROLE OPSIONAL
USER_ROLES = {
    "admin": "admin",
    "user": "viewer",
}

#Autentikasi

def authenticate(username, password):
    if username in USER_DB:
        return USER_DB[username] == hash_password(password)
    return False

#Login Page

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
    st.stop()

#Logout Button
def logout_button():
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

##Login Page

def require_login():
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        login_page()
    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    logout_button()

require_login()

#Konversi UTM 

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
    
    easting = k0 * N * (A + (1 - T + C) * A**3 / 6
                        + (5 - 18*T + T*T + 72*C - 58*e2) * A**5 / 120) + 500000
    
    northing = k0 * (M + N * np.tan(lat_rad) * (A**2/2
                     + (5 - T + 9*C + 4*C*C) * A**4/24
                     + (61 - 58*T + T*T + 600*C - 330*e2) * A**6 / 720))
    
    hemi = np.where(lat >= 0, "north", "south")
    northing = np.where(hemi == "south", northing + 10000000, northing)
    
    return easting, northing, zone, hemi

def load_geotiff_without_tfw(file):
    """
    Load GeoTIFF using only TIFF metadata (GeoKeys).
    """
    img = Image.open(file)
    arr = np.array(img, dtype=float)
    meta = img.tag_v2
    
    if 33550 not in meta or 33922 not in meta:
        raise ValueError("GeoTIFF metadata not found. TIFF requires TFW or manual bounding box.")
    
    scaleX, scaleY, _ = meta[33550]
    tiepoint = meta[33922]
    X0 = tiepoint[3]
    Y0 = tiepoint[4]
    
    rows, cols = arr.shape
    X = X0 + np.arange(cols) * scaleX
    Y = Y0 - np.arange(rows) * abs(scaleY)
    XX, YY = np.meshgrid(X, Y)
    
    df = pd.DataFrame({
        "Easting": XX.ravel(),
        "Northing": YY.ravel(),
        "Elev": arr.ravel()
    })
    return df

#DEM LOADER (.xyz, .csv, atau .txt)
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
        E, N, _, _ = latlon_to_utm_redfearn(df["Lat"], df["Lon"])
        return pd.DataFrame({"Easting": E, "Northing": N, "Elev": df["Elev"]})
    
    # TIFF → GeoTIFF metadata
    if name.endswith((".tif", ".tiff")):
        return load_geotiff_without_tfw(file)
    
    raise ValueError("DEM format unsupported. Use: CSV, XYZ, TXT")

## Perhitungan Drift
def compute_drift(df, G_base, debug_mode=False):
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="raise")
    df["G_read (mGal)"] = pd.to_numeric(df["G_read (mGal)"], errors="coerce")
    
    # VALIDASI: Cek duplikat nama dengan koordinat berbeda
    duplicate_check = df.groupby('Nama').agg({
        'Lat': 'nunique',
        'Lon': 'nunique',
        'Elev': 'nunique'
    })
    
    problematic_stations = duplicate_check[
        (duplicate_check['Lat'] > 1) | 
        (duplicate_check['Lon'] > 1) | 
        (duplicate_check['Elev'] > 1)
    ]
    
    if not problematic_stations.empty and debug_mode:
        st.warning(f"⚠️ Found stations with same name but different coordinates: {problematic_stations.index.tolist()}")
    
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

#Koreksi Latitude
def latitude_correction(lat):
    phi = np.radians(lat)
    s = np.sin(phi); s2 = np.sin(2*phi)
    return 978032.67715 * (1 + 0.0053024 * s*s - 0.0000059 * s2*s2)

def free_air(elev):
    return 0.3086 * elev

#Konstanta Untuk Dari Paper OSS

G = 6.67430e-11           # m^3 kg^-1 s^-2
RHO = 2670.0              # kg/m^3 (2.67 g/cm^3)
NANO_TO_MGAL = 1e-6       # 1 nGal = 1e-6 mGal

def terrain_effect_cylindrical_sector(R1, R2, theta1, theta2, z, density):
    """
    PERSAMAAN (1) dari paper: Δg_T = GρΔθ[R2-R1+√(R1²+z²)-√(R2²+z²)]
    """
    theta1_rad = np.radians(theta1)
    theta2_rad = np.radians(theta2)
    Delta_theta = theta2_rad - theta1_rad
    
    term = (R2 - R1) + np.sqrt(R1**2 + z**2) - np.sqrt(R2**2 + z**2)
    delta_g_si = G * density * Delta_theta * term
    
    return delta_g_si * 1e5

def optimized_elevation(z_avg, deviations, R1, R2, Delta_theta, r_points):
    """
    PERSAMAAN (8) dari paper: Optimized elevation z'
    """
    if len(deviations) == 0:
        return z_avg
    
    if R2 == R1 or Delta_theta == 0:
        return z_avg
    
    numerator_sum = 0.0
    denominator_sum = 0.0
    
    for l, r in zip(deviations, r_points):
        if r < 1e-10:
            continue
            
        sign_l = 1.0 if l >= 0 else -1.0
        numerator = 2 * z_avg * l + l**2
        denominator = Delta_theta * 2 * (r**3)
        
        numerator_sum += sign_l * numerator
        denominator_sum += denominator
    
    if denominator_sum == 0:
        return z_avg
    
    factor = (R2 * R1) / (R2 - R1)
    z_prime_sq = z_avg**2 + factor * (numerator_sum / denominator_sum)
    
    z_prime_sq = max(z_prime_sq, 0.0)
    
    return np.sqrt(z_prime_sq)

#Class untuk OSS Koreksi Medan

class OSSTerrainCorrector:
    def __init__(self, dem_df, station_coords, params=None):
        self.e0, self.n0, self.z0 = station_coords
        self.dem_df = dem_df.copy()
        
        self.params = {
            'max_radius': 5000.0,
            'tolerance_nGal': 1.0,
            'threshold_mGal': 0.1,
            'theta_step': 1.0,
            'r_step_near': 10.0,
            'r_step_far': 50.0,
            'min_points_per_sector': 15,
            'use_optimized_elevation': True,
            'debug': False,
            'density': RHO
        }
        
        if params:
            self.params.update(params)
        
        
        #Definisi z yang BENAR
        # z = elevasi_DEM - elevasi_stasiun
      
        dx = self.dem_df['Easting'] - self.e0
        dy = self.dem_df['Northing'] - self.n0
        
        self.r = np.sqrt(dx**2 + dy**2)
        self.theta_rad = np.arctan2(dy, dx)
        self.theta_deg = np.degrees(self.theta_rad) % 360.0
        
        # z = h_dem - h_station
        self.z_rel = self.dem_df['Elev'] - self.z0
        
        if self.params.get('debug', False):
            st.write(f"🚨 DEBUG: Station elevation z0 = {self.z0:.1f} m")
            st.write(f"🚨 DEBUG: z = h_dem - h_station = {self.z_rel.min():.1f} to {self.z_rel.max():.1f} m")
            st.write(f"🚨 DEBUG: Mean z = {self.z_rel.mean():.1f} m")
            st.write(f"🚨 DEBUG: Points with z > 0 (DEM higher): {(self.z_rel > 0).sum()}")
            st.write(f"🚨 DEBUG: Points with z < 0 (DEM lower): {(self.z_rel < 0).sum()}")
        
        mask = self.r <= self.params['max_radius']
        self.r_filtered = self.r[mask].values
        self.theta_filtered = self.theta_deg[mask].values
        self.z_rel_filtered = self.z_rel[mask].values
        
        if self.params.get('debug', False):
            st.write(f"🚨 DEBUG: Points within radius: {len(self.r_filtered)}")
    
    def _sector_effect(self, R1, R2, theta1, theta2, z_avg):
        """Wrapper untuk terrain_effect_cylindrical_sector"""
        return terrain_effect_cylindrical_sector(
            R1, R2, theta1, theta2, z_avg, self.params['density']
        )
    
    def _find_turning_points_theta(self, theta1, theta2, R1, R2):
        """Find turning points in angular direction - DIPERBAIKI"""
        theta_step = self.params['theta_step']
        tolerance = self.params['tolerance_nGal'] * NANO_TO_MGAL
        
        angles = []
        terrain_values = []
        
        theta_current = theta1
        while theta_current <= theta2:
            mask_left = (self.theta_filtered >= theta1) & (self.theta_filtered <= theta_current) & \
                       (self.r_filtered >= R1) & (self.r_filtered <= R2)
            
            mask_right = (self.theta_filtered >= theta_current) & (self.theta_filtered <= theta2) & \
                        (self.r_filtered >= R1) & (self.r_filtered <= R2)
            
            terrain_total = 0.0
            
            if np.any(mask_left):
                z_avg_left = np.mean(self.z_rel_filtered[mask_left])
                terrain_left = self._sector_effect(R1, R2, theta1, theta_current, z_avg_left)
                terrain_total += terrain_left
            
            if np.any(mask_right):
                z_avg_right = np.mean(self.z_rel_filtered[mask_right])
                terrain_right = self._sector_effect(R1, R2, theta_current, theta2, z_avg_right)
                terrain_total += terrain_right
            
            angles.append(theta_current)
            terrain_values.append(terrain_total)
            theta_current += theta_step
        
        turning_points = []
        if len(terrain_values) > 2:
            for i in range(1, len(terrain_values)-1):
                second_deriv = terrain_values[i+1] - 2*terrain_values[i] + terrain_values[i-1]
                if abs(second_deriv) > tolerance:
                    turning_points.append(angles[i])
        
        return turning_points
    
    def _find_turning_points_radius(self, theta1, theta2, R1, R2):
        """Find turning points in radial direction - DIPERBAIKI"""
        r_step = self.params['r_step_near'] if R2 < 1000 else self.params['r_step_far']
        tolerance = self.params['tolerance_nGal'] * NANO_TO_MGAL
        
        radii = []
        terrain_values = []
        
        R_current = R1
        while R_current <= R2:
            mask_inner = (self.r_filtered >= R1) & (self.r_filtered <= R_current) & \
                        (self.theta_filtered >= theta1) & (self.theta_filtered <= theta2)
            
            mask_outer = (self.r_filtered >= R_current) & (self.r_filtered <= R2) & \
                        (self.theta_filtered >= theta1) & (self.theta_filtered <= theta2)
            
            terrain_total = 0.0
            
            if np.any(mask_inner):
                z_avg_inner = np.mean(self.z_rel_filtered[mask_inner])
                terrain_inner = self._sector_effect(R1, R_current, theta1, theta2, z_avg_inner)
                terrain_total += terrain_inner
            
            if np.any(mask_outer):
                z_avg_outer = np.mean(self.z_rel_filtered[mask_outer])
                terrain_outer = self._sector_effect(R_current, R2, theta1, theta2, z_avg_outer)
                terrain_total += terrain_outer
            
            radii.append(R_current)
            terrain_values.append(terrain_total)
            R_current += r_step
        
        turning_points = []
        if len(terrain_values) > 2:
            for i in range(1, len(terrain_values)-1):
                second_deriv = terrain_values[i+1] - 2*terrain_values[i] + terrain_values[i-1]
                if abs(second_deriv) > tolerance:
                    turning_points.append(radii[i])
        
        return turning_points
    
    def _process_sector(self, theta1, theta2, R1, R2, depth=0):
        """Recursive sector processing - DIPERBAIKI"""
        threshold = self.params['threshold_mGal']
        
        debug = self.params.get('debug', False)
        
        mask = (self.theta_filtered >= theta1) & (self.theta_filtered <= theta2) & \
               (self.r_filtered >= R1) & (self.r_filtered <= R2)
        
        if not np.any(mask):
            return 0.0, []
        
        z_sector = self.z_rel_filtered[mask]
        r_sector = self.r_filtered[mask]
        
        if len(z_sector) < self.params['min_points_per_sector']:
            z_avg = np.mean(z_sector) if len(z_sector) > 0 else 0.0
            terrain = self._sector_effect(R1, R2, theta1, theta2, z_avg)
            return terrain, [(theta1, theta2, R1, R2, z_avg, terrain)]
        
        z_avg = np.mean(z_sector)
        terrain_avg = self._sector_effect(R1, R2, theta1, theta2, z_avg)
        
        if debug and depth <= 2:
            st.write(f"  {'  '*depth}θ=[{theta1:.1f},{theta2:.1f}], r=[{R1:.0f},{R2:.0f}], "
                    f"n={len(z_sector)}, z_avg={z_avg:.1f}, Δg={terrain_avg:.6f}")
        
        if abs(terrain_avg) < threshold:
            if self.params['use_optimized_elevation']:
                Delta_theta = np.radians(theta2 - theta1)
                deviations = z_sector - z_avg
                z_opt = optimized_elevation(z_avg, deviations, R1, R2, Delta_theta, r_sector)
                terrain_final = self._sector_effect(R1, R2, theta1, theta2, z_opt)
            else:
                terrain_final = terrain_avg
                z_opt = z_avg
            
            return terrain_final, [(theta1, theta2, R1, R2, z_opt, terrain_final)]
        
        if debug and depth <= 2:
            st.write(f"  {'  '*depth}  Δg={terrain_avg:.6f} > threshold={threshold}, subdividing...")
        
        turning_points_theta = self._find_turning_points_theta(theta1, theta2, R1, R2)
        
        if turning_points_theta:
            if debug and depth <= 2:
                st.write(f"  {'  '*depth}  Found {len(turning_points_theta)} angular turning points")
            
            total_terrain = 0.0
            all_subsectors = []
            angles = [theta1] + sorted(turning_points_theta) + [theta2]
            
            for i in range(len(angles)-1):
                sub_theta1, sub_theta2 = angles[i], angles[i+1]
                sub_terrain, sub_subsectors = self._process_sector(
                    sub_theta1, sub_theta2, R1, R2, depth+1
                )
                total_terrain += sub_terrain
                all_subsectors.extend(sub_subsectors)
            
            return total_terrain, all_subsectors
        
        turning_points_r = self._find_turning_points_radius(theta1, theta2, R1, R2)
        
        if turning_points_r:
            if debug and depth <= 2:
                st.write(f"  {'  '*depth}  Found {len(turning_points_r)} radial turning points")
            
            total_terrain = 0.0
            all_subsectors = []
            radii = [R1] + sorted(turning_points_r) + [R2]
            
            for i in range(len(radii)-1):
                sub_R1, sub_R2 = radii[i], radii[i+1]
                sub_terrain, sub_subsectors = self._process_sector(
                    theta1, theta2, sub_R1, sub_R2, depth+1
                )
                total_terrain += sub_terrain
                all_subsectors.extend(sub_subsectors)
            
            return total_terrain, all_subsectors
        
        if self.params['use_optimized_elevation']:
            Delta_theta = np.radians(theta2 - theta1)
            deviations = z_sector - z_avg
            z_opt = optimized_elevation(z_avg, deviations, R1, R2, Delta_theta, r_sector)
            terrain_final = self._sector_effect(R1, R2, theta1, theta2, z_opt)
        else:
            terrain_final = terrain_avg
            z_opt = z_avg
        
        return terrain_final, [(theta1, theta2, R1, R2, z_opt, terrain_final)]
    
    def calculate_terrain_correction(self):
        """Main method to calculate terrain correction"""
        if self.params.get('debug', False):
            st.write(f"Starting OSS calculation for station at ({self.e0:.1f}, {self.n0:.1f})")
        
        total_tc, subsectors = self._process_sector(
            theta1=0.0,
            theta2=360.0,
            R1=0.0,
            R2=self.params['max_radius']
        )
        
        # Validasi: TC harus positif
        if total_tc < 0:
            if self.params.get('debug', False):
                st.warning(f"Negative TC ({total_tc:.3f} mGal). Taking absolute value.")
            total_tc = abs(total_tc)
        
        if self.params.get('debug', False):
            st.write(f"Total TC: {total_tc:.3f} mGal")
            if subsectors:
                st.write(f"Number of subsectors: {len(subsectors)}")
        
        return total_tc, subsectors

def calculate_oss_correction(dem_df, station_row, params=None):
    """
    Wrapper function dengan koreksi yang benar
    """
    station_coords = (
        float(station_row['Easting']),
        float(station_row['Northing']),
        float(station_row['Elev'])
    )
    
    corrector = OSSTerrainCorrector(dem_df, station_coords, params)
    tc_value, _ = corrector.calculate_terrain_correction()
    
    return tc_value

# Fungsi Untuk Validasi dan Plotting

def validate_tc_value(tc_value, station_name, debug=False):
    """Validasi nilai TC yang reasonable"""
    validated_tc = tc_value
    
    if tc_value < -0.5:
        if debug:
            st.warning(f"Station {station_name}: TC bernilai negatif ({tc_value:.3f} mGal)")
        validated_tc = max(tc_value, 0.0)
    
    elif tc_value > 50.0:
        if debug:
            st.warning(f"Station {station_name}: TC terlalu besar ({tc_value:.1f} mGal)")
        validated_tc = min(tc_value, 100.0)
    
    elif 0 <= tc_value < 0.01:
        if debug:
            st.warning(f"Station {station_name}: TC sangat kecil ({tc_value:.6f} mGal)")
    
    return validated_tc

def plot_dem_elevation(dem_df, stations_df=None):

    x_dem = dem_df["Easting"]
    y_dem = dem_df["Northing"]
    z_dem = dem_df["Elev"]
    
    xi = np.linspace(x_dem.min(), x_dem.max(), 200)
    yi = np.linspace(y_dem.min(), y_dem.max(), 200)
    XI, YI = np.meshgrid(xi, yi)
    
    ZI = griddata((x_dem, y_dem), z_dem, (XI, YI), method='cubic')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(XI, YI, ZI, 40, cmap='terrain', alpha=0.8)
    
    if stations_df is not None:
        ax.scatter(stations_df['Easting'], stations_df['Northing'], 
                  c='red', s=50, marker='^', edgecolor='black',
                  label='Gravity Stations', zorder=5)
    
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_title('Topografi DEM dan Stasiun Pengukuran')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Elevasi (m)')
    
    if stations_df is not None:
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def plot_cont(x, y, z, title):
    """Plot contour dari data"""
    gx = np.linspace(x.min(), x.max(), 200)
    gy = np.linspace(y.min(), y.max(), 200)
    GX, GY = np.meshgrid(gx, gy)
    
    Z = griddata((x, y), z, (GX, GY), method="cubic")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(GX, GY, Z, 40, cmap="jet")
    ax.scatter(x, y, c=z, cmap="jet", s=12, edgecolor="k")
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_title(title)
    fig.colorbar(cf, ax=ax, label='Value (mGal)')
    
    return fig

def create_interactive_scatter(df, x_col, y_col, color_col, title, hover_cols=None):
    """Buat scatter plot interaktif dengan Plotly"""
    if not PLOTLY_AVAILABLE:
        return None
    
    if hover_cols is None:
        hover_cols = ['Nama', 'Elev', 'Lon', 'Lat']
    
    hover_data = {col: df[col] for col in hover_cols if col in df.columns}
    
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_data=hover_data,
        title=title,
        color_continuous_scale='viridis',
        size_max=15
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode='closest',
        showlegend=True
    )
    
    return fig


# TEST FUNCTION UNTUK VALIDASI

def run_oss_test():
    """Test OSS dengan data sederhana untuk validasi"""
    st.subheader("OSS Algorithm Test")
    
    # Buat data test
    test_dem_data = {
        'Easting': [0, 50, 50, 0, -50, -50, 0, 35, -35, 25, -25],
        'Northing': [50, 0, -50, -50, 0, 50, 0, 35, -35, -25, 25],
        'Elev': [200, 150, 50, 80, 120, 180, 100, 130, 70, 90, 110]
    }
    
    test_dem_df = pd.DataFrame(test_dem_data)
    
    test_stations = [
        {'Nama': 'TEST_CENTER', 'Easting': 0, 'Northing': 0, 'Elev': 100, 'Lat': 0, 'Lon': 0},
        {'Nama': 'TEST_HILL', 'Easting': 30, 'Northing': 30, 'Elev': 150, 'Lat': 0, 'Lon': 0},
        {'Nama': 'TEST_VALLEY', 'Easting': -30, 'Northing': -30, 'Elev': 50, 'Lat': 0, 'Lon': 0}
    ]
    
    results = []
    
    for station in test_stations:
        for th in [0.001, 0.01, 0.05, 0.1, 0.5]:
            params = {
                'threshold_mGal': th,
                'debug': False,
                'max_radius': 100,
                'density': RHO
            }
            
            tc = calculate_oss_correction(test_dem_df, pd.Series(station), params)
            
            results.append({
                'Station': station['Nama'],
                'Threshold': th,
                'TC': tc,
                'Elevation': station['Elev']
            })
    
    results_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for station in results_df['Station'].unique():
        station_data = results_df[results_df['Station'] == station]
        axes[0].plot(station_data['Threshold'], station_data['TC'], 
                    'o-', label=station, linewidth=2)
    
    axes[0].set_xlabel('Threshold (mGal)')
    axes[0].set_ylabel('Terrain Correction (mGal)')
    axes[0].set_title('TC vs Threshold (Should have plateau)')
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    scatter = axes[1].scatter(results_df['Elevation'], results_df['TC'], 
                             c=results_df['Threshold'], cmap='viridis', s=100)
    axes[1].set_xlabel('Elevasi Stasiun (m)')
    axes[1].set_ylabel('Koreksi Medan (mGal)')
    axes[1].set_title('Koreksi medan vs Elevasi (Should correlate)')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1], label='Threshold (mGal)')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("**Expected Results:**")
    st.write("1. **Left plot:** TC should stabilize (plateau) as threshold changes")
    st.write("2. **Right plot:** TC should correlate with station elevation")
    st.write("3. **TC values:** Should be positive and realistic (0-20 mGal for this test)")
    
    return results_df

# ============================================================
# UI STREAMLIT
# ============================================================
st.markdown(
    f"""
    <div style="display:flex; align-items:center;">
        <img src="https://raw.githubusercontent.com/dzakyw/AutoGrav/main/logo esdm.png" style="width:200px; margin-right:5px;">
        <div>
            <h2 style="margin-bottom:0;">Auto Grav - Semua Terasa Cepat</h2>
            <p style="color:red; font-weight:bold;">OSS Algorithm Corrected Version</p>
        </div>
    </div>
    <hr>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("Input Files")
grav = st.sidebar.file_uploader("Input Gravity Multi-Sheets (.xlsx)", type=["xlsx"])
demf = st.sidebar.file_uploader("Upload DEM (CSV/XYZ/TXT)", type=["csv","txt","xyz"])
kmf = st.sidebar.file_uploader("Koreksi Medan manual (optional)", type=["csv","xlsx"])
G_base = st.sidebar.number_input("G Absolute di Base", value=0.0)

# Test button
if st.sidebar.button("Run OSS Test"):
    run_oss_test()

st.sidebar.subheader("Interactive Plot Options")
enable_interactive = st.sidebar.checkbox("Enable Interactive Plots", value=True)


# PARAMETER OSS

st.sidebar.subheader("OSS Algorithm Parameters")
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
st.session_state.debug_mode = debug_mode

threshold_mgal = st.sidebar.slider(
    "Threshold (mGal) for subdivision",
    min_value=0.001,
    max_value=0.1,
    value=0.05,
    step=0.001,
    help="Start with 0.05 mGal for optimal results"
)

max_radius = st.sidebar.number_input(
    "Maximum Radius (m)",
    value=4500,
    step=100,
    help="Jangkauan maksimum untuk koreksi medan"
)

density = st.sidebar.number_input(
    "Densitas Batuan (kg/m³)",
    value=2670.0,
    step=10.0,
    format="%.1f",
    help="Densitas untuk perhitungan terrain correction"
)

use_optimized_elev = st.sidebar.checkbox(
    "Use optimized elevation (z')", 
    value=True,
    help="Menggunakan persamaan (8) untuk optimasi z'"
)

min_points_sector = st.sidebar.number_input(
    "Minimum points per sector",
    min_value=1,
    max_value=70,
    value=10,
    help="Sector dengan points < ini tidak di-subdivide"
)

run = st.sidebar.button("Run Processing", type="primary")

st.sidebar.subheader("Contoh File Input")
st.sidebar.write("[Contoh Data Input Gravity](https://github.com/dzakyw/AutoGrav/raw/9bb43e1559c823350f2371360309d84eaab5ea38/sample_gravity.xlsx)")
st.sidebar.write("[Contoh DEM dengan format .txt](https://github.com/dzakyw/AutoGrav/raw/9bb43e1559c823350f2371360309d84eaab5ea38/sample_dem.csv)")


# MAIN PROCESSING

if run:
    if grav is None:
        st.error("Upload file gravity .xlsx (multi-sheet).")
        st.stop()
    
    dem = None
    if demf:
        try:
            dem = load_dem(demf)
            st.success(f"DEM loaded: {len(dem):,} points.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**DEM Statistics:**")
                st.write(f"- Points: {len(dem):,}")
                st.write(f"- Elevation range: {dem['Elev'].min():.1f} to {dem['Elev'].max():.1f} m")
                st.write(f"- Mean elevation: {dem['Elev'].mean():.1f} m")
            
            with col2:
                st.info(f"**Spatial Coverage:**")
                st.write(f"- Easting: {dem['Easting'].min():.0f} to {dem['Easting'].max():.0f} m")
                st.write(f"- Northing: {dem['Northing'].min():.0f} to {dem['Northing'].max():.0f} m")
        
        except Exception as e:
            st.error(f"DEM load failed: {e}")
            st.stop()
    
    km_map = None
    if kmf:
        try:
            km = pd.read_csv(kmf)
        except:
            km = pd.read_excel(kmf)
        if {"Nama","Koreksi_Medan"}.issubset(km.columns):
            km["Koreksi_Medan"] = pd.to_numeric(km["Koreksi_Medan"], errors="coerce")
            km_map = km.set_index("Nama")["Koreksi_Medan"].to_dict()
            st.info(f"Manual terrain correction loaded: {len(km_map)} stations")
        else:
            st.warning("File koreksi medan manual harus kolom: Nama, Koreksi_Medan. Ignored.")
    
    if (km_map is None) and (dem is None):
        st.error("Anda harus upload DEM atau file koreksi medan manual.")
        st.stop()
    
    try:
        xls = pd.ExcelFile(grav)
    except Exception as e:
        st.error(f"Gagal baca Excel gravitasi: {e}")
        st.stop()
    
    all_dfs = []
    t0 = time.time()
    
    tc_stats = []
    station_details = []
    
    oss_params = {
        'max_radius': max_radius,
        'tolerance_nGal': 1.0,
        'threshold_mGal': threshold_mgal,
        'theta_step': 1.0,
        'r_step_near': 10.0,
        'r_step_far': 50.0,
        'min_points_per_sector': min_points_sector,
        'use_optimized_elevation': use_optimized_elev,
        'debug': debug_mode,
        'density': density
    }
    
    st.info(f"**OSS Parameters:** Max radius = {max_radius} m, Threshold = {threshold_mgal} mGal, Density = {density} kg/m³")
    
    total_sheets = len(xls.sheet_names)
    sheet_progress_bar = st.progress(0)
    
    for sheet_idx, sh in enumerate(xls.sheet_names):
        df = pd.read_excel(grav, sheet_name=sh)
        required = {"Nama","Time","G_read (mGal)","Lat","Lon","Elev"}
        
        if not required.issubset(set(df.columns)):
            st.warning(f"Sheet {sh} dilewati (kolom tidak lengkap).")
            continue
        
        sheet_progress = (sheet_idx + 1) / total_sheets
        sheet_progress_bar.progress(sheet_progress)
        
        E, N, _, _ = latlon_to_utm_redfearn(df["Lat"].to_numpy(), df["Lon"].to_numpy())
        df["Easting"] = E
        df["Northing"] = N
        
        Gmap, D = compute_drift(df, G_base, debug_mode)
        df["G_read (mGal)"] = df["Nama"].map(Gmap)
        
        df["Koreksi Lintang"] = latitude_correction(df["Lat"])
        df["Free Air Correction"] = free_air(df["Elev"])
        df["FAA"] = df["G_read (mGal)"] - df["Koreksi Lintang"] + df["Free Air Correction"]
        
        tc_list = []
        nstations = len(df)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(nstations):
            station_data = df.iloc[i]
            station_name = station_data['Nama']
            
            progress = (i + 1) / nstations
            progress_bar.progress(progress)
            status_text.text(f"Sheet {sh}: Station {i+1}/{nstations} ({station_name})")
            
            if km_map is not None and station_name in km_map:
                tc_val = km_map[station_name]
                if debug_mode:
                    st.write(f"{station_name}: Using manual TC = {tc_val:.3f} mGal")
            
            elif dem is not None:
                tc_val = calculate_oss_correction(dem, station_data, oss_params)
                tc_val = validate_tc_value(tc_val, station_name, debug_mode)
                
                if debug_mode:
                    st.write(f"{station_name}: OSS TC = {tc_val:.3f} mGal")
            else:
                tc_val = 0.0
                if debug_mode:
                    st.write(f"{station_name}: No DEM or manual TC available, using 0.0 mGal")
            
            station_details.append({
                'Sheet': sh,
                'Station': station_name,
                'Lon': station_data['Lon'],
                'Lat': station_data['Lat'],
                'Easting': station_data['Easting'],
                'Northing': station_data['Northing'],
                'Elevation': station_data['Elev'],
                'TC_OSS': tc_val,
                'Source': 'Manual' if (km_map is not None and station_name in km_map) else 'OSS'
            })
            
            tc_stats.append(tc_val)
            tc_list.append(tc_val)
        
        progress_bar.empty()
        status_text.empty()
        
        if debug_mode and tc_list:
            tc_array = np.array(tc_list)
            with st.expander(f"TC Statistics for Sheet {sh}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean", f"{tc_array.mean():.3f} mGal")
                with col2:
                    st.metric("Min", f"{tc_array.min():.3f} mGal")
                with col3:
                    st.metric("Max", f"{tc_array.max():.3f} mGal")
        
        df["Koreksi Medan"] = tc_list
        df["X-Parasnis"] = 0.04192 * df["Elev"] - df["Koreksi Medan"]
        df["Y-Parasnis"] = df["Free Air Correction"]
        df["Hari"] = sh
        
        all_dfs.append(df)
    
    sheet_progress_bar.empty()
    
    if len(all_dfs) == 0:
        st.error("No valid sheets processed.")
        st.stop()
    
    df_all = pd.concat(all_dfs, ignore_index=True)
    elapsed = time.time() - t0
    
    st.success(f"Processing completed in {elapsed:.1f} seconds")
    
    if tc_stats:
        tc_values = np.array(tc_stats)
        st.subheader("Terrain Correction Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean TC", f"{tc_values.mean():.3f} mGal")
        with col2:
            st.metric("Median TC", f"{np.median(tc_values):.3f} mGal")
        with col3:
            st.metric("Min TC", f"{tc_values.min():.3f} mGal")
        with col4:
            st.metric("Max TC", f"{tc_values.max():.3f} mGal")
        
        fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
        ax_hist.hist(tc_values, bins=30, alpha=0.7, edgecolor='black')
        ax_hist.set_xlabel('Terrain Correction (mGal)')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title('Distribution of Terrain Correction Values')
        ax_hist.grid(True, alpha=0.3)
        st.pyplot(fig_hist)
    
    mask = df_all[["X-Parasnis","Y-Parasnis"]].notnull().all(axis=1)
    if mask.sum() >= 2:
        slope, intercept = np.polyfit(df_all.loc[mask,"X-Parasnis"], df_all.loc[mask,"Y-Parasnis"], 1)
        
        y_pred = slope * df_all.loc[mask,"X-Parasnis"] + intercept
        y_actual = df_all.loc[mask,"Y-Parasnis"]
        r_squared = 1 - np.sum((y_actual - y_pred)**2) / np.sum((y_actual - np.mean(y_actual))**2)
        
        st.info(f"**Parasnis Regression:** Slope (K) = {slope:.5f}, R² = {r_squared:.3f}")
        
        if r_squared < 0.7:
            st.warning("Low R² value in Parasnis regression! Check TC calculations.")
    else:
        slope = np.nan
        st.warning("Not enough data for Parasnis regression.")
    
    df_all["Bouger Correction"] = 0.04192 * slope * df_all["Elev"]
    df_all["Simple Bouger Anomaly"] = df_all["FAA"] - df_all["Bouger Correction"]
    df_all["Complete Bouger Anomaly"] = df_all["Simple Bouger Anomaly"] + df_all["Koreksi Medan"]
    
    st.subheader("Hasil Yang Diproses")
    
    with st.expander("Lihat Data Preview", expanded=True):
        st.dataframe(df_all.head(20))
        st.write(f"**Total rows processed:** {len(df_all)}")
        st.write(f"**Total sheets processed:** {len(all_dfs)}")
    
    st.subheader("Visualisasi Hasil")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Parasnis Plot", "Topography", "CBA Map", "Data Export"])
    
    with tab1:
        if mask.sum() >= 2:
            X = df_all.loc[mask, "X-Parasnis"].values
            Y = df_all.loc[mask, "Y-Parasnis"].values
            
            fig_parasnis, ax_parasnis = plt.subplots(figsize=(8, 6))
            ax_parasnis.scatter(X, Y, s=25, color="blue", label="Data Parasnis", alpha=0.7)
            
            X_line = np.linspace(min(X), max(X), 100)
            Y_line = slope * X_line + intercept
            ax_parasnis.plot(X_line, Y_line, color="red", linewidth=2,
                          label=f"Regresi: Y = {slope:.5f} X + {intercept:.5f}")
            
            ax_parasnis.set_xlabel("X-Parasnis (mGal)")
            ax_parasnis.set_ylabel("Y-Parasnis (mGal)")
            ax_parasnis.set_title("Diagram Parasnis (X vs Y)")
            ax_parasnis.grid(True, linestyle="--", alpha=0.5)
            ax_parasnis.legend()
            st.pyplot(fig_parasnis)
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Slope (K):** {slope:.5f}")
            with col2:
                st.info(f"**R-squared:** {r_squared:.3f}")
        else:
            st.warning("Not enough data for Parasnis plot.")
    
    with tab2:
        if dem is not None:
            fig_topo = plot_dem_elevation(dem, df_all)
            st.pyplot(fig_topo)
        else:
            st.info("No DEM available for topography plot.")
    
    with tab3:
        if len(df_all) > 0:
            fig_cba = plot_cont(df_all["Easting"], df_all["Northing"], 
                               df_all["Complete Bouger Anomaly"], 
                               "Complete Bouguer Anomaly")
            st.pyplot(fig_cba)
        else:
            st.warning("No data available for CBA plot.")
    
    with tab4:
        st.subheader("Download Hasil Perhitungan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df_all.to_csv(index=False)
            st.download_button(
                label="Download Processed Data (.csv)",
                data=csv.encode('utf-8'),
                file_name="autograv_results.csv",
                mime="text/csv"
            )
        
        with col2:
            if station_details:
                details_df = pd.DataFrame(station_details)
                details_csv = details_df.to_csv(index=False)
                st.download_button(
                    label="Download Station Details (.csv)",
                    data=details_csv.encode('utf-8'),
                    file_name="autograv_station_details.csv",
                    mime="text/csv"
                )
        
        st.info("Processing Sudah Selesai, Download data hasil")



