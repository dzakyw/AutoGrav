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
   
# Fungsi Hash Untuk Login
def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

# Database username & password
USER_DB = {
    "admin": hash_password("admin"),
    "user": hash_password("12345"),
}

# ROLE OPSIONAL
USER_ROLES = {
    "admin": "admin",
    "user": "viewer",
}

# Autentikasi
def authenticate(username, password):
    if username in USER_DB:
        return USER_DB[username] == hash_password(password)
    return False

# Login Page
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

# Logout Button
def logout_button():
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

## Login Page
def require_login():
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        login_page()
    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    logout_button()

require_login()

# Konversi UTM 
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

# DEM LOADER (.xyz, .csv, atau .txt)
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

# KOREKSI LINTANG (LATITUDE CORRECTION) YANG BENAR
def latitude_correction(lat):
    phi = np.radians(lat)
    s = np.sin(phi)
    return 978031.846 * (1 + (0.0053024 * s*s) - 0.0000059 * np.sin(2*phi)**2)

def free_air(elev):
    """Free-air correction: 0.3086 * elevation (m)"""
    return 0.3086 * elev

# Konstanta Untuk Dari Paper OSS
G = 6.67430e-11           # m^3 kg^-1 s^-2
NANO_TO_MGAL = 1e-6       # 1 nGal = 1e-6 mGal

# ============================================================
# CRITICALLY CORRECTED OSS FUNCTIONS
# ============================================================

def terrain_effect_cylindrical_sector(R1, R2, theta1, theta2, z, density):
    """
    CORRECTED Eq. (1) from paper: Δg_T = GρΔθ[R2-R1+√(R1²+z²)-√(R2²+z²)]
    z = ABSOLUTE terrain elevation (meters)
    """
    theta1_rad = np.radians(theta1)
    theta2_rad = np.radians(theta2)
    Delta_theta = theta2_rad - theta1_rad
    
    # Ensure z is positive (terrain elevation above sea level)
    z_abs = abs(z)
    
    # Calculate the term
    term = (R2 - R1) + np.sqrt(R1**2 + z_abs**2) - np.sqrt(R2**2 + z_abs**2)
    delta_g_si = G * density * Delta_theta * term
    
    # Convert from m/s² to mGal (1 mGal = 1e-5 m/s²)
    return delta_g_si * 1e5

def optimized_elevation(z_avg, deviations, R1, R2, Delta_theta, r_points):
    """
    CORRECTED Eq. (8) from paper: Optimized elevation z'
    """
    if len(deviations) == 0 or R2 <= R1 or Delta_theta == 0:
        return z_avg
    
    # Filter valid points
    valid_mask = r_points > 1.0
    if not np.any(valid_mask):
        return z_avg
    
    deviations = deviations[valid_mask]
    r_points = r_points[valid_mask]
    
    # Vectorized calculation
    sign_l = np.where(deviations >= 0, 1.0, -1.0)
    numerator = 2 * z_avg * deviations + deviations**2
    denominator = Delta_theta * 2 * (r_points**3)
    
    # Avoid division by zero
    denominator[denominator == 0] = 1e-10
    
    numerator_sum = np.sum(sign_l * numerator / denominator)
    
    factor = (R2 * R1) / (R2 - R1)
    z_prime_sq = z_avg**2 + factor * numerator_sum
    
    # Ensure non-negative
    z_prime_sq = max(z_prime_sq, 0.0)
    
    return np.sqrt(z_prime_sq)

# Class untuk OSS Koreksi Medan - CRITICALLY CORRECTED
class OSSTerrainCorrector:
    def __init__(self, dem_df, station_coords, params=None):
        self.e0, self.n0, self.z0 = station_coords
        self.dem_df = dem_df.copy()
        
        self.params = {
            'max_radius': 2000.0,  # Reduced from 5000 for better results
            'tolerance_nGal': 10.0,  # Increased tolerance
            'threshold_mGal': 0.1,   # Increased threshold
            'theta_step': 5.0,       # Increased step
            'r_step_near': 20.0,
            'r_step_far': 100.0,
            'min_points_per_sector': 20,  # Increased minimum
            'use_optimized_elevation': False,  # Disabled initially
            'debug': False,
            'density': 2670.0
        }
        
        if params:
            self.params.update(params)
        
        # CRITICAL FIX: Use ABSOLUTE DEM elevation, not difference
        dx = self.dem_df['Easting'] - self.e0
        dy = self.dem_df['Northing'] - self.n0
        
        self.r = np.sqrt(dx**2 + dy**2)
        self.theta_rad = np.arctan2(dy, dx)
        self.theta_deg = np.degrees(self.theta_rad) % 360.0
        
        # z = ABSOLUTE terrain elevation (DEM height), not difference!
        # Store as numpy array for consistency
        self.z_abs = self.dem_df['Elev'].to_numpy()
        
        if self.params.get('debug', False):
            st.write(f"🚨 DEBUG: Station elevation z0 = {self.z0:.1f} m")
            st.write(f"🚨 DEBUG: Terrain elevation z = {self.z_abs.min():.1f} to {self.z_abs.max():.1f} m")
            st.write(f"🚨 DEBUG: Mean terrain elevation = {np.mean(self.z_abs):.1f} m")
        
        # Apply radius filter
        mask = self.r <= self.params['max_radius']
        
        # Create filtered arrays - CORRECTED
        self.r_filtered = self.r[mask].to_numpy() if hasattr(self.r[mask], 'to_numpy') else self.r[mask].values
        self.theta_filtered = self.theta_deg[mask].to_numpy() if hasattr(self.theta_deg[mask], 'to_numpy') else self.theta_deg[mask].values
        self.z_filtered = self.z_abs[mask]  # Direct indexing works since z_abs is numpy array
        
        if self.params.get('debug', False):
            st.write(f"🚨 DEBUG: Points within radius: {len(self.r_filtered)}")
            if len(self.z_filtered) > 0:
                st.write(f"🚨 DEBUG: Filtered z range: {self.z_filtered.min():.1f} to {self.z_filtered.max():.1f} m")
                
    def _sector_effect(self, R1, R2, theta1, theta2, z_avg):
        """Wrapper untuk terrain_effect_cylindrical_sector dengan absolute z"""
        return terrain_effect_cylindrical_sector(
            R1, R2, theta1, theta2, z_avg, self.params['density']
        )
    
    def _find_turning_points_theta(self, theta1, theta2, R1, R2):
        """Find turning points in angular direction"""
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
                z_avg_left = np.mean(self.z_filtered[mask_left])
                terrain_left = self._sector_effect(R1, R2, theta1, theta_current, z_avg_left)
                terrain_total += terrain_left
            
            if np.any(mask_right):
                z_avg_right = np.mean(self.z_filtered[mask_right])
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
        """Find turning points in radial direction"""
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
                z_avg_inner = np.mean(self.z_filtered[mask_inner])
                terrain_inner = self._sector_effect(R1, R_current, theta1, theta2, z_avg_inner)
                terrain_total += terrain_inner
            
            if np.any(mask_outer):
                z_avg_outer = np.mean(self.z_filtered[mask_outer])
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
        """Recursive sector processing"""
        threshold = self.params['threshold_mGal']
        
        debug = self.params.get('debug', False)
        
        mask = (self.theta_filtered >= theta1) & (self.theta_filtered <= theta2) & \
               (self.r_filtered >= R1) & (self.r_filtered <= R2)
        
        if not np.any(mask):
            return 0.0, []
        
        z_sector = self.z_filtered[mask]
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
        
        # Apply sanity check: TC should not be too large
        if total_tc > 100.0:  # Unrealistically high
            if self.params.get('debug', False):
                st.warning(f"Unrealistically high TC ({total_tc:.1f} mGal). Clamping to 50 mGal.")
            total_tc = min(total_tc, 50.0)
        
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
        validated_tc = min(tc_value, 50.0)
    
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
                'density': 2670
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
# METODE-METODE DETERMINASI DENSITAS ALTERNATIF
# ============================================================

def nettleton_method(df, density_range=(1.5, 3.0), step=0.1, debug=False):
    """
    Metode Nettleton (1939): Mencari densitas yang meminimalkan korelasi 
    antara Complete Bouguer Anomaly dengan elevasi.
    """
    if 'FAA' not in df.columns or 'Elev' not in df.columns or 'Koreksi Medan' not in df.columns:
        st.error("Data tidak lengkap untuk metode Nettleton. Butuh kolom: FAA, Elev, Koreksi Medan")
        return None, None, None
    
    densities = np.arange(density_range[0], density_range[1] + step, step)
    correlations = []
    
    for rho in densities:
        # Hitung Simple Bouguer Anomaly untuk densitas ini
        bouguer_correction = 0.04192 * rho * df['Elev']
        simple_bouguer = df['FAA'] - bouguer_correction
        
        # Hitung Complete Bouguer Anomaly (dengan TC)
        complete_bouguer = simple_bouguer + df['Koreksi Medan']
        
        # Hitung korelasi absolut dengan elevasi
        corr = np.abs(np.corrcoef(complete_bouguer, df['Elev'])[0, 1])
        correlations.append(corr)
    
    # Densitas optimal = yang punya korelasi TERENDAH
    optimal_idx = np.argmin(correlations)
    optimal_density = densities[optimal_idx]
    
    if debug:
        st.write(f"**Metode Nettleton:**")
        st.write(f"- Densitas optimal: {optimal_density:.3f} g/cm³")
        st.write(f"- Korelasi minimal: {correlations[optimal_idx]:.4f}")
    
    return optimal_density, densities, correlations

def maximum_entropy_method(df, density_range=(2.0, 3.0), n_points=50, debug=False):
    """
    Maximum entropy density determination (Komatiitsch, 1994).
    Memilih densitas yang memaksimalkan entropi dari distribusi residual.
    """
    if 'FAA' not in df.columns or 'Elev' not in df.columns:
        st.error("Data tidak lengkap untuk metode Maximum Entropy. Butuh kolom: FAA, Elev")
        return None, None, None
    
    from scipy.stats import entropy
    
    densities = np.linspace(density_range[0], density_range[1], n_points)
    entropy_values = []
    
    for rho in densities:
        # Hitung residual Bouguer anomaly
        bouguer_anomaly = df['FAA'] - 0.04192 * rho * df['Elev']
        
        # Histogram residuals (normalisasi ke PDF)
        hist, bins = np.histogram(bouguer_anomaly, bins=min(30, len(df)//5), density=True)
        
        # Hitung entropy (hindari log(0))
        hist_safe = hist.copy()
        hist_safe[hist_safe == 0] = 1e-10
        ent = entropy(hist_safe)
        entropy_values.append(ent)
    
    # Densitas optimal = maksimum entropy
    optimal_idx = np.argmax(entropy_values)
    optimal_density = densities[optimal_idx]
    
    if debug:
        st.write(f"**Metode Maximum Entropy:**")
        st.write(f"- Densitas optimal: {optimal_density:.3f} g/cm³")
        st.write(f"- Entropi maksimal: {entropy_values[optimal_idx]:.4f}")
    
    return optimal_density, densities, entropy_values

def iterative_least_squares_method(df, initial_rho=2.67, max_iter=20, tol=0.001, debug=False):
    """
    Iterative least squares density determination.
    """
    if 'FAA' not in df.columns or 'Elev' not in df.columns:
        st.error("Data tidak lengkap untuk metode Iterative LS. Butuh kolom: FAA, Elev")
        return None, 0
    
    rho = initial_rho
    prev_rho = 0
    history = []
    
    for i in range(max_iter):
        # Hitung Bouguer anomaly
        bouguer = df['FAA'] - 0.04192 * rho * df['Elev']
        
        # Linear regression: FAA = slope * Elev + intercept
        X = df['Elev'].values.reshape(-1, 1)
        y = df['FAA'].values
        
        # Simple least squares
        slope, intercept = np.polyfit(df['Elev'], df['FAA'], 1)
        
        # Update density
        new_rho = slope / 0.04192
        
        history.append(new_rho)
        
        # Check convergence
        if abs(new_rho - rho) < tol:
            if debug:
                st.write(f"**Metode Iterative Least Squares:**")
                st.write(f"- Densitas optimal: {new_rho:.3f} g/cm³")
                st.write(f"- Iterasi: {i+1}")
                st.write(f"- Slope: {slope:.6f}")
            return new_rho, i+1
        
        rho = new_rho
    
    if debug:
        st.write(f"**Metode Iterative Least Squares:** (tidak konvergen dalam {max_iter} iterasi)")
        st.write(f"- Densitas akhir: {rho:.3f} g/cm³")
    
    return rho, max_iter

def comprehensive_density_analysis(df, debug=True):
    """
    Analisis densitas komprehensif dengan berbagai metode.
    TANPA METODE PARASNIS.
    """
    st.subheader("🔄 Analisis Densitas Komprehensif")
    
    results = {}
    
    # 1. Metode Nettleton
    if 'FAA' in df.columns and 'Elev' in df.columns and 'Koreksi Medan' in df.columns:
        density_nettleton, densities_net, correlations = nettleton_method(
            df, density_range=(1.5, 3.5), step=0.05, debug=debug
        )
        if density_nettleton is not None:
            results['Nettleton'] = density_nettleton
            
            # Plot Nettleton
            fig_net, ax_net = plt.subplots(figsize=(10, 5))
            ax_net.plot(densities_net, correlations, 'b-o', linewidth=2, markersize=5)
            ax_net.axvline(density_nettleton, color='r', linestyle='--', 
                          label=f'Optimal: {density_nettleton:.3f} g/cm³')
            ax_net.set_xlabel('Densitas (g/cm³)')
            ax_net.set_ylabel('|Korelasi dengan Elevasi|')
            ax_net.set_title('Metode Nettleton: Densitas Optimal Meminimalkan Korelasi')
            ax_net.grid(True, alpha=0.3)
            ax_net.legend()
            st.pyplot(fig_net)
    
    # 2. Metode Maximum Entropy
    density_entropy, densities_ent, entropy_vals = maximum_entropy_method(
        df, density_range=(1.5, 3.5), n_points=50, debug=debug
    )
    if density_entropy is not None:
        results['Maximum Entropy'] = density_entropy
    
    # 3. Metode Iterative Least Squares
    density_ils, iterations = iterative_least_squares_method(
        df, initial_rho=2.67, max_iter=20, tol=0.001, debug=debug
    )
    if density_ils is not None:
        results['Iterative LS'] = density_ils
    
    # 4. Tambahkan metode rata-rata elevation jika diperlukan
    if 'Elev' in df.columns:
        # Estimasi sederhana berdasarkan elevasi rata-rata
        elev_mean = df['Elev'].mean()
        if elev_mean < 100:
            estimated_density = 2.1
        elif elev_mean < 500:
            estimated_density = 2.3
        elif elev_mean < 1000:
            estimated_density = 2.5
        else:
            estimated_density = 2.7
        results['Elevation-Based'] = estimated_density
    
    # 5. Tampilkan hasil semua metode
    if results:
        st.subheader("📊 Hasil Semua Metode (Tanpa Parasnis)")
        
        result_df = pd.DataFrame.from_dict(results, orient='index', 
                                          columns=['Densitas (g/cm³)'])
        result_df['Metode'] = result_df.index
        
        # Urutkan berdasarkan densitas
        result_df = result_df.sort_values('Densitas (g/cm³)')
        
        # Hitung statistik TANPA PARASNIS
        mean_density = result_df['Densitas (g/cm³)'].mean()
        median_density = result_df['Densitas (g/cm³)'].median()
        std_density = result_df['Densitas (g/cm³)'].std()
        
        # Tampilkan tabel
        st.dataframe(result_df.style.format({
            'Densitas (g/cm³)': '{:.3f}'
        }))
        
        # Visualisasi perbandingan
        fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
        methods = result_df['Metode'].values
        densities = result_df['Densitas (g/cm³)'].values
        
        bars = ax_comp.barh(methods, densities, color='skyblue', alpha=0.7)
        ax_comp.axvline(mean_density, color='red', linestyle='--', 
                       label=f'Rata-rata: {mean_density:.3f} g/cm³')
        ax_comp.axvline(median_density, color='green', linestyle='--', 
                       label=f'Median: {median_density:.3f} g/cm³')
        
        # Tambahkan nilai di bar
        for i, (bar, density) in enumerate(zip(bars, densities)):
            ax_comp.text(density + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{density:.3f}', va='center', fontweight='bold')
        
        ax_comp.set_xlabel('Densitas (g/cm³)')
        ax_comp.set_title('Perbandingan Hasil Berbagai Metode (Tanpa Parasnis)')
        ax_comp.legend()
        ax_comp.grid(True, alpha=0.3, axis='x')
        
        st.pyplot(fig_comp)
        
        # Rekomendasi
        st.subheader("✅ Rekomendasi Densitas")
        
        # Pilih median karena lebih robust terhadap outlier
        recommended = median_density
        confidence = "Sedang-Tinggi" if std_density < 0.15 else "Sedang"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Densitas Rekomendasi", f"{recommended:.3f} g/cm³")
        with col2:
            st.metric("Konsistensi Metode", f"{std_density:.3f} g/cm³")
        with col3:
            st.metric("Tingkat Keyakinan", confidence)
        
        # Saran berdasarkan nilai densitas
        if recommended < 2.0:
            st.warning("⚠️ Densitas rendah (< 2.0 g/cm³). Kemungkinan batuan sedimen tak terkompaksi atau data error.")
        elif recommended < 2.3:
            st.info("ℹ️ Densitas rendah-sedang (2.0-2.3 g/cm³). Khas untuk batuan sedimen.")
        elif recommended < 2.7:
            st.success("✅ Densitas sedang (2.3-2.7 g/cm³). Khas untuk batuan sedimen terkompaksi atau batuan beku asam.")
        elif recommended < 3.0:
            st.info("ℹ️ Densitas tinggi (2.7-3.0 g/cm³). Khas untuk batuan beku basa atau batuan metamorf.")
        else:
            st.warning("⚠️ Densitas sangat tinggi (> 3.0 g/cm³). Periksa kemungkinan error dalam data atau koreksi.")
        
        return recommended, result_df
    else:
        st.error("Tidak ada metode yang berhasil dihitung.")
        return None, None

def plot_density_validation(df, optimal_density):
    """
    Plot untuk validasi densitas optimal.
    DIMODIFIKASI: Hilangkan plot Parasnis jika tidak relevan.
    """
    if optimal_density is None:
        return None
    
    st.subheader("📈 Validasi Densitas Optimal")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Elevation vs FAA dan Bouguer Anomaly
    axes[0,0].scatter(df['Elev'], df['FAA'], alpha=0.5, s=30, label='FAA', color='blue')
    
    # Hitung Bouguer anomaly dengan densitas optimal
    if 'FAA' in df.columns and 'Elev' in df.columns:
        bouguer_optimal = df['FAA'] - 0.04192 * optimal_density * df['Elev']
        axes[0,0].scatter(df['Elev'], bouguer_optimal, alpha=0.5, s=30, 
                         label=f'CBA (ρ={optimal_density:.2f})', color='red')
        
        # Regresi linear
        slope_faa, _ = np.polyfit(df['Elev'], df['FAA'], 1)
        slope_cba, _ = np.polyfit(df['Elev'], bouguer_optimal, 1)
        
        axes[0,0].plot(df['Elev'], slope_faa * df['Elev'] + np.mean(df['FAA'] - slope_faa*df['Elev']), 
                      'b--', alpha=0.7, label=f'FAA slope: {slope_faa:.4f}')
        axes[0,0].plot(df['Elev'], slope_cba * df['Elev'] + np.mean(bouguer_optimal - slope_cba*df['Elev']), 
                      'r--', alpha=0.7, label=f'CBA slope: {slope_cba:.4f}')
    
    axes[0,0].set_xlabel('Elevasi (m)')
    axes[0,0].set_ylabel('Anomali (mGal)')
    axes[0,0].set_title('Elevasi vs Anomali')
    axes[0,0].legend(loc='best', fontsize=8)
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Distribusi Residual CBA (ganti Parasnis plot)
    if 'FAA' in df.columns and 'Elev' in df.columns:
        bouguer_residual = df['FAA'] - 0.04192 * optimal_density * df['Elev']
        axes[0,1].hist(bouguer_residual, bins=20, alpha=0.7, edgecolor='black', color='orange')
        axes[0,1].axvline(bouguer_residual.mean(), color='red', linestyle='--',
                         label=f'Mean: {bouguer_residual.mean():.2f} mGal')
        axes[0,1].set_xlabel('Residual CBA (mGal)')
        axes[0,1].set_ylabel('Frekuensi')
        axes[0,1].set_title('Distribusi Residual CBA')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Korelasi vs Densitas (Nettleton-style)
    densities_test = np.linspace(1.5, 3.5, 50)
    correlations = []
    
    for rho in densities_test:
        bouguer_test = df['FAA'] - 0.04192 * rho * df['Elev']
        if 'Koreksi Medan' in df.columns:
            cba_test = bouguer_test + df['Koreksi Medan']
        else:
            cba_test = bouguer_test
        
        corr = np.abs(np.corrcoef(cba_test, df['Elev'])[0, 1])
        correlations.append(corr)
    
    axes[0,2].plot(densities_test, correlations, 'b-', linewidth=2)
    axes[0,2].axvline(optimal_density, color='r', linestyle='--', 
                     label=f'ρ optimal: {optimal_density:.2f} g/cm³')
    axes[0,2].set_xlabel('Densitas (g/cm³)')
    axes[0,2].set_ylabel('|Korelasi|')
    axes[0,2].set_title('Korelasi CBA-Elevasi vs Densitas')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: Korelasi antara metode (jika ada multiple methods)
    axes[1,0].text(0.5, 0.5, 'Analisis Metode Alternatif\n(Tanpa Parasnis)', 
                  ha='center', va='center', transform=axes[1,0].transAxes,
                  fontsize=12)
    axes[1,0].set_title('Metode Alternatif')
    axes[1,0].axis('off')
    
    # Plot 5: Peta spasial CBA
    if 'Easting' in df.columns and 'Northing' in df.columns:
        sc = axes[1,1].scatter(df['Easting'], df['Northing'], 
                              c=bouguer_optimal, cmap='viridis', s=50, alpha=0.7)
        axes[1,1].set_xlabel('Easting (m)')
        axes[1,1].set_ylabel('Northing (m)')
        axes[1,1].set_title('CBA Spatial Distribution')
        plt.colorbar(sc, ax=axes[1,1], label='CBA (mGal)')
    
    # Plot 6: Perbandingan dengan densitas referensi
    axes[1,2].axhline(y=optimal_density, color='red', linewidth=3, label=f'Optimal: {optimal_density:.2f}')
    
    # Densitas referensi untuk batuan umum
    reference_densities = {
        'Alluvium': 1.8,
        'Sandstone': 2.3,
        'Shale': 2.4,
        'Limestone': 2.5,
        'Granite': 2.65,
        'Basalt': 2.9,
        'Gabbro': 3.0
    }
    
    y_pos = np.arange(len(reference_densities))
    densities_ref = list(reference_densities.values())
    names_ref = list(reference_densities.keys())
    
    bars = axes[1,2].barh(y_pos, densities_ref, alpha=0.6, color='gray')
    
    # Warn bars based on comparison with optimal
    for i, density_ref in enumerate(densities_ref):
        if abs(density_ref - optimal_density) < 0.1:
            bars[i].set_color('green')
        elif abs(density_ref - optimal_density) < 0.2:
            bars[i].set_color('orange')
        else:
            bars[i].set_color('gray')
    
    axes[1,2].set_yticks(y_pos)
    axes[1,2].set_yticklabels(names_ref)
    axes[1,2].set_xlabel('Densitas (g/cm³)')
    axes[1,2].set_title('Perbandingan dengan Batuan Referensi')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig

# ============================================================
# DEBUG FUNCTION UNTUK PARASNIS SLOPE TINGGI
# ============================================================

def debug_parasnis_slope(df_all, debug_mode=True):
    """
    Fungsi khusus untuk debugging gradien Parasnis yang terlalu tinggi.
    """
    if debug_mode:
        st.subheader("🔍 Debugging Gradien Parasnis Tinggi")
        
        # Analisis penyebab
        causes = []
        
        # 1. Cek range data
        if 'X-Parasnis' in df_all.columns and 'Y-Parasnis' in df_all.columns:
            x_range = df_all['X-Parasnis'].max() - df_all['X-Parasnis'].min()
            y_range = df_all['Y-Parasnis'].max() - df_all['Y-Parasnis'].min()
            
            ratio = y_range / x_range if x_range != 0 else float('inf')
            
            if ratio > 1000:
                causes.append(f"Range Y ({y_range:.1f}) >> Range X ({x_range:.1f}), ratio: {ratio:.0f}")
        
        # 2. Cek outlier
        from scipy import stats
        if 'Y-Parasnis' in df_all.columns:
            z_scores = np.abs(stats.zscore(df_all['Y-Parasnis'].dropna()))
            outlier_count = np.sum(z_scores > 3)
            if outlier_count > 0:
                causes.append(f"{outlier_count} outlier dalam Y-Parasnis (|z| > 3)")
        
        # 3. Cek koreksi medan (TC)
        if 'Koreksi Medan' in df_all.columns:
            tc_stats = df_all['Koreksi Medan'].describe()
            if tc_stats['max'] > 50:
                causes.append(f"TC terlalu besar (max={tc_stats['max']:.1f} mGal)")
            if tc_stats['mean'] > 20:
                causes.append(f"TC rata-rata terlalu besar (mean={tc_stats['mean']:.1f} mGal)")
        
        # 4. Cek elevasi
        if 'Elev' in df_all.columns:
            elev_range = df_all['Elev'].max() - df_all['Elev'].min()
            if elev_range > 1000:
                causes.append(f"Range elevasi sangat besar ({elev_range:.0f} m)")
        
        # 5. Tampilkan penyebab
        if causes:
            st.warning("**Penyebab potensial gradien tinggi:**")
            for cause in causes:
                st.write(f"- {cause}")
            
            # Solusi
            st.info("**Solusi yang dicoba:**")
            st.write("1. **Periksa koreksi medan (TC)** - nilai harus antara 0-50 mGal untuk kebanyakan area")
            st.write("2. **Kurangi max_radius OSS** ke 2000-3000 m")
            st.write("3. **Matikan optimized elevation** (use_optimized_elevation=False)")
            st.write("4. **Tingkatkan threshold_mGal** ke 0.1-0.2 mGal")
            st.write("5. **Verifikasi formula X-Parasnis**: X = 0.04192 * Elev - TC")
            
            return True
        
    return False

# ============================================================
# UI STREAMLIT
# ============================================================
st.markdown(
    f"""
    <div style="display:flex; align-items:center;">
        <img src="https://raw.githubusercontent.com/dzakyw/AutoGrav/main/logo esdm.png" style="width:200px; margin-right:5px;">
        <div>
            <h2 style="margin-bottom:0;">Auto Grav - Semua Terasa Cepat</h2>
            <p style="color:red; font-weight:bold;">OSS Algorithm CRITICALLY CORRECTED Version</p>
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

# PARAMETER OSS - OPTIMIZED SETTINGS
st.sidebar.subheader("OSS Algorithm Parameters (Optimized)")
debug_mode = st.sidebar.checkbox("Debug Mode", value=True)  # Enabled for debugging
st.session_state.debug_mode = debug_mode

threshold_mgal = st.sidebar.slider(
    "Threshold (mGal) for subdivision",
    min_value=0.01,
    max_value=0.5,
    value=0.1,  # Increased from 0.05
    step=0.01,
    help="Start with 0.1 mGal for stable results"
)

max_radius = st.sidebar.number_input(
    "Maximum Radius (m) - RECOMMENDED: 2000-3000",
    value=2000,  # Reduced from 4500
    step=100,
    help="Jangkauan maksimum untuk koreksi medan. Kurangi jika slope terlalu tinggi."
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
    value=False,  # Disabled initially
    help="Menggunakan persamaan (8) untuk optimasi z'. MATIKAN jika slope tinggi"
)

min_points_sector = st.sidebar.number_input(
    "Minimum points per sector",
    min_value=5,
    max_value=100,
    value=20,  # Increased
    help="Sector dengan points < ini tidak di-subdivide"
)

# Advanced parameters
with st.sidebar.expander("Advanced Parameters"):
    tolerance_nGal = st.number_input("Tolerance (nGal)", value=10.0, min_value=0.1, max_value=100.0)
    theta_step = st.number_input("Theta step (degrees)", value=5.0, min_value=0.5, max_value=10.0)
    r_step_near = st.number_input("R step near (m)", value=20.0, min_value=5.0, max_value=50.0)
    r_step_far = st.number_input("R step far (m)", value=100.0, min_value=20.0, max_value=200.0)

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
        'tolerance_nGal': tolerance_nGal,
        'threshold_mGal': threshold_mgal,
        'theta_step': theta_step,
        'r_step_near': r_step_near,
        'r_step_far': r_step_far,
        'min_points_per_sector': min_points_sector,
        'use_optimized_elevation': use_optimized_elev,
        'debug': debug_mode,
        'density': density
    }
    
    st.info(f"**OSS Parameters:** Max radius = {max_radius} m, Threshold = {threshold_mgal} mGal, Density = {density} kg/m³")
    st.warning(f"**IMPORTANT:** Using ABSOLUTE elevation (z = DEM height), not elevation difference!")
    
    # Tampilkan contoh perhitungan latitude correction untuk validasi
    st.subheader("Latitude Correction Validation")
    sample_lat = -7.723
    sample_corr = latitude_correction(sample_lat)
    st.write(f"Contoh untuk latitude {sample_lat}°:")
    st.write(f"- Latitude correction (g_teoritis): **{sample_corr:.6f} mGal**")
    st.info("""
    **CATATAN PENTING:**
    - Nilai latitude correction (~978,000 mGal) adalah nilai GRAVITASI TEORITIS di lintang tersebut
    - Bukan koreksi kecil seperti free-air atau terrain correction
    - Nilai ini digunakan sebagai referensi dalam perhitungan FAA
    - Nilai ini NORMAL dan BENAR!
    """)
    
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
        
        # Perhitungan koreksi
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
        
        # X-Parasnis calculation - KEEPING YOUR ORIGINAL FORMULA
        df["Koreksi Medan"] = tc_list
        df["X-Parasnis"] = 0.04192 * df["Elev"] - df["Koreksi Medan"]  # YOUR ORIGINAL FORMULA
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
        
        density_parasnis = slope / 0.04192
        
        st.info(f"**Parasnis Regression:** Slope (K) = {slope:.5f}, R² = {r_squared:.3f}")
        st.info(f"**Implied Density:** ρ = {density_parasnis:.3f} g/cm³")
        
        if r_squared < 0.7:
            st.warning("Low R² value in Parasnis regression! Check TC calculations.")
        
        # DEBUG jika slope terlalu tinggi
        if abs(slope) > 0.15:  # Jika gradien > 0.15
            st.error(f"⚠️ VERY HIGH SLOPE DETECTED: {slope:.3f}")
            debug_parasnis_slope(df_all, debug_mode)
    else:
        slope = np.nan
        st.warning("Not enough data for Parasnis regression.")
    
    # PERHATIAN: Formula Bouguer correction menggunakan slope dari Parasnis
    if not np.isnan(slope):
        df_all["Bouger Correction"] = 0.04192 * slope * df_all["Elev"]
    else:
        df_all["Bouger Correction"] = 0.04192 * 2.67 * df_all["Elev"]  # Fallback density
    
    df_all["Simple Bouger Anomaly"] = df_all["FAA"] - df_all["Bouger Correction"]
    df_all["Complete Bouger Anomaly"] = df_all["Simple Bouger Anomaly"] + df_all["Koreksi Medan"]
    
    st.subheader("Hasil Yang Diproses")
    
    with st.expander("Lihat Data Preview", expanded=True):
        st.dataframe(df_all.head(20))
        st.write(f"**Total rows processed:** {len(df_all)}")
        st.write(f"**Total sheets processed:** {len(all_dfs)}")
    
    # TABS
    st.subheader("Visualisasi Hasil")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Parasnis Plot", "Topography", "CBA Map", "Analisis Densitas", "Data Export"])

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
            ax_parasnis.set_title(f"Diagram Parasnis (Slope = {slope:.5f}, ρ = {density_parasnis:.2f} g/cm³)")
            ax_parasnis.grid(True, linestyle="--", alpha=0.5)
            ax_parasnis.legend()
            st.pyplot(fig_parasnis)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"**Slope (K):** {slope:.5f}")
            with col2:
                st.info(f"**R-squared:** {r_squared:.3f}")
            with col3:
                st.info(f"**Implied Density:** {density_parasnis:.3f} g/cm³")
                
            # Warning jika density tidak realistic
            if density_parasnis < 1.5 or density_parasnis > 3.5:
                st.error(f"⚠️ Unrealistic density from Parasnis: {density_parasnis:.2f} g/cm³")
                st.write("Possible causes:")
                st.write("1. OSS TC values are too large/small")
                st.write("2. X-Parasnis formula might need adjustment")
                st.write("3. Try Nettleton method for better density estimation")
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
    
    # Tab 4: Analisis Densitas
    with tab4:
    st.header("📊 Analisis Densitas Komprehensif")
    
    if 'df_all' in locals() and len(df_all) > 0:
        # Jalankan analisis densitas komprehensif TANPA PARASNIS
        with st.spinner("Menghitung densitas dengan berbagai metode (tanpa Parasnis)..."):
            optimal_density, density_results = comprehensive_density_analysis(df_all, debug=debug_mode)
        
        if optimal_density is not None:
            # Update perhitungan dengan densitas optimal
            st.subheader("🔄 Perhitungan Ulang dengan Densitas Optimal")
            
            # Hitung Bouguer anomaly dengan densitas optimal
            df_all["Bouger Correction Optimal"] = 0.04192 * optimal_density * df_all["Elev"]
            df_all["Simple Bouger Anomaly Optimal"] = df_all["FAA"] - df_all["Bouger Correction Optimal"]
            df_all["Complete Bouger Anomaly Optimal"] = df_all["Simple Bouger Anomaly Optimal"] + df_all["Koreksi Medan"]
            
            # Tampilkan plot validasi
            fig_validation = plot_density_validation(df_all, optimal_density)
            if fig_validation:
                st.pyplot(fig_validation)
            
            # Opsi untuk menggunakan densitas yang berbeda
            st.subheader("⚙️ Pilih Densitas untuk Perhitungan Final")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                use_optimal = st.checkbox("Gunakan densitas optimal", value=True)
            with col2:
                # Tampilkan metode lain jika tersedia
                if density_results is not None and len(density_results) > 1:
                    other_methods = density_results[density_results['Metode'] != 'Nettleton']
                    if len(other_methods) > 0:
                        selected_method = st.selectbox(
                            "Pilih metode lain",
                            options=other_methods['Metode'].tolist()
                        )
                        other_density = other_methods[other_methods['Metode'] == selected_method]['Densitas (g/cm³)'].values[0]
                        use_other = st.checkbox(f"Gunakan {selected_method}", value=False)
            with col3:
                custom_density = st.number_input("Densitas kustom (g/cm³)", 
                                                value=float(optimal_density), 
                                                min_value=1.0, max_value=4.0, step=0.01)
            
            if use_optimal:
                final_density = optimal_density
                st.success(f"Menggunakan densitas optimal (median): {final_density:.3f} g/cm³")
            elif 'use_other' in locals() and use_other:
                final_density = other_density
                st.info(f"Menggunakan densitas dari {selected_method}: {final_density:.3f} g/cm³")
            else:
                final_density = custom_density
                st.warning(f"Menggunakan densitas kustom: {final_density:.3f} g/cm³")
            
            # Perhitungan final dengan densitas terpilih
            df_all["Bouger Correction Final"] = 0.04192 * final_density * df_all["Elev"]
            df_all["Simple Bouger Anomaly Final"] = df_all["FAA"] - df_all["Bouger Correction Final"]
            df_all["Complete Bouger Anomaly Final"] = df_all["Simple Bouger Anomaly Final"] + df_all["Koreksi Medan"]
            
            # Tampilkan statistik
            st.subheader("📈 Statistik Anomali Bouguer Final")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min CBA", f"{df_all['Complete Bouger Anomaly Final'].min():.1f} mGal")
            with col2:
                st.metric("Max CBA", f"{df_all['Complete Bouger Anomaly Final'].max():.1f} mGal")
            with col3:
                st.metric("Mean CBA", f"{df_all['Complete Bouger Anomaly Final'].mean():.1f} mGal")
            with col4:
                st.metric("Std Dev CBA", f"{df_all['Complete Bouger Anomaly Final'].std():.1f} mGal")
            
            # Plot distribusi CBA final
            fig_cba_final, ax_cba_final = plt.subplots(figsize=(10, 5))
            ax_cba_final.hist(df_all['Complete Bouger Anomaly Final'], bins=30, 
                             alpha=0.7, edgecolor='black', color='green')
            ax_cba_final.axvline(df_all['Complete Bouger Anomaly Final'].mean(), 
                               color='red', linestyle='--',
                               label=f'Mean: {df_all["Complete Bouger Anomaly Final"].mean():.1f} mGal')
            ax_cba_final.set_xlabel('Complete Bouguer Anomaly (mGal)')
            ax_cba_final.set_ylabel('Frequency')
            ax_cba_final.set_title(f'Distribusi CBA Final (ρ = {final_density:.3f} g/cm³)')
            ax_cba_final.legend()
            ax_cba_final.grid(True, alpha=0.3)
            st.pyplot(fig_cba_final)
            
            # Simpan densitas final ke session state
            st.session_state.final_density = final_density
            st.session_state.density_results = density_results
            
    else:
        st.warning("Harap proses data terlebih dahulu di tab utama.")
    # Tab 5: Data Export
    with tab5:
        st.subheader("Download Hasil Perhitungan")
    
    # Tambahkan informasi densitas ke file export jika ada
    if 'final_density' in st.session_state:
        st.info(f"Densitas yang digunakan: {st.session_state.final_density:.3f} g/cm³")
    
    col1, col2, col3 = st.columns(3)
    
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
    
    with col3:
        # Export hasil analisis densitas jika ada
        if 'density_results' in st.session_state:
            density_df = st.session_state.density_results
            density_csv = density_df.to_csv(index=False)
            st.download_button(
                label="Download Density Analysis (.csv)",
                data=density_csv.encode('utf-8'),
                file_name="autograv_density_analysis.csv",
                mime="text/csv"
            )
    
    # Tampilkan summary
    st.info("### Ringkasan Hasil")
    
    if 'final_density' in st.session_state:
        st.write(f"- **Densitas yang digunakan:** {st.session_state.final_density:.3f} g/cm³")
    
    if 'slope' in locals() and not np.isnan(slope):
        st.write(f"- **Gradien Parasnis (K):** {slope:.5f}")
        st.write(f"- **R² Parasnis:** {r_squared:.3f}")
    
    st.write(f"- **Total stasiun diproses:** {len(df_all)}")
    st.write(f"- **Total sheets:** {len(all_dfs)}")
    st.write(f"- **Mean Terrain Correction:** {tc_values.mean():.2f} mGal")
    st.write(f"- **TC Range:** {tc_values.min():.2f} to {tc_values.max():.2f} mGal")
    
    # Final recommendation
    if 'slope' in locals() and not np.isnan(slope) and abs(slope) > 0.15:
        st.error("""
        **⚠️ WARNING: HIGH PARASNIS SLOPE DETECTED**
        
        Your Parasnis slope is > 0.15, indicating potential issues:
        
        1. **TC values may be incorrect** - Check OSS parameters
        2. **Try reducing max_radius** to 2000 m
        3. **Disable optimized elevation** 
        4. **Consider using Nettleton method** for density determination
        
        The corrected OSS algorithm now uses ABSOLUTE elevation (z = DEM height).
        This should give more realistic TC values and Parasnis slopes.
        """)
    else:
        st.success("""
        **✅ PROCESSING COMPLETE**
        
        The corrected OSS algorithm has been applied with:
        - Absolute elevation (z = DEM height)
        - Optimized parameters for stable results
        - Comprehensive density analysis
        
        Download your results using the buttons above.
        """)

# ============================================================
# TAMBAHAN: Informasi troubleshooting
# ============================================================
st.sidebar.subheader("Troubleshooting Tips")

with st.sidebar.expander("Jika slope Parasnis > 3"):
    st.write("""
    1. **Kurangi max_radius** ke 2000 m
    2. **Matikan optimized elevation** (use_optimized_elev=False)
    3. **Tingkatkan threshold** ke 0.2-0.5 mGal
    4. **Verifikasi formula X-Parasnis**: X = 0.04192 * Elev - TC
    5. **TC values** harus antara 0-50 mGal untuk kebanyakan area
    """)

with st.sidebar.expander("Jika OSS terlalu lambat"):
    st.write("""
    1. **Kurangi max_radius** 
    2. **Tingkatkan theta_step** ke 5-10 derajat
    3. **Tingkatkan r_step** values
    4. **Tingkatkan min_points_per_sector** 
    5. **Matikan debug mode**
    """)

with st.sidebar.expander("Koreksi yang dilakukan"):
    st.write("""
    **CRITICAL FIXES APPLIED:**
    
    1. **z = ABSOLUTE DEM elevation** (not elevation difference)
    2. **Optimized default parameters** for stable results
    3. **Added sanity checks** for TC values
    4. **Debug function** for high slopes
    5. **Keep original X-Parasnis formula**: X = 0.04192 * Elev - TC
    """)


