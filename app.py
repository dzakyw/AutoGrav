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
# 1. HASH FUNCTION
# ---------------------------------------------
def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

# ---------------------------------------------
# 2. USER DATABASE (EDIT SESUAI KEBUTUHAN)
# ---------------------------------------------
USER_DB = {
    "admin": hash_password("admin"),  # ubah sesuai kebutuhan
    "user": hash_password("12345"),   # ubah sesuai kebutuhan
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
    st.stop()  # hanya menampilkan login page

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
    Works if TIFF contains:
    - ModelPixelScaleTag (33550)
    - ModelTiepointTag (33922)
    """
    img = Image.open(file)
    arr = np.array(img, dtype=float)
    meta = img.tag_v2
    
    if 33550 not in meta or 33922 not in meta:
        raise ValueError("GeoTIFF metadata not found. TIFF requires TFW or manual bounding box.")
    
    # Metadata
    scaleX, scaleY, _ = meta[33550]  # pixel size
    tiepoint = meta[33922]           # tiepoint structure
    # According to GeoTIFF specs:
    X0 = tiepoint[3]  # model_space_X of UL corner
    Y0 = tiepoint[4]  # model_space_Y of UL corner
    
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
        E, N, _, _ = latlon_to_utm_redfearn(df["Lat"], df["Lon"])
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

def simple_nagy_prism(e0, n0, z0, dem_df, density, max_radius):
    """
    Implementasi sederhana Nagy prism untuk referensi.
    HANYA untuk perbandingan, bukan implementasi lengkap.
    """
    import numpy as np
    
    if dem_df is None or len(dem_df) == 0:
        return 0.0
    
    # Pilih titik dalam radius
    dx = dem_df['Easting'].to_numpy() - e0
    dy = dem_df['Northing'].to_numpy() - n0
    r = np.sqrt(dx**2 + dy**2)
    mask = r <= max_radius
    
    if not np.any(mask):
        return 0.0
    
    # Data dalam radius
    z_diff = dem_df['Elev'].to_numpy()[mask] - z0
    r_vals = r[mask]
    
    # Pendekatan sederhana: g ~ G * ρ * ΔV * z / r^3
    G = 6.67430e-11
    # Asumsikan setiap titik mewakili area grid
    # Coba infer grid spacing dari data
    unique_x = np.unique(dem_df['Easting'].to_numpy())
    unique_y = np.unique(dem_df['Northing'].to_numpy())
    
    if len(unique_x) > 1 and len(unique_y) > 1:
        dx_grid = np.median(np.diff(np.sort(unique_x)))
        dy_grid = np.median(np.diff(np.sort(unique_y)))
        cell_area = dx_grid * dy_grid
    else:
        cell_area = 30 * 30  # Default 30x30 m jika tidak bisa diinfer
    
    # Hitung efek prism sederhana
    volumes = cell_area * np.abs(z_diff)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        g_si = G * density * np.sum(volumes * z_diff / (r_vals**3 + 1e-10))
    
    # Convert to mGal
    g_mgal = g_si * 1e5
    
    return float(g_mgal)

# ============================================================
# KONSTANTA & KERNEL DASAR SESUAI PAPER
# ============================================================
G = 6.67430e-11           # m^3 kg^-1 s^-2
RHO = 2670.0              # kg/m^3 (2.67 g/cm^3)
NANO_TO_MGAL = 1e-6       # 1 nGal = 1e-6 mGal
MICRO_TO_MGAL = 1e-3      # 1 μGal = 1e-3 mGal

def terrain_effect_cylindrical_sector(R1, R2, theta1, theta2, z, density):
    """
    PERSAMAAN (1) dari paper: Δg_T = GρΔθ[R2-R1+√(R1²+z²)-√(R2²+z²)]
    """
    # Convert angles to radians
    theta1_rad = np.radians(theta1)
    theta2_rad = np.radians(theta2)
    Delta_theta = theta2_rad - theta1_rad
    
    # Calculate terrain effect in SI units (m/s²)
    term = (R2 - R1) + np.sqrt(R1**2 + z**2) - np.sqrt(R2**2 + z**2)
    delta_g_si = G * density * Delta_theta * term
    
    # Convert to mGal (1 mGal = 10^-5 m/s²)
    return delta_g_si * 1e5

def optimized_elevation(z_avg, deviations, R1, R2, Delta_theta, r_points):
    """
    PERSAMAAN (8) dari paper: Optimized elevation z'
    """
    if len(deviations) == 0:
        return z_avg
    
    # Avoid division by zero
    if R2 == R1 or Delta_theta == 0:
        return z_avg
    
    # Calculate weighted sum according to Eq. (8)
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
    
    # Calculate z'²
    factor = (R2 * R1) / (R2 - R1)
    z_prime_sq = z_avg**2 + factor * (numerator_sum / denominator_sum)
    
    # Ensure non-negative
    z_prime_sq = max(z_prime_sq, 0.0)
    
    return np.sqrt(z_prime_sq)

# ============================================================
# FUNGSI VALIDASI
# ============================================================
def validate_tc_value(tc_value, station_name, method, debug=False):
    """Validasi nilai TC yang reasonable"""
    validated_tc = tc_value
    
    # Rule 1: TC tidak mungkin negatif besar
    if tc_value < -0.5:
        if debug:
            st.warning(f"Station {station_name}: TC sangat negatif ({tc_value:.3f} mGal)")
        validated_tc = max(tc_value, 0.0)
    
    # Rule 2: TC biasanya antara 0-50 mGal untuk topografi normal
    elif tc_value > 50.0:
        if debug:
            st.warning(f"Station {station_name}: TC sangat besar ({tc_value:.1f} mGal)")
        validated_tc = min(tc_value, 100.0)
    
    # Rule 3: TC < 0.01 mGal mungkin error
    elif 0 <= tc_value < 0.01:
        if debug:
            st.warning(f"Station {station_name}: TC sangat kecil ({tc_value:.6f} mGal)")
    
    return validated_tc

def validate_data(dem_df, station_df):
    """Validasi konsistensi data sebelum perhitungan OSS"""
    
    st.subheader("Data Validation")
    
    # 1. Cek range elevasi
    col1, col2 = st.columns(2)
    with col1:
        st.write("**DEM Statistics**")
        st.write(f"- Points: {len(dem_df):,}")
        st.write(f"- Elev min: {dem_df['Elev'].min():.1f} m")
        st.write(f"- Elev max: {dem_df['Elev'].max():.1f} m")
        st.write(f"- Elev mean: {dem_df['Elev'].mean():.1f} m")
    
    with col2:
        st.write("**Station Statistics**")
        st.write(f"- Stations: {len(station_df)}")
        st.write(f"- Elev min: {station_df['Elev'].min():.1f} m")
        st.write(f"- Elev max: {station_df['Elev'].max():.1f} m")
    
    # 2. Cek apakah stasiun dalam range DEM
    dem_bounds = {
        'E_min': dem_df['Easting'].min(),
        'E_max': dem_df['Easting'].max(),
        'N_min': dem_df['Northing'].min(),
        'N_max': dem_df['Northing'].max()
    }
    
    stations_outside = station_df[
        (station_df['Easting'] < dem_bounds['E_min']) |
        (station_df['Easting'] > dem_bounds['E_max']) |
        (station_df['Northing'] < dem_bounds['N_min']) |
        (station_df['Northing'] > dem_bounds['N_max'])
    ]
    
    if len(stations_outside) > 0:
        st.error(f"{len(stations_outside)} stasiun di luar area DEM!")
        st.dataframe(stations_outside[['Nama', 'Easting', 'Northing', 'Elev']])
    
    # 3. Plot distribusi
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(dem_df['Elev'], bins=50, alpha=0.7, label='DEM')
    axes[0].hist(station_df['Elev'], bins=20, alpha=0.7, label='Stations')
    axes[0].set_xlabel('Elevation (m)')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].set_title('Elevation Distribution')
    
    axes[1].scatter(dem_df['Easting'], dem_df['Northing'], 
                   c=dem_df['Elev'], s=1, alpha=0.5, cmap='terrain')
    axes[1].scatter(station_df['Easting'], station_df['Northing'], 
                   c='red', s=20, marker='^', label='Stations')
    axes[1].set_xlabel('Easting')
    axes[1].set_ylabel('Northing')
    axes[1].legend()
    axes[1].set_title('Spatial Distribution')
    
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================
# CLASS OSSTerrainCorrector YANG DIPERBAIKI
# ============================================================
class OSSTerrainCorrector:
    def __init__(self, dem_df, station_coords, params=None):
        self.e0, self.n0, self.z0 = station_coords
        self.dem_df = dem_df.copy()
        
        # Default parameters
        self.params = {
            'max_radius': 4500.0,
            'tolerance_nGal': 1.0,
            'threshold_mGal': 0.1,  # DIKURANGI dari 1.0
            'theta_step': 1.0,
            'r_step_near': 10.0,
            'r_step_far': 50.0,
            'min_points_per_sector': 10,
            'use_optimized_elevation': True,
            'debug': False
        }
        
        if params:
            self.params.update(params)
        
        # FIX KRITIS: Sesuai paper, z = station_height - average_topographic_height
        # Jadi jika station lebih tinggi dari terrain, z positif
        dx = self.dem_df['Easting'] - self.e0
        dy = self.dem_df['Northing'] - self.n0
        
        self.r = np.sqrt(dx**2 + dy**2)
        self.theta_rad = np.arctan2(dy, dx)
        self.theta_deg = np.degrees(self.theta_rad) % 360.0
        
        # PERBAIKAN: Gunakan station - dem (sesuai paper)
        self.z_rel = self.z0 - self.dem_df['Elev']  # FIXED!
        
        # Debug info
        if self.params.get('debug', False):
            st.write(f"DEBUG: Station elevation = {self.z0:.1f} m")
            st.write(f"DEBUG: z_rel range = {self.z_rel.min():.1f} to {self.z_rel.max():.1f} m")
        
        # Filter by max radius
        mask = self.r <= self.params['max_radius']
        self.r_filtered = self.r[mask].values
        self.theta_filtered = self.theta_deg[mask].values
        self.z_rel_filtered = self.z_rel[mask].values
        
    def _find_turning_points_theta(self, theta1, theta2, R1, R2):
        """Find turning points in angular direction"""
        theta_step = self.params['theta_step']
        tolerance = self.params['tolerance_nGal'] * NANO_TO_MGAL
        
        angles = []
        terrain_values = []
        
        theta_current = theta1
        while theta_current <= theta2:
            mask_left = (self.theta_filtered >= theta1) & (self.theta_filtered < theta_current)
            mask_right = (self.theta_filtered >= theta_current) & (self.theta_filtered <= theta2)
            
            mask_left_full = mask_left & (self.r_filtered >= R1) & (self.r_filtered <= R2)
            mask_right_full = mask_right & (self.r_filtered >= R1) & (self.r_filtered <= R2)
            
            terrain_left = 0.0
            if np.any(mask_left_full):
                z_avg_left = np.mean(self.z_rel_filtered[mask_left_full])
                terrain_left = terrain_effect_cylindrical_sector(
                    R1, R2, theta1, theta_current, z_avg_left, RHO
                )
            
            terrain_right = 0.0
            if np.any(mask_right_full):
                z_avg_right = np.mean(self.z_rel_filtered[mask_right_full])
                terrain_right = terrain_effect_cylindrical_sector(
                    R1, R2, theta_current, theta2, z_avg_right, RHO
                )
            
            total_terrain = terrain_left + terrain_right
            angles.append(theta_current)
            terrain_values.append(total_terrain)
            theta_current += theta_step
        
        turning_points = []
        if len(terrain_values) > 2:
            for i in range(1, len(terrain_values)-1):
                diff1 = terrain_values[i] - terrain_values[i-1]
                diff2 = terrain_values[i+1] - terrain_values[i]
                if abs(diff2 - diff1) > tolerance:
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
            mask_inner = (self.r_filtered >= R1) & (self.r_filtered < R_current)
            mask_outer = (self.r_filtered >= R_current) & (self.r_filtered <= R2)
            
            mask_inner_full = mask_inner & (self.theta_filtered >= theta1) & (self.theta_filtered <= theta2)
            mask_outer_full = mask_outer & (self.theta_filtered >= theta1) & (self.theta_filtered <= theta2)
            
            terrain_inner = 0.0
            if np.any(mask_inner_full):
                z_avg_inner = np.mean(self.z_rel_filtered[mask_inner_full])
                terrain_inner = terrain_effect_cylindrical_sector(
                    R1, R_current, theta1, theta2, z_avg_inner, RHO
                )
            
            terrain_outer = 0.0
            if np.any(mask_outer_full):
                z_avg_outer = np.mean(self.z_rel_filtered[mask_outer_full])
                terrain_outer = terrain_effect_cylindrical_sector(
                    R_current, R2, theta1, theta2, z_avg_outer, RHO
                )
            
            total_terrain = terrain_inner + terrain_outer
            radii.append(R_current)
            terrain_values.append(total_terrain)
            R_current += r_step
        
        turning_points = []
        if len(terrain_values) > 2:
            for i in range(1, len(terrain_values)-1):
                diff1 = terrain_values[i] - terrain_values[i-1]
                diff2 = terrain_values[i+1] - terrain_values[i]
                if abs(diff2 - diff1) > tolerance:
                    turning_points.append(radii[i])
        
        return turning_points
    
    def _process_sector(self, theta1, theta2, R1, R2, depth=0):
        """Recursive sector processing"""
        threshold = self.params['threshold_mGal']
        
        mask = (self.theta_filtered >= theta1) & (self.theta_filtered <= theta2) & \
               (self.r_filtered >= R1) & (self.r_filtered <= R2)
        
        if not np.any(mask):
            return 0.0, []
        
        z_sector = self.z_rel_filtered[mask]
        r_sector = self.r_filtered[mask]
        
        if len(z_sector) < self.params['min_points_per_sector']:
            z_avg = np.mean(z_sector) if len(z_sector) > 0 else 0.0
            terrain = terrain_effect_cylindrical_sector(R1, R2, theta1, theta2, z_avg, RHO)
            return terrain, [(theta1, theta2, R1, R2, z_avg, terrain)]
        
        z_avg = np.mean(z_sector)
        terrain_avg = terrain_effect_cylindrical_sector(R1, R2, theta1, theta2, z_avg, RHO)
        
        if abs(terrain_avg) < threshold:
            if self.params['use_optimized_elevation']:
                Delta_theta = np.radians(theta2 - theta1)
                deviations = z_sector - z_avg
                z_opt = optimized_elevation(z_avg, deviations, R1, R2, Delta_theta, r_sector)
                terrain_final = terrain_effect_cylindrical_sector(R1, R2, theta1, theta2, z_opt, RHO)
            else:
                terrain_final = terrain_avg
                z_opt = z_avg
            
            return terrain_final, [(theta1, theta2, R1, R2, z_opt, terrain_final)]
        
        turning_points_theta = self._find_turning_points_theta(theta1, theta2, R1, R2)
        
        if turning_points_theta:
            total_terrain = 0.0
            all_subsectors = []
            angles = [theta1] + sorted(turning_points_theta) + [theta2]
            
            for i in range(len(angles)-1):
                sub_theta1, sub_theta2 = angles[i], angles[i+1]
                sub_mask = (self.theta_filtered >= sub_theta1) & (self.theta_filtered <= sub_theta2) & \
                          (self.r_filtered >= R1) & (self.r_filtered <= R2)
                
                if np.any(sub_mask):
                    z_sub = self.z_rel_filtered[sub_mask]
                    z_avg_sub = np.mean(z_sub)
                    terrain_sub = terrain_effect_cylindrical_sector(R1, R2, sub_theta1, sub_theta2, z_avg_sub, RHO)
                    
                    if abs(terrain_sub) >= threshold:
                        turning_points_r = self._find_turning_points_radius(sub_theta1, sub_theta2, R1, R2)
                        
                        if turning_points_r:
                            radii = [R1] + sorted(turning_points_r) + [R2]
                            for j in range(len(radii)-1):
                                sub_R1, sub_R2 = radii[j], radii[j+1]
                                sub_terrain, sub_subsectors = self._process_sector(
                                    sub_theta1, sub_theta2, sub_R1, sub_R2, depth+1
                                )
                                total_terrain += sub_terrain
                                all_subsectors.extend(sub_subsectors)
                        else:
                            if self.params['use_optimized_elevation']:
                                r_sub = self.r_filtered[sub_mask]
                                Delta_theta = np.radians(sub_theta2 - sub_theta1)
                                deviations = z_sub - z_avg_sub
                                z_opt = optimized_elevation(z_avg_sub, deviations, R1, R2, Delta_theta, r_sub)
                                terrain_final = terrain_effect_cylindrical_sector(
                                    R1, R2, sub_theta1, sub_theta2, z_opt, RHO
                                )
                            else:
                                terrain_final = terrain_sub
                                z_opt = z_avg_sub
                            
                            total_terrain += terrain_final
                            all_subsectors.append((sub_theta1, sub_theta2, R1, R2, z_opt, terrain_final))
                    else:
                        if self.params['use_optimized_elevation']:
                            r_sub = self.r_filtered[sub_mask]
                            Delta_theta = np.radians(sub_theta2 - sub_theta1)
                            deviations = z_sub - z_avg_sub
                            z_opt = optimized_elevation(z_avg_sub, deviations, R1, R2, Delta_theta, r_sub)
                            terrain_final = terrain_effect_cylindrical_sector(
                                R1, R2, sub_theta1, sub_theta2, z_opt, RHO
                            )
                        else:
                            terrain_final = terrain_sub
                            z_opt = z_avg_sub
                        
                        total_terrain += terrain_final
                        all_subsectors.append((sub_theta1, sub_theta2, R1, R2, z_opt, terrain_final))
            
            return total_terrain, all_subsectors
        
        else:
            turning_points_r = self._find_turning_points_radius(theta1, theta2, R1, R2)
            
            if turning_points_r:
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
            else:
                if self.params['use_optimized_elevation']:
                    Delta_theta = np.radians(theta2 - theta1)
                    deviations = z_sector - z_avg
                    z_opt = optimized_elevation(z_avg, deviations, R1, R2, Delta_theta, r_sector)
                    terrain_final = terrain_effect_cylindrical_sector(R1, R2, theta1, theta2, z_opt, RHO)
                else:
                    terrain_final = terrain_avg
                    z_opt = z_avg
                
                return terrain_final, [(theta1, theta2, R1, R2, z_opt, terrain_final)]
    
    def calculate_terrain_correction(self):
        """Main method to calculate terrain correction"""
        if self.params['debug']:
            st.write(f"Starting OSS calculation for station at ({self.e0:.1f}, {self.n0:.1f})")
            st.write(f"DEM points in radius: {len(self.r_filtered)}")
        
        total_tc, subsectors = self._process_sector(
            theta1=0.0,
            theta2=360.0,
            R1=0.0,
            R2=self.params['max_radius']
        )
        
        if self.params['debug'] and subsectors:
            st.write(f"Total TC: {total_tc:.6f} mGal")
            st.write(f"Number of subsectors: {len(subsectors)}")
            
            subsectors_sorted = sorted(subsectors, key=lambda x: abs(x[5]), reverse=True)[:3]
            st.write("Top 3 subsectors:")
            for i, (t1, t2, r1, r2, z, tc) in enumerate(subsectors_sorted):
                st.write(f"  {i+1}: θ={t1:.1f}-{t2:.1f}°, r={r1:.0f}-{r2:.0f}m, z={z:.1f}m, Δg={tc:.4f}mGal")
        
        return total_tc, subsectors

def calculate_oss_correction(dem_df, station_row, params=None):
    """
    Wrapper function for Streamlit integration dengan validasi
    """
    station_coords = (
        float(station_row['Easting']),
        float(station_row['Northing']),
        float(station_row['Elev'])
    )
    
    corrector = OSSTerrainCorrector(dem_df, station_coords, params)
    tc_value, _ = corrector.calculate_terrain_correction()
    
    return tc_value

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
demf = st.sidebar.file_uploader("Upload DEM (CSV/XYZ)", type=["csv","txt","xyz"])
kmf = st.sidebar.file_uploader("Koreksi Medan manual (optional jika punya)", type=["csv","xlsx"])
G_base = st.sidebar.number_input("G Absolute di Base", value=0.0)

# ============================================================
# TAMBAHAN: DEBUG MODE DAN PARAMETER OSS
# ============================================================
debug_mode = st.sidebar.checkbox("🛠️ Debug Mode", value=False)

with st.sidebar.expander("⚙️ OSS Algorithm Parameters", expanded=debug_mode):
    threshold_mgal = st.slider(
        "Threshold (mGal) for subdivision",
        min_value=0.01,
        max_value=2.0,
        value=0.1,  # DIKURANGI dari 1.0 ke 0.1
        step=0.01,
        help="Nilai lebih kecil = lebih banyak subdivision, lebih akurat"
    )
    
    use_optimized_elev = st.checkbox(
        "Use optimized elevation (z')", 
        value=True,
        help="Menggunakan persamaan (8) untuk optimasi z'"
    )
    
    min_points_sector = st.number_input(
        "Minimum points per sector",
        min_value=1,
        max_value=100,
        value=10,
        help="Sector dengan points < ini tidak di-subdivide"
    )

method_options = ["Optimally Selecting Sectors (OSS) algorithm", "HAMMER (Legacy)", "NAGY Prism (Reference)"]
method = st.sidebar.selectbox("Metode Pengukuran Terrain", method_options)
density = st.sidebar.number_input("Densitas Koreksi Medan (kg/m³)", value=2670.0, step=10.0, format="%.1f")
max_radius = st.sidebar.number_input("Jarak Maksimum (m) untuk Nagy", value=10000, step=1000)
z_ref = st.sidebar.number_input("z_ref (bottom prism reference, m)", value=0.0)
run = st.sidebar.button("Run")

st.sidebar.subheader("Contoh File Input")
st.sidebar.write("[Contoh Data Input Gravity](https://github.com/dzakyw/AutoGrav/raw/9bb43e1559c823350f2371360309d84eaab5ea38/sample_gravity.xlsx)")
st.sidebar.write("[Contoh DEM dengan format .txt](https://github.com/dzakyw/AutoGrav/raw/9bb43e1559c823350f2371360309d84eaab5ea38/sample_dem.csv)")
st.sidebar.write("[Contoh Koreksi Medan](https://github.com/dzakyw/AutoGrav/raw/9bb43e1559c823350f2371360309d84eaab5ea38/sample_koreksi_medan.csv)")

# ============================================================
# MAIN PROCESSING
# ============================================================
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
            st.success(f"✅ DEM loaded: {len(dem):,} points.")
        except Exception as e:
            st.error(f"DEM load failed: {e}")
            st.stop()
    
    # Validasi data jika debug mode
    if dem is not None and debug_mode:
        st.subheader("Data Validation")
        try:
            xls = pd.ExcelFile(grav)
            df_sample = pd.read_excel(grav, sheet_name=xls.sheet_names[0])
            
            # Konversi koordinat
            E, N, _, _ = latlon_to_utm_redfearn(df_sample["Lat"].to_numpy(), df_sample["Lon"].to_numpy())
            df_sample["Easting"] = E
            df_sample["Northing"] = N
            
            # Tampilkan validasi sederhana
            col1, col2 = st.columns(2)
            with col1:
                st.write("**DEM Stats:**")
                st.write(f"- Points: {len(dem):,}")
                st.write(f"- Elev: {dem['Elev'].min():.1f} to {dem['Elev'].max():.1f} m")
            with col2:
                st.write("**Station Stats:**")
                st.write(f"- Stations: {len(df_sample)}")
                st.write(f"- Elev: {df_sample['Elev'].min():.1f} to {df_sample['Elev'].max():.1f} m")
        except:
            pass
    
    # load manual koreksi jika ada
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
    
    # Container untuk statistics
    tc_stats = []
    
    for sh in xls.sheet_names:
        df = pd.read_excel(grav, sheet_name=sh)
        required = {"Nama","Time","G_read (mGal)","Lat","Lon","Elev"}
        
        if not required.issubset(set(df.columns)):
            st.warning(f"Sheet {sh} dilewati (kolom tidak lengkap).")
            continue
        
        # UTM conversion for stations
        E, N, _, _ = latlon_to_utm_redfearn(df["Lat"].to_numpy(), df["Lon"].to_numpy())
        df["Easting"] = E
        df["Northing"] = N
        
        # Drift correction
        Gmap, D = compute_drift(df, G_base)
        df["G_read (mGal)"] = df["Nama"].map(Gmap)
        
        # Basic corrections
        df["Koreksi Lintang"] = latitude_correction(df["Lat"])
        df["Free Air Correction"] = free_air(df["Elev"])
        df["FAA"] = df["G_read (mGal)"] - df["Koreksi Lintang"] + df["Free Air Correction"]
        
        # ============================================================
        # TERRAIN CORRECTION DENGAN VALIDASI
        # ============================================================
        tc_list = []
        nstations = len(df)
        
        # Parameter OSS yang sudah dioptimasi
        oss_params = {
            'max_radius': max_radius,
            'tolerance_nGal': 1.0,
            'threshold_mGal': threshold_mgal,  # PAKAI VALUE DARI SLIDER
            'theta_step': 1.0,
            'r_step_near': 10.0,
            'r_step_far': 50.0,
            'min_points_per_sector': min_points_sector,
            'use_optimized_elevation': use_optimized_elev,
            'debug': debug_mode
        }
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(nstations):
            station_data = df.iloc[i]
            station_name = station_data['Nama']
            
            # Update progress
            progress = (i + 1) / nstations
            progress_bar.progress(progress)
            status_text.text(f"Sheet {sh}: Station {i+1}/{nstations} ({station_name})")
            
            if method == "OSS (Algorithm from Paper)":
                if dem is not None:
                    tc_val = calculate_oss_correction(dem, station_data, oss_params)
                    
                    # VALIDASI TC VALUE
                    tc_val = validate_tc_value(tc_val, station_name, method, debug_mode)
                    
                    # Bandingkan dengan Hammer jika debug
                    if debug_mode:
                        e0 = float(station_data["Easting"])
                        n0 = float(station_data["Northing"])
                        z0 = float(station_data["Elev"])
                        tc_hammer = hammer_tc(e0, n0, z0, dem)
                        st.write(f"{station_name}: OSS={tc_val:.3f} mGal, Hammer={tc_hammer:.3f} mGal")
                    
                    tc_stats.append(tc_val)
                else:
                    st.error("DEM diperlukan untuk metode OSS.")
                    tc_val = 0.0
            
            elif method == "HAMMER (Legacy)":
                e0 = float(station_data["Easting"])
                n0 = float(station_data["Northing"])
                z0 = float(station_data["Elev"])
                tc_val = hammer_tc(e0, n0, z0, dem) if dem is not None else 0.0
                tc_stats.append(tc_val)
            
            elif method == "NAGY Prism (Reference)":
                e0 = float(station_data["Easting"])
                n0 = float(station_data["Northing"])
                z0 = float(station_data["Elev"])
                if dem is not None:
                    tc_val = simple_nagy_prism(e0, n0, z0, dem, density, max_radius)
                    tc_val = validate_tc_value(tc_val, station_name, method, debug_mode)
                    tc_stats.append(tc_val)
                else:
                    tc_val = 0.0
            
            tc_list.append(tc_val)
        
        progress_bar.empty()
        status_text.empty()
        
        # Tampilkan statistik TC untuk sheet ini jika debug
        if debug_mode and tc_list:
            tc_array = np.array(tc_list)
            st.write(f"**Sheet {sh} TC Stats:**")
            st.write(f"Mean: {tc_array.mean():.3f} mGal, Min: {tc_array.min():.3f} mGal, Max: {tc_array.max():.3f} mGal")
            if tc_array.max() - tc_array.min() > 10:
                st.warning("⚠️ Wide range of TC values in this sheet!")
        
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
    
    # ============================================================
    # TAMPILKAN FINAL STATISTICS
    # ============================================================
    st.success(f"✅ Processing completed in {elapsed:.1f} seconds")
    
    if tc_stats:
        tc_values = np.array(tc_stats)
        st.write(f"**Overall TC Statistics ({method}):**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean TC", f"{tc_values.mean():.3f} mGal")
        with col2:
            st.metric("Min TC", f"{tc_values.min():.3f} mGal")
        with col3:
            st.metric("Max TC", f"{tc_values.max():.3f} mGal")
        with col4:
            st.metric("Std Dev", f"{tc_values.std():.3f} mGal")
        
        # Warning jika range terlalu besar
        if tc_values.max() - tc_values.min() > 15:
            st.warning("⚠️ Very wide range of TC values! Some stations may have issues.")
    
    # slope and Bouguer
    mask = df_all[["X-Parasnis","Y-Parasnis"]].notnull().all(axis=1)
    if mask.sum() >= 2:
        slope, intercept = np.polyfit(df_all.loc[mask,"X-Parasnis"], df_all.loc[mask,"Y-Parasnis"], 1)
        
        # Hitung R-squared
        y_pred = slope * df_all.loc[mask,"X-Parasnis"] + intercept
        y_actual = df_all.loc[mask,"Y-Parasnis"]
        r_squared = 1 - np.sum((y_actual - y_pred)**2) / np.sum((y_actual - np.mean(y_actual))**2)
        
        st.info(f"**Parasnis Regression:** Slope (K) = {slope:.5f}, R² = {r_squared:.3f}")
        
        if r_squared < 0.7:
            st.warning("Low R² value in Parasnis regression! Check TC calculations.")
    else:
        slope = np.nan
    
    df_all["Bouger Correction"] = 0.04192 * slope * df_all["Elev"]
    df_all["Simple Bouger Anomaly"] = df_all["FAA"] - df_all["Bouger Correction"]
    df_all["Complete Bouger Correction"] = df_all["Simple Bouger Anomaly"] + df_all["Koreksi Medan"]
    
    st.success("Done")
    st.dataframe(df_all.head(20))
    
    # contour plots
    st.subheader("Plot Parasnis X–Y")
    
    mask = df_all[["X-Parasnis", "Y-Parasnis"]].notnull().all(axis=1)
    df_parasnis = df_all.loc[mask].copy()
    
    if len(df_parasnis) < 2:
        st.warning("Data tidak cukup untuk regresi Parasnis.")
    else:
        X = df_parasnis["X-Parasnis"].values
        Y = df_parasnis["Y-Parasnis"].values
        
        slope, intercept = np.polyfit(X, Y, 1)
        X_line = np.linspace(min(X), max(X), 100)
        Y_line = slope * X_line + intercept
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X, Y, s=25, color="blue", label="Data Parasnis", alpha=0.7)
        ax.plot(X_line, Y_line, color="red", linewidth=2,
                label=f"Regresi: Y = {slope:.5f} X + {intercept:.5f}")
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
    st.download_button("Download CSV", df_all.to_csv(index=False).encode("utf-8"),
                      "Hasil Perhitungan.csv")



