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

# -----------------------
# Nagy prism — improved
# -----------------------
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt

# ============================================================
# KONSTANTA & KERNEL DASAR SESUAI PAPER
# ============================================================
G = 6.67430e-11           # m^3 kg^-1 s^-2
RHO = 2670.0              # kg/m^3 (2.67 g/cm^3)
NANO_TO_MGAL = 1e-6       # 1 nGal = 1e-6 mGal
MICRO_TO_MGAL = 1e-3      # 1 μGal = 1e-3 mGal

def terrain_effect_cylindrical_sector(R1, R2, theta1, theta2, z):
    """
    PERSAMAAN (1) dari paper: Δg_T = GρΔθ[R2-R1+√(R1²+z²)-√(R2²+z²)]
    
    Parameters:
    -----------
    R1, R2 : float
        Inner and outer radius (meters)
    theta1, theta2 : float
        Azimuth angles in degrees
    z : float
        Height difference: station_elevation - sector_average_elevation (meters)
    
    Returns:
    --------
    delta_g : float
        Terrain effect in mGal
    """
    # Convert angles to radians
    theta1_rad = np.radians(theta1)
    theta2_rad = np.radians(theta2)
    Delta_theta = theta2_rad - theta1_rad
    
    # Calculate terrain effect in SI units (m/s²)
    term = (R2 - R1) + np.sqrt(R1**2 + z**2) - np.sqrt(R2**2 + z**2)
    delta_g_si = G * RHO * Delta_theta * term
    
    # Convert to mGal (1 mGal = 10^-5 m/s²)
    return delta_g_si * 1e5

def optimized_elevation(z_avg, deviations, R1, R2, Delta_theta, r_points):
    """
    PERSAMAAN (8) dari paper: Optimized elevation z'
    z'² = z² + sgn(l) * (R2R1/(R2-R1)) * (2zl + l²) / (Δθ * 2r³)
    
    Parameters:
    -----------
    z_avg : float
        Average elevation difference in sector
    deviations : array
        l values: deviations of each point from z_avg
    R1, R2 : float
        Inner and outer radii of sector
    Delta_theta : float
        Angular width in radians
    r_points : array
        Radial distances of each point
    
    Returns:
    --------
    z_opt : float
        Optimized elevation for the sector
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
        if r < 1e-10:  # Avoid division by zero
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

class OSSTerrainCorrector:
    """
    Implementasi lengkap Optimally Selecting Sectors (OSS) algorithm
    sesuai dengan paper: DOI 10.1007/s11200-019-0273-0
    """
    
    def __init__(self, dem_df, station_coords, params=None):
        """
        Initialize OSS corrector.
        
        Parameters:
        -----------
        dem_df : pandas.DataFrame
            DEM data with columns ['Easting', 'Northing', 'Elev']
        station_coords : tuple
            (easting, northing, elevation) of gravity station
        params : dict
            Algorithm parameters (see set_parameters())
        """
        self.dem_df = dem_df.copy()
        self.e0, self.n0, self.z0 = station_coords
        
        # Default parameters from paper
        self.params = {
            'max_radius': 4500.0,        # meters (4.5 km as in paper)
            'tolerance_nGal': 1.0,       # 1 nGal tolerance
            'threshold_mGal': 1.0,       # 1 mGal subdivision threshold
            'theta_step': 1.0,           # degree
            'r_step_near': 10.0,         # meters for near zone
            'r_step_far': 50.0,          # meters for far zone
            'min_points_per_sector': 10,
            'use_optimized_elevation': True,
            'debug': False
        }
        
        if params:
            self.params.update(params)
        
        # Convert DEM to polar coordinates relative to station
        self._prepare_polar_coords()
        
    def _prepare_polar_coords(self):
        """Convert DEM to polar coordinates relative to station."""
        dx = self.dem_df['Easting'] - self.e0
        dy = self.dem_df['Northing'] - self.n0
        
        self.r = np.sqrt(dx**2 + dy**2)
        self.theta_rad = np.arctan2(dy, dx)
        self.theta_deg = np.degrees(self.theta_rad) % 360.0
        self.z_rel = self.dem_df['Elev'] - self.z0
        
        # Filter by max radius
        mask = self.r <= self.params['max_radius']
        self.r_filtered = self.r[mask].values
        self.theta_filtered = self.theta_deg[mask].values
        self.z_rel_filtered = self.z_rel[mask].values
        
    def _find_turning_points_theta(self, theta1, theta2, R1, R2):
        """
        Find turning points in angular direction (first approach).
        Returns list of angles where significant changes occur.
        """
        theta_step = self.params['theta_step']
        tolerance = self.params['tolerance_nGal'] * NANO_TO_MGAL
        
        angles = []
        terrain_values = []
        
        # Sample terrain effect at different θ_m
        theta_current = theta1
        while theta_current <= theta2:
            # Calculate terrain for two sub-sectors
            mask_left = (self.theta_filtered >= theta1) & (self.theta_filtered < theta_current)
            mask_right = (self.theta_filtered >= theta_current) & (self.theta_filtered <= theta2)
            
            mask_left_full = mask_left & (self.r_filtered >= R1) & (self.r_filtered <= R2)
            mask_right_full = mask_right & (self.r_filtered >= R1) & (self.r_filtered <= R2)
            
            terrain_left = 0.0
            if np.any(mask_left_full):
                z_avg_left = np.mean(self.z_rel_filtered[mask_left_full])
                terrain_left = terrain_effect_cylindrical_sector(
                    R1, R2, theta1, theta_current, z_avg_left
                )
            
            terrain_right = 0.0
            if np.any(mask_right_full):
                z_avg_right = np.mean(self.z_rel_filtered[mask_right_full])
                terrain_right = terrain_effect_cylindrical_sector(
                    R1, R2, theta_current, theta2, z_avg_right
                )
            
            total_terrain = terrain_left + terrain_right
            
            angles.append(theta_current)
            terrain_values.append(total_terrain)
            
            theta_current += theta_step
        
        # Detect turning points (significant changes in slope)
        turning_points = []
        if len(terrain_values) > 2:
            # Calculate second differences
            for i in range(1, len(terrain_values)-1):
                diff1 = terrain_values[i] - terrain_values[i-1]
                diff2 = terrain_values[i+1] - terrain_values[i]
                
                # Check for significant change in slope
                if abs(diff2 - diff1) > tolerance:
                    turning_points.append(angles[i])
        
        return turning_points
    
    def _find_turning_points_radius(self, theta1, theta2, R1, R2):
        """
        Find turning points in radial direction (second approach).
        Returns list of radii where significant changes occur.
        """
        # Adaptive step size: smaller near station, larger far away
        r_step = self.params['r_step_near'] if R2 < 1000 else self.params['r_step_far']
        tolerance = self.params['tolerance_nGal'] * NANO_TO_MGAL
        
        radii = []
        terrain_values = []
        
        R_current = R1
        while R_current <= R2:
            # Calculate terrain for two annular sub-sectors
            mask_inner = (self.r_filtered >= R1) & (self.r_filtered < R_current)
            mask_outer = (self.r_filtered >= R_current) & (self.r_filtered <= R2)
            
            mask_inner_full = mask_inner & (self.theta_filtered >= theta1) & (self.theta_filtered <= theta2)
            mask_outer_full = mask_outer & (self.theta_filtered >= theta1) & (self.theta_filtered <= theta2)
            
            terrain_inner = 0.0
            if np.any(mask_inner_full):
                z_avg_inner = np.mean(self.z_rel_filtered[mask_inner_full])
                terrain_inner = terrain_effect_cylindrical_sector(
                    R1, R_current, theta1, theta2, z_avg_inner
                )
            
            terrain_outer = 0.0
            if np.any(mask_outer_full):
                z_avg_outer = np.mean(self.z_rel_filtered[mask_outer_full])
                terrain_outer = terrain_effect_cylindrical_sector(
                    R_current, R2, theta1, theta2, z_avg_outer
                )
            
            total_terrain = terrain_inner + terrain_outer
            
            radii.append(R_current)
            terrain_values.append(total_terrain)
            
            R_current += r_step
        
        # Detect turning points
        turning_points = []
        if len(terrain_values) > 2:
            for i in range(1, len(terrain_values)-1):
                diff1 = terrain_values[i] - terrain_values[i-1]
                diff2 = terrain_values[i+1] - terrain_values[i]
                
                if abs(diff2 - diff1) > tolerance:
                    turning_points.append(radii[i])
        
        return turning_points
    
    def _process_sector(self, theta1, theta2, R1, R2, depth=0):
        """
        Recursive sector processing as per Fig. 4 flowchart.
        
        Returns:
        --------
        total_terrain : float
            Terrain effect for this sector
        subsectors : list
            List of final subsectors with their properties
        """
        threshold = self.params['threshold_mGal']
        
        # Get data in current sector
        mask = (self.theta_filtered >= theta1) & (self.theta_filtered <= theta2) & \
               (self.r_filtered >= R1) & (self.r_filtered <= R2)
        
        if not np.any(mask):
            return 0.0, []
        
        z_sector = self.z_rel_filtered[mask]
        r_sector = self.r_filtered[mask]
        
        if len(z_sector) < self.params['min_points_per_sector']:
            # Too few points, use average elevation
            z_avg = np.mean(z_sector) if len(z_sector) > 0 else 0.0
            terrain = terrain_effect_cylindrical_sector(R1, R2, theta1, theta2, z_avg)
            return terrain, [(theta1, theta2, R1, R2, z_avg, terrain)]
        
        # Calculate terrain with average elevation
        z_avg = np.mean(z_sector)
        terrain_avg = terrain_effect_cylindrical_sector(R1, R2, theta1, theta2, z_avg)
        
        # Check if sector needs subdivision
        if abs(terrain_avg) < threshold:
            # Small effect, no subdivision needed
            if self.params['use_optimized_elevation']:
                # Calculate optimized elevation
                Delta_theta = np.radians(theta2 - theta1)
                deviations = z_sector - z_avg
                z_opt = optimized_elevation(z_avg, deviations, R1, R2, Delta_theta, r_sector)
                terrain_final = terrain_effect_cylindrical_sector(R1, R2, theta1, theta2, z_opt)
            else:
                terrain_final = terrain_avg
                z_opt = z_avg
            
            return terrain_final, [(theta1, theta2, R1, R2, z_opt, terrain_final)]
        
        # Sector needs subdivision - try angular division first
        turning_points_theta = self._find_turning_points_theta(theta1, theta2, R1, R2)
        
        if turning_points_theta:
            # Divide by angles
            total_terrain = 0.0
            all_subsectors = []
            
            angles = [theta1] + sorted(turning_points_theta) + [theta2]
            for i in range(len(angles)-1):
                sub_theta1, sub_theta2 = angles[i], angles[i+1]
                
                # Check if angular subsector needs radial division
                sub_mask = (self.theta_filtered >= sub_theta1) & (self.theta_filtered <= sub_theta2) & \
                          (self.r_filtered >= R1) & (self.r_filtered <= R2)
                
                if np.any(sub_mask):
                    z_sub = self.z_rel_filtered[sub_mask]
                    z_avg_sub = np.mean(z_sub)
                    terrain_sub = terrain_effect_cylindrical_sector(R1, R2, sub_theta1, sub_theta2, z_avg_sub)
                    
                    if abs(terrain_sub) >= threshold:
                        # Needs radial division
                        turning_points_r = self._find_turning_points_radius(sub_theta1, sub_theta2, R1, R2)
                        
                        if turning_points_r:
                            # Divide radially
                            radii = [R1] + sorted(turning_points_r) + [R2]
                            for j in range(len(radii)-1):
                                sub_R1, sub_R2 = radii[j], radii[j+1]
                                sub_terrain, sub_subsectors = self._process_sector(
                                    sub_theta1, sub_theta2, sub_R1, sub_R2, depth+1
                                )
                                total_terrain += sub_terrain
                                all_subsectors.extend(sub_subsectors)
                        else:
                            # No radial division needed
                            if self.params['use_optimized_elevation']:
                                r_sub = self.r_filtered[sub_mask]
                                Delta_theta = np.radians(sub_theta2 - sub_theta1)
                                deviations = z_sub - z_avg_sub
                                z_opt = optimized_elevation(z_avg_sub, deviations, R1, R2, Delta_theta, r_sub)
                                terrain_final = terrain_effect_cylindrical_sector(
                                    R1, R2, sub_theta1, sub_theta2, z_opt
                                )
                            else:
                                terrain_final = terrain_sub
                                z_opt = z_avg_sub
                            
                            total_terrain += terrain_final
                            all_subsectors.append((sub_theta1, sub_theta2, R1, R2, z_opt, terrain_final))
                    else:
                        # Angular subsector is small enough
                        if self.params['use_optimized_elevation']:
                            r_sub = self.r_filtered[sub_mask]
                            Delta_theta = np.radians(sub_theta2 - sub_theta1)
                            deviations = z_sub - z_avg_sub
                            z_opt = optimized_elevation(z_avg_sub, deviations, R1, R2, Delta_theta, r_sub)
                            terrain_final = terrain_effect_cylindrical_sector(
                                R1, R2, sub_theta1, sub_theta2, z_opt
                            )
                        else:
                            terrain_final = terrain_sub
                            z_opt = z_avg_sub
                        
                        total_terrain += terrain_final
                        all_subsectors.append((sub_theta1, sub_theta2, R1, R2, z_opt, terrain_final))
            
            return total_terrain, all_subsectors
        
        else:
            # No angular turning points, try radial division
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
                # No subdivision possible, use optimized elevation
                if self.params['use_optimized_elevation']:
                    Delta_theta = np.radians(theta2 - theta1)
                    deviations = z_sector - z_avg
                    z_opt = optimized_elevation(z_avg, deviations, R1, R2, Delta_theta, r_sector)
                    terrain_final = terrain_effect_cylindrical_sector(R1, R2, theta1, theta2, z_opt)
                else:
                    terrain_final = terrain_avg
                    z_opt = z_avg
                
                return terrain_final, [(theta1, theta2, R1, R2, z_opt, terrain_final)]
    
    def calculate_terrain_correction(self):
        """
        Main method to calculate terrain correction using OSS algorithm.
        
        Returns:
        --------
        total_tc : float
            Total terrain correction in mGal
        subsectors : list
            List of all optimally selected subsectors
        """
        if self.params['debug']:
            print(f"Starting OSS calculation for station at ({self.e0:.1f}, {self.n0:.1f})")
            print(f"DEM points in radius: {len(self.r_filtered)}")
        
        # Start with full circle and maximum radius
        total_tc, subsectors = self._process_sector(
            theta1=0.0,
            theta2=360.0,
            R1=0.0,
            R2=self.params['max_radius']
        )
        
        if self.params['debug']:
            print(f"Total TC: {total_tc:.6f} mGal")
            print(f"Number of subsectors: {len(subsectors)}")
            
            # Print largest subsectors
            subsectors_sorted = sorted(subsectors, key=lambda x: abs(x[5]), reverse=True)[:5]
            print("\nTop 5 subsectors:")
            for i, (t1, t2, r1, r2, z, tc) in enumerate(subsectors_sorted):
                print(f"  {i+1}: θ={t1:.1f}-{t2:.1f}°, r={r1:.0f}-{r2:.0f}m, z={z:.1f}m, Δg={tc:.6f}mGal")
        
        return total_tc, subsectors
    
    def visualize_sectors(self, subsectors, max_sectors=50):
        """
        Visualize the optimally selected sectors (polar plot).
        
        Parameters:
        -----------
        subsectors : list
            Output from calculate_terrain_correction()
        max_sectors : int
            Maximum number of sectors to plot (for clarity)
        """
        if not subsectors:
            print("No subsectors to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        # Plot only the largest sectors if there are too many
        if len(subsectors) > max_sectors:
            subsectors_to_plot = sorted(subsectors, key=lambda x: abs(x[5]), reverse=True)[:max_sectors]
        else:
            subsectors_to_plot = subsectors
        
        # Color by terrain effect magnitude
        tc_values = [s[5] for s in subsectors_to_plot]
        abs_tc = np.abs(tc_values)
        norm = plt.Normalize(min(abs_tc), max(abs_tc))
        cmap = plt.cm.viridis
        
        for i, (theta1, theta2, r1, r2, z, tc) in enumerate(subsectors_to_plot):
            # Convert to radians
            theta1_rad = np.radians(theta1)
            theta2_rad = np.radians(theta2)
            
            # Create polygon for sector
            theta_poly = [theta1_rad, theta1_rad, theta2_rad, theta2_rad]
            r_poly = [r1, r2, r2, r1]
            
            # Plot sector
            color = cmap(norm(abs(tc)))
            ax.fill(theta_poly, r_poly, alpha=0.6, color=color, edgecolor='black', linewidth=0.5)
        
        ax.set_title(f"OSS Selected Sectors\nStation: ({self.e0:.0f}, {self.n0:.0f})", pad=20)
        ax.set_xlabel("Radius (m)")
        ax.grid(True)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.1)
        cbar.set_label('|Terrain Effect| (mGal)')
        
        plt.tight_layout()
        return fig

# ============================================================
# FUNGSI INTEGRASI DENGAN STREAMLIT
# ============================================================
def calculate_oss_correction(dem_df, station_row, params=None):
    """
    Wrapper function for Streamlit integration.
    
    Parameters:
    -----------
    dem_df : pandas.DataFrame
        DEM data
    station_row : pandas.Series
        Row containing station data (must have Easting, Northing, Elev)
    params : dict
        OSS algorithm parameters
    
    Returns:
    --------
    tc_value : float
        Terrain correction value in mGal
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
density = st.sidebar.number_input("Densitas Koreksi Medan (kg/m³)", value=2670.0, step=10.0, format="%.1f")
max_radius = st.sidebar.number_input("Jarak Maksimum (m) untuk Nagy", value=10000, step=1000)
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
        # TERRAIN CORRECTION — INSIDE SHEET LOOP
        # -----------------------------------------
        # GANTI BLOK INI (sekitar baris 395-415):
        # tc_list = []
        # nstations = len(df)
        # for i in range(nstations):
        #     e0 = float(df.iloc[i]["Easting"])
        #     n0 = float(df.iloc[i]["Northing"])
        #     z0 = float(df.iloc[i]["Elev"])
        #     
        #     if method.startswith("NAGY"):
        #         tc_val, diag = compute_nagy_tc_debug(...)
        #     else:
        #         tc_val = hammer_tc(e0, n0, z0, dem)
        #     
        #     tc_list.append(tc_val)
        
        # DENGAN INI:
        tc_list = []
        nstations = len(df)
        
        # Parameter OSS (sesuai paper)
        oss_params = {
            'max_radius': max_radius,  # dari sidebar
            'tolerance_nGal': 1.0,     # 1 nGal seperti di paper
            'threshold_mGal': 1.0,     # 1 mGal threshold
            'theta_step': 1.0,         # 1 derajat
            'r_step_near': 10.0,       # 10 m untuk zona dekat
            'r_step_far': 50.0,        # 50 m untuk zona jauh
            'use_optimized_elevation': True,
            'debug': False  # Set True untuk output debugging
        }
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(nstations):
            station_data = df.iloc[i]
            
            # Update progress
            progress = (i + 1) / nstations
            progress_bar.progress(progress)
            status_text.text(f"Processing station {i+1}/{nstations}...")
            
            if method == "OSS (Algorithm from Paper)":
                if dem is not None:
                    tc_val = calculate_oss_correction(dem, station_data, oss_params)
                else:
                    st.error("DEM diperlukan untuk metode OSS.")
                    tc_val = 0.0
            elif method == "HAMMER (Legacy)":
                e0 = float(station_data["Easting"])
                n0 = float(station_data["Northing"])
                z0 = float(station_data["Elev"])
                tc_val = hammer_tc(e0, n0, z0, dem) if dem is not None else 0.0
            else:  # "NAGY Prism (Reference)"
                e0 = float(station_data["Easting"])
                n0 = float(station_data["Northing"])
                z0 = float(station_data["Elev"])
                if dem is not None:
                    tc_val, _ = compute_nagy_tc_debug(e0, n0, z0, dem, density, max_radius, debug=False)
                else:
                    tc_val = 0.0
            
            tc_list.append(tc_val)
        
        progress_bar.empty()
        status_text.empty()
        
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

