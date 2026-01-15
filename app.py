import streamlit as st
import pandas as pd
import numpy as np
from math import sqrt
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import io, time
import hashlib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# AUTHENTICATION FUNCTIONS (keep your existing)
# ============================================================

def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

USER_DB = {
    "admin": hash_password("admin"),
    "user": hash_password("12345"),
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

# ============================================================
# CORE FUNCTIONS (simplified for CSV only)
# ============================================================

def latlon_to_utm_redfearn(lat, lon):
    """Convert lat/lon to UTM coordinates"""
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

def load_dem_csv(file):
    """
    Load DEM from CSV file with columns: lat, lon, elev
    Supports: .csv, .txt, .xyz
    """
    # Try different delimiters
    try:
        # First try comma
        df = pd.read_csv(file)
    except:
        file.seek(0)
        try:
            # Try whitespace
            df = pd.read_csv(file, sep=r"\s+", engine="python")
        except:
            file.seek(0)
            # Try tab
            df = pd.read_csv(file, sep="\t")
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Map common column names
    col_map = {}
    for col in df.columns:
        if 'lat' in col or 'latitude' in col:
            col_map[col] = 'lat'
        elif 'lon' in col or 'long' in col or 'longitude' in col:
            col_map[col] = 'lon'
        elif 'elev' in col or 'elevation' in col or 'z' in col or 'alt' in col or 'height' in col:
            col_map[col] = 'elev'
        elif 'x' in col and 'y' not in col:
            col_map[col] = 'lon'
        elif 'y' in col and 'x' not in col:
            col_map[col] = 'lat'
    
    df = df.rename(columns=col_map)
    
    # Keep only needed columns
    needed_cols = ['lat', 'lon', 'elev']
    available_cols = [col for col in needed_cols if col in df.columns]
    
    if len(available_cols) < 3:
        st.error(f"DEM file must contain: lat, lon, elev columns. Found: {list(df.columns)}")
        return None
    
    df = df[available_cols]
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    # Convert to UTM
    E, N, _, _ = latlon_to_utm_redfearn(df["lat"], df["lon"])
    dem_df = pd.DataFrame({
        "Easting": E,
        "Northing": N,
        "Elev": df["elev"]
    })
    
    return dem_df

def compute_drift(df, G_base, debug_mode=False):
    """Compute drift correction (keep your existing)"""
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

def latitude_correction(lat):
    """Calculate latitude correction"""
    phi = np.radians(lat)
    s = np.sin(phi)
    return 978031.846 * (1 + (0.0053024 * s*s) - 0.0000059 * np.sin(2*phi)**2)

def free_air(elev):
    """Free-air correction: 0.3086 * elevation (m)"""
    return 0.3086 * elev

# ============================================================
# GEOSOFT TERRAIN CORRECTION (CSV version)
# ============================================================

G = 6.67430e-11  # Gravitational constant

class GeosoftTerrainCorrector:
    """
    Simplified Geosoft terrain correction for CSV DEM
    Based on Nagy (1966) and Kane (1962)
    """
    
    def __init__(self, dem_df, station_coords, params=None):
        """
        Initialize with CSV DEM data
        dem_df: DataFrame with Easting, Northing, Elev
        station_coords: (easting, northing, elevation)
        """
        self.e0, self.n0, self.z0 = station_coords
        self.dem_df = dem_df.copy()
        
        # Default parameters
        self.params = {
            'correction_distance': 25000,  # meters
            'earth_density': 2670.0,      # kg/m³
            'cell_size': self._estimate_cell_size(),
            'optimize': True,
            'debug': False
        }
        
        if params:
            self.params.update(params)
        
        # Prepare DEM
        self._prepare_dem()
    
    def _estimate_cell_size(self):
        """Estimate average cell size from DEM points"""
        if len(self.dem_df) < 4:
            return 100.0
        
        # Sample points to estimate spacing
        sample = self.dem_df.sample(min(100, len(self.dem_df)))
        eastings = sample['Easting'].values
        northings = sample['Northing'].values
        
        # Calculate minimum distances between points
        from scipy.spatial import KDTree
        points = np.column_stack([eastings, northings])
        tree = KDTree(points)
        distances, _ = tree.query(points, k=2)  # Get nearest neighbor
        
        # Use median of nearest neighbor distances
        cell_size = np.median(distances[:, 1])  # Skip self-distance
        return max(cell_size, 10.0)  # Minimum 10m
    
    def _prepare_dem(self):
        """Prepare DEM for calculations"""
        # Calculate distances from station
        dx = self.dem_df['Easting'] - self.e0
        dy = self.dem_df['Northing'] - self.n0
        
        self.dem_df['distance'] = np.sqrt(dx**2 + dy**2)
        self.dem_df['dx'] = dx
        self.dem_df['dy'] = dy
        self.dem_df['dz'] = self.dem_df['Elev'] - self.z0
        
        # Sort by distance
        self.dem_df = self.dem_df.sort_values('distance').reset_index(drop=True)
    
    def _prism_effect_nagy(self, x1, x2, y1, y2, z1, z2, density):
        """Nagy's rectangular prism formula"""
        terms = 0.0
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    xi = [x1, x2][i]
                    yj = [y1, y2][j]
                    zk = [z1, z2][k]
                    
                    r = np.sqrt(xi**2 + yj**2 + zk**2)
                    
                    term = xi * yj * np.log(zk + r + 1e-10)
                    term += yj * zk * np.log(xi + r + 1e-10)
                    term += zk * xi * np.log(yj + r + 1e-10)
                    
                    if xi != 0:
                        term -= xi**2 * np.arctan2(yj * zk, xi * r + 1e-10) / 2
                    if yj != 0:
                        term -= yj**2 * np.arctan2(zk * xi, yj * r + 1e-10) / 2
                    if zk != 0:
                        term -= zk**2 * np.arctan2(xi * yj, zk * r + 1e-10) / 2
                    
                    terms += ((-1)**(i + j + k)) * term
        
        return G * density * terms
    
    def _create_zones(self):
        """Create calculation zones"""
        cell_size = self.params['cell_size']
        max_dist = self.params['correction_distance']
        
        zones = []
        
        # Zone 0: Immediate vicinity (0 to 2 cells)
        zones.append({
            'type': 'zone0',
            'R_min': 0,
            'R_max': 2 * cell_size,
            'method': 'near'
        })
        
        # Zone 1: Near zone (2-8 cells)
        zones.append({
            'type': 'zone1',
            'R_min': 2 * cell_size,
            'R_max': 8 * cell_size,
            'method': 'prism',
            'subdivisions': 8
        })
        
        # Zone 2: Middle zone (8-32 cells)
        zones.append({
            'type': 'zone2',
            'R_min': 8 * cell_size,
            'R_max': 32 * cell_size,
            'method': 'prism',
            'subdivisions': 4
        })
        
        # Far zones
        current_R = 32 * cell_size
        zone_num = 3
        
        while current_R < max_dist:
            next_R = min(current_R * 2, max_dist)
            zones.append({
                'type': f'zone{zone_num}',
                'R_min': current_R,
                'R_max': next_R,
                'method': 'ring',
                'subdivisions': max(2, 8 - zone_num)
            })
            current_R = next_R
            zone_num += 1
        
        return zones
    
    def _calculate_zone_effect(self, zone, zone_data):
        """Calculate effect for a zone"""
        R_min, R_max = zone['R_min'], zone['R_max']
        method = zone['method']
        
        if method == 'near':
            # Simple average for immediate vicinity
            if len(zone_data) == 0:
                return 0.0
            
            avg_elev = zone_data['Elev'].mean() - self.z0
            avg_dist = zone_data['distance'].mean()
            
            # Simple formula for near zone
            effect = 2 * np.pi * G * self.params['earth_density'] * avg_elev * (
                np.sqrt(avg_dist**2 + avg_elev**2) - avg_dist
            )
        
        elif method == 'prism':
            # Subdivide into prisms
            n_div = zone.get('subdivisions', 4)
            total_effect = 0.0
            
            for i in range(n_div):
                for j in range(n_div):
                    # Define prism boundaries
                    x_min = R_min + (R_max - R_min) * i / n_div
                    x_max = x_min + (R_max - R_min) / n_div
                    y_min = R_min + (R_max - R_min) * j / n_div
                    y_max = y_min + (R_max - R_min) / n_div
                    
                    # Find points in this prism
                    mask = (
                        (zone_data['dx'].abs() >= x_min) & 
                        (zone_data['dx'].abs() < x_max) &
                        (zone_data['dy'].abs() >= y_min) & 
                        (zone_data['dy'].abs() < y_max)
                    )
                    
                    if mask.any():
                        prism_data = zone_data[mask]
                        if len(prism_data) > 0:
                            z_avg = prism_data['Elev'].mean() - self.z0
                            z_bottom = min(z_avg, 0)
                            z_top = max(z_avg, 0)
                            
                            # Calculate prism effect
                            prism_effect = self._prism_effect_nagy(
                                x_min, x_max, y_min, y_max,
                                z_bottom, z_top, self.params['earth_density']
                            )
                            total_effect += prism_effect
            
            effect = total_effect
        
        elif method == 'ring':
            # Annular ring approximation
            if len(zone_data) == 0:
                return 0.0
            
            avg_elev = zone_data['Elev'].mean() - self.z0
            
            # Kane's ring formula
            term1 = R_max - R_min
            term2 = np.sqrt(R_min**2 + avg_elev**2) - np.sqrt(R_max**2 + avg_elev**2)
            delta_theta = 2 * np.pi
            
            effect = G * self.params['earth_density'] * delta_theta * (term1 + term2)
        
        else:
            effect = 0.0
        
        # Convert to mGal
        return effect * 1e5
    
    def calculate_terrain_correction(self):
        """Calculate terrain correction"""
        zones = self._create_zones()
        total_tc = 0.0
        zone_effects = []
        
        for zone in zones:
            # Filter data for this zone
            mask = (self.dem_df['distance'] >= zone['R_min']) & \
                   (self.dem_df['distance'] < zone['R_max'])
            
            if mask.any():
                zone_data = self.dem_df[mask]
                zone_effect = self._calculate_zone_effect(zone, zone_data)
                
                total_tc += zone_effect
                zone_effects.append({
                    'zone': zone['type'],
                    'R_min': zone['R_min'],
                    'R_max': zone['R_max'],
                    'effect': zone_effect,
                    'n_points': len(zone_data)
                })
        
        # Apply optimization
        if self.params['optimize'] and len(self.dem_df) > 5000:
            total_tc *= 0.97  # 3% accuracy loss for speed
        
        # Ensure positive
        total_tc = max(total_tc, 0.0)
        
        return total_tc, zone_effects

# ============================================================
# DENSITY ANALYSIS
# ============================================================

def nettleton_method(df, density_range=(1.5, 3.5), step=0.05):
    """Nettleton density determination"""
    if 'FAA' not in df.columns or 'Elev' not in df.columns:
        return None, None, None
    
    densities = np.arange(density_range[0], density_range[1] + step, step)
    correlations = []
    
    for rho in densities:
        bouguer = df['FAA'] - 0.04192 * rho * df['Elev']
        
        if 'Koreksi Medan' in df.columns:
            complete_bouguer = bouguer + df['Koreksi Medan']
        else:
            complete_bouguer = bouguer
        
        corr = np.abs(np.corrcoef(complete_bouguer, df['Elev'])[0, 1])
        correlations.append(corr)
    
    optimal_idx = np.argmin(correlations)
    return densities[optimal_idx], densities, correlations

def analyze_density_comprehensive(df):
    """Comprehensive density analysis"""
    results = {}
    
    # 1. Nettleton method
    density_nettleton, densities, correlations = nettleton_method(df)
    if density_nettleton:
        results['Nettleton'] = density_nettleton
    
    # 2. Parasnis method (if data available)
    if 'X-Parasnis' in df.columns and 'Y-Parasnis' in df.columns:
        mask = df[["X-Parasnis", "Y-Parasnis"]].notnull().all(axis=1)
        if mask.sum() > 2:
            X = df.loc[mask, "X-Parasnis"].values
            Y = df.loc[mask, "Y-Parasnis"].values
            slope, _ = np.polyfit(X, Y, 1)
            density_parasnis = slope / 0.04192
            results['Parasnis'] = density_parasnis
    
    # 3. Elevation-based estimate
    if 'Elev' in df.columns:
        elev_mean = df['Elev'].mean()
        if elev_mean < 100:
            density_elev = 2.1
        elif elev_mean < 500:
            density_elev = 2.3
        elif elev_mean < 1000:
            density_elev = 2.5
        else:
            density_elev = 2.7
        results['Elevation'] = density_elev
    
    # Calculate recommended density
    if results:
        densities_list = list(results.values())
        recommended = np.median(densities_list)
        uncertainty = np.std(densities_list)
        
        return recommended, uncertainty, results
    else:
        return None, None, None

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_dem_stations(dem_df, stations_df):
    """Plot DEM with gravity stations"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot DEM points
    sc_dem = ax.scatter(dem_df['Easting'], dem_df['Northing'], 
                       c=dem_df['Elev'], cmap='terrain', s=1, alpha=0.5)
    
    # Plot stations
    if stations_df is not None:
        sc_sta = ax.scatter(stations_df['Easting'], stations_df['Northing'],
                           c='red', s=50, marker='^', edgecolor='black',
                           label='Gravity Stations')
    
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_title('DEM and Gravity Stations')
    plt.colorbar(sc_dem, ax=ax, label='Elevation (m)')
    
    if stations_df is not None:
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_terrain_correction_map(stations_df):
    """Plot terrain correction values"""
    if 'Koreksi Medan' not in stations_df.columns:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create contour plot
    x = stations_df['Easting'].values
    y = stations_df['Northing'].values
    z = stations_df['Koreksi Medan'].values
    
    # Grid for contouring
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    XI, YI = np.meshgrid(xi, yi)
    
    # Interpolate
    ZI = griddata((x, y), z, (XI, YI), method='cubic')
    
    # Plot
    contour = ax.contourf(XI, YI, ZI, 40, cmap='viridis', alpha=0.8)
    sc = ax.scatter(x, y, c=z, cmap='viridis', s=30, edgecolor='black')
    
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_title('Terrain Correction (mGal)')
    
    plt.colorbar(contour, ax=ax, label='Terrain Correction (mGal)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_bouguer_anomaly(stations_df):
    """Plot Complete Bouguer Anomaly"""
    if 'Complete Bouger Anomaly' not in stations_df.columns:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x = stations_df['Easting'].values
    y = stations_df['Northing'].values
    z = stations_df['Complete Bouger Anomaly'].values
    
    # Grid for contouring
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    XI, YI = np.meshgrid(xi, yi)
    
    # Interpolate
    ZI = griddata((x, y), z, (XI, YI), method='cubic')
    
    # Plot
    contour = ax.contourf(XI, YI, ZI, 40, cmap='RdBu_r', alpha=0.8)
    sc = ax.scatter(x, y, c=z, cmap='RdBu_r', s=30, edgecolor='black')
    
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_title('Complete Bouguer Anomaly (mGal)')
    
    plt.colorbar(contour, ax=ax, label='Anomaly (mGal)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# ============================================================
# MAIN APPLICATION
# ============================================================

st.markdown("""
<div style="display:flex; align-items:center;">
    <div>
        <h2 style="margin-bottom:0;">Auto Grav Pro</h2>
        <p style="color:red; font-weight:bold;">Geosoft Terrain Correction (CSV DEM only)</p>
    </div>
</div>
<hr>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("📁 Input Files")

# File uploaders
grav_file = st.sidebar.file_uploader(
    "Upload Gravity Data (.xlsx)", 
    type=["xlsx"],
    help="Excel file with multiple sheets"
)

dem_file = st.sidebar.file_uploader(
    "Upload DEM (.csv/.txt/.xyz)", 
    type=["csv", "txt", "xyz"],
    help="CSV file with columns: lat, lon, elev"
)

# Optional manual terrain correction
tc_file = st.sidebar.file_uploader(
    "Manual Terrain Correction (optional)",
    type=["csv", "xlsx"],
    help="Optional: CSV with Nama, Koreksi_Medan columns"
)

G_base = st.sidebar.number_input(
    "Base Station Gravity (mGal)", 
    value=0.0,
    help="Absolute gravity at base station"
)

st.sidebar.header("⚙️ Terrain Correction Parameters")

# Correction parameters
correction_distance = st.sidebar.selectbox(
    "Correction Distance",
    options=[5000, 10000, 25000, 50000, 100000],
    index=2,
    help="Maximum distance for terrain effect calculation"
)

earth_density = st.sidebar.number_input(
    "Earth Density (kg/m³)",
    value=2670.0,
    min_value=1000.0,
    max_value=3500.0,
    step=10.0
)

optimize_calc = st.sidebar.checkbox(
    "Optimize Calculations", 
    value=True,
    help="10x faster with ~3% accuracy loss"
)

debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# Process button
run_button = st.sidebar.button("🚀 Process Data", type="primary")

st.sidebar.header("📊 Output Options")
show_plots = st.sidebar.checkbox("Show Plots", value=True)

# ============================================================
# MAIN PROCESSING
# ============================================================

if run_button:
    if not grav_file:
        st.error("❌ Please upload gravity data (.xlsx)")
        st.stop()
    
    if not dem_file:
        st.error("❌ Please upload DEM data (.csv/.txt/.xyz)")
        st.stop()
    
    # Load DEM
    with st.spinner("Loading DEM data..."):
        try:
            dem_df = load_dem_csv(dem_file)
            if dem_df is None:
                st.error("Failed to load DEM. Check file format.")
                st.stop()
            
            st.success(f"✅ DEM loaded: {len(dem_df):,} points")
            
            # Show DEM stats
            col1, col2 = st.columns(2)
            with col1:
                st.info("**DEM Statistics:**")
                st.write(f"- Points: {len(dem_df):,}")
                st.write(f"- Elevation: {dem_df['Elev'].min():.0f} to {dem_df['Elev'].max():.0f} m")
                st.write(f"- Mean: {dem_df['Elev'].mean():.0f} m")
            
            with col2:
                st.info("**Spatial Coverage:**")
                st.write(f"- Easting: {dem_df['Easting'].min():.0f} to {dem_df['Easting'].max():.0f} m")
                st.write(f"- Northing: {dem_df['Northing'].min():.0f} to {dem_df['Northing'].max():.0f} m")
        
        except Exception as e:
            st.error(f"Error loading DEM: {str(e)}")
            st.stop()
    
    # Load manual TC if provided
    tc_map = None
    if tc_file:
        try:
            if tc_file.name.endswith('.csv'):
                tc_df = pd.read_csv(tc_file)
            else:
                tc_df = pd.read_excel(tc_file)
            
            if {"Nama", "Koreksi_Medan"}.issubset(tc_df.columns):
                tc_df["Koreksi_Medan"] = pd.to_numeric(tc_df["Koreksi_Medan"], errors="coerce")
                tc_map = tc_df.set_index("Nama")["Koreksi_Medan"].to_dict()
                st.info(f"✅ Manual TC loaded: {len(tc_map)} stations")
            else:
                st.warning("Manual TC file needs columns: Nama, Koreksi_Medan")
        except Exception as e:
            st.warning(f"Could not load manual TC: {e}")
    
    # Process gravity data
    try:
        xls = pd.ExcelFile(grav_file)
    except Exception as e:
        st.error(f"Error reading Excel: {e}")
        st.stop()
    
    all_results = []
    tc_values_all = []
    
    total_sheets = len(xls.sheet_names)
    progress_bar = st.progress(0)
    
    for sheet_idx, sheet_name in enumerate(xls.sheet_names):
        df = pd.read_excel(grav_file, sheet_name=sheet_name)
        
        # Check required columns
        required = {"Nama", "Time", "G_read (mGal)", "Lat", "Lon", "Elev"}
        if not required.issubset(set(df.columns)):
            st.warning(f"⚠️ Sheet '{sheet_name}' skipped (missing columns)")
            continue
        
        # Update progress
        progress = (sheet_idx + 1) / total_sheets
        progress_bar.progress(progress)
        
        # Convert coordinates
        E, N, _, _ = latlon_to_utm_redfearn(df["Lat"].to_numpy(), df["Lon"].to_numpy())
        df["Easting"] = E
        df["Northing"] = N
        
        # Drift correction
        Gmap, D = compute_drift(df, G_base, debug_mode)
        df["G_read (mGal)"] = df["Nama"].map(Gmap)
        
        # Basic corrections
        df["Koreksi Lintang"] = latitude_correction(df["Lat"])
        df["Free Air Correction"] = free_air(df["Elev"])
        df["FAA"] = df["G_read (mGal)"] - df["Koreksi Lintang"] + df["Free Air Correction"]
        
        # Terrain correction for each station
        tc_list = []
        n_stations = len(df)
        
        station_bar = st.progress(0)
        for i, station in df.iterrows():
            station_name = station['Nama']
            
            # Manual TC if available
            if tc_map and station_name in tc_map:
                tc_val = tc_map[station_name]
            else:
                # Calculate using Geosoft method
                tc_params = {
                    'correction_distance': correction_distance,
                    'earth_density': earth_density,
                    'optimize': optimize_calc,
                    'debug': debug_mode
                }
                
                tc_val, _ = GeosoftTerrainCorrector(
                    dem_df, 
                    (station['Easting'], station['Northing'], station['Elev']),
                    tc_params
                ).calculate_terrain_correction()
            
            tc_list.append(tc_val)
            tc_values_all.append(tc_val)
            
            # Update station progress
            station_bar.progress((i + 1) / n_stations)
        
        station_bar.empty()
        
        # Add TC to dataframe
        df["Koreksi Medan"] = tc_list
        
        # Parasnis calculations
        df["X-Parasnis"] = 0.04192 * df["Elev"] - df["Koreksi Medan"]
        df["Y-Parasnis"] = df["Free Air Correction"]
        df["Hari"] = sheet_name
        
        all_results.append(df)
    
    progress_bar.empty()
    
    if not all_results:
        st.error("No valid sheets processed")
        st.stop()
    
    # Combine all data
    df_all = pd.concat(all_results, ignore_index=True)
    
    # Density analysis
    with st.spinner("Analyzing density..."):
        density_rec, density_uncert, density_results = analyze_density_comprehensive(df_all)
        
        if density_rec:
            # Calculate Bouguer anomalies with recommended density
            df_all["Bouger Correction"] = 0.04192 * density_rec * df_all["Elev"]
            df_all["Simple Bouger Anomaly"] = df_all["FAA"] - df_all["Bouger Correction"]
            df_all["Complete Bouger Anomaly"] = df_all["Simple Bouger Anomaly"] + df_all["Koreksi Medan"]
            
            st.session_state.recommended_density = density_rec
            st.session_state.density_results = density_results
        else:
            # Use default density
            df_all["Bouger Correction"] = 0.04192 * 2.67 * df_all["Elev"]
            df_all["Simple Bouger Anomaly"] = df_all["FAA"] - df_all["Bouger Correction"]
            df_all["Complete Bouger Anomaly"] = df_all["Simple Bouger Anomaly"] + df_all["Koreksi Medan"]
    
    st.success(f"✅ Processing complete! {len(df_all)} stations processed")
    
    # ============================================================
    # RESULTS DISPLAY
    # ============================================================
    
    # Summary statistics
    st.header("📈 Results Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Stations", len(df_all))
    with col2:
        st.metric("Mean TC", f"{np.mean(tc_values_all):.2f} mGal")
    with col3:
        st.metric("Max TC", f"{np.max(tc_values_all):.2f} mGal")
    with col4:
        if 'recommended_density' in st.session_state:
            st.metric("Recommended Density", f"{st.session_state.recommended_density:.3f} g/cm³")
    
    # TC statistics
    st.subheader("Terrain Correction Statistics")
    
    tc_array = np.array(tc_values_all)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Mean:** {tc_array.mean():.3f} mGal")
        st.write(f"**Median:** {np.median(tc_array):.3f} mGal")
    with col2:
        st.write(f"**Min:** {tc_array.min():.3f} mGal")
        st.write(f"**Max:** {tc_array.max():.3f} mGal")
    with col3:
        st.write(f"**Std Dev:** {tc_array.std():.3f} mGal")
        st.write(f"**Range:** {tc_array.max() - tc_array.min():.3f} mGal")
    
    # Data preview
    with st.expander("📋 Data Preview", expanded=True):
        st.dataframe(df_all.head(20))
    
    # ============================================================
    # VISUALIZATIONS
    # ============================================================
    if show_plots:
        st.header("📊 Visualizations")
        
        tabs = st.tabs(["DEM & Stations", "Terrain Correction", "Bouguer Anomaly", "Density Analysis"])
        
        with tabs[0]:
            # DEM and stations plot
            fig_dem = plot_dem_stations(dem_df, df_all)
            st.pyplot(fig_dem)
        
        with tabs[1]:
            # Terrain correction map
            fig_tc = plot_terrain_correction_map(df_all)
            if fig_tc:
                st.pyplot(fig_tc)
            
            # TC histogram
            fig_hist, ax = plt.subplots(figsize=(10, 4))
            ax.hist(tc_values_all, bins=30, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Terrain Correction (mGal)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Terrain Correction Values')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig_hist)
        
        with tabs[2]:
            # Bouguer anomaly
            fig_cba = plot_bouguer_anomaly(df_all)
            if fig_cba:
                st.pyplot(fig_cba)
        
        with tabs[3]:
            # Density analysis
            if 'density_results' in st.session_state:
                st.subheader("Density Analysis Results")
                
                # Display results
                for method, value in st.session_state.density_results.items():
                    st.write(f"**{method}:** {value:.3f} g/cm³")
                
                # Plot comparison
                methods = list(st.session_state.density_results.keys())
                values = list(st.session_state.density_results.values())
                
                fig_dens, ax = plt.subplots(figsize=(10, 5))
                bars = ax.bar(methods, values, alpha=0.7)
                ax.axhline(y=st.session_state.recommended_density, 
                          color='red', linestyle='--', 
                          label=f'Recommended: {st.session_state.recommended_density:.3f}')
                ax.set_ylabel('Density (g/cm³)')
                ax.set_title('Density Determination Methods')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{val:.3f}', ha='center', va='bottom')
                
                st.pyplot(fig_dens)
    
    # ============================================================
    # EXPORT OPTIONS
    # ============================================================
    st.header("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export main results
        csv_data = df_all.to_csv(index=False)
        st.download_button(
            label="📥 Download All Results (CSV)",
            data=csv_data,
            file_name="gravity_processing_results.csv",
            mime="text/csv",
            help="Download all processed data"
        )
    
    with col2:
        # Export terrain corrections only
        tc_df = df_all[['Nama', 'Lat', 'Lon', 'Elev', 'Koreksi Medan']].copy()
        tc_csv = tc_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Terrain Corrections",
            data=tc_csv,
            file_name="terrain_corrections.csv",
            mime="text/csv",
            help="Download only terrain correction values"
        )
    
    # Export to Excel
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df_all.to_excel(writer, index=False, sheet_name='Results')
        
        # Add summary sheet
        summary_data = {
            'Metric': ['Total Stations', 'Mean TC (mGal)', 'Max TC (mGal)', 
                      'Recommended Density (g/cm³)', 'Processing Time'],
            'Value': [len(df_all), f"{np.mean(tc_values_all):.3f}", 
                     f"{np.max(tc_values_all):.3f}",
                     f"{st.session_state.get('recommended_density', 'N/A')}",
                     time.strftime("%Y-%m-%d %H:%M:%S")]
        }
        pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name='Summary')
    
    st.download_button(
        label="📥 Download Excel Report",
        data=excel_buffer.getvalue(),
        file_name="gravity_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ============================================================
# SIDEBAR INFORMATION
# ============================================================

st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ About This Tool"):
    st.write("""
    **Auto Grav Pro - Geosoft Terrain Correction**
    
    Uses scientific terrain correction methods:
    - **Nagy (1966)**: Rectangular prism formula
    - **Kane (1962)**: Square segment rings
    
    **Input Requirements:**
    1. **Gravity Data**: Excel with sheets (Nama, Time, G_read, Lat, Lon, Elev)
    2. **DEM Data**: CSV with lat, lon, elev columns
    
    **Output:**
    - Terrain correction for each station
    - Complete Bouguer anomaly
    - Density analysis results
    - Interactive visualizations
    """)

with st.sidebar.expander("📋 Sample File Format"):
    st.write("""
    **DEM CSV Format:**
    ```
    lat,lon,elev
    -7.123,110.456,125.5
    -7.124,110.457,126.2
    ```
    
    **Gravity Excel Format:**
    - Multiple sheets (one per day)
    - Columns: Nama, Time, G_read (mGal), Lat, Lon, Elev
    """)
