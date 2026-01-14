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

# ============================================================
# KOREKSI UTAMA YANG DIPERBAIKI
# ============================================================

# -----------------------
# KOREKSI LATITUDE CORRECTION YANG BENAR
# -----------------------
def latitude_correction(lat):
    """
    KOREKSI YANG BENAR: Menghitung KOREKSI LINTANG (bukan gravitasi normal absolut)
    Rumus koreksi lintang internasional (dalam mGal)
    Koreksi = perbedaan gravitasi antara stasiun dan ekuator
    """
    phi = np.radians(lat)
    sin_2phi = np.sin(2*phi)
    sin_4phi = np.sin(4*phi)
    
    # Koefisien untuk ellipsoid GRS67 (dalam mGal)
    k1 = 0.812  # mGal untuk sin(2φ)
    k2 = 0.542  # mGal untuk sin(4φ)
    
    # Koreksi lintang relatif terhadap ekuator
    correction = k1 * sin_2phi - k2 * sin_4phi
    
    # Konversi ke mGal (koefisien sudah dalam mGal)
    return correction * 1000.0  # KOREKSI: dikali 1000

def free_air(elev):
    """Koreksi udara bebas: 0.3086 mGal/meter"""
    return 0.3086 * elev

def bouguer_correction(elev, density=2670.0):
    """Koreksi Bouguer slab: 0.04192 × density/2670 × elev (mGal)"""
    return 0.04192 * (density/2670.0) * elev

# -----------------------
# KOREKSI OSS DENGAN DENSITY PARAMETER
# -----------------------
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
    delta_g_si = 6.67430e-11 * density * Delta_theta * term
    
    # Convert to mGal (1 mGal = 10^-5 m/s²)
    return delta_g_si * 1e5

# Update class OSSTerrainCorrector untuk menerima density
class OSSTerrainCorrector:
    def __init__(self, dem_df, station_coords, params=None, density=2670.0):
        self.dem_df = dem_df.copy()
        self.e0, self.n0, self.z0 = station_coords
        
        # Default parameters dari paper
        self.params = {
            'max_radius': 4500.0,
            'tolerance_nGal': 1.0,
            'threshold_mGal': 1.0,
            'theta_step': 1.0,
            'r_step_near': 10.0,
            'r_step_far': 50.0,
            'min_points_per_sector': 10,
            'use_optimized_elevation': True,
            'debug': False,
            'density': density  # SIMPAN DENSITY
        }
        
        if params:
            self.params.update(params)
        
        # Convert DEM ke koordinat polar
        self._prepare_polar_coords()
    
    def _sector_effect(self, R1, R2, theta1, theta2, z_avg):
        """Wrapper untuk terrain_effect_cylindrical_sector dengan density"""
        return terrain_effect_cylindrical_sector(
            R1, R2, theta1, theta2, z_avg, self.params['density']
        )
    
    # ... [method lainnya tetap sama, tapi ganti semua pemanggilan 
    #     terrain_effect_cylindrical_sector dengan self._sector_effect]
    
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

# Update fungsi wrapper
def calculate_oss_correction(dem_df, station_row, density=2670.0, params=None):
    """Wrapper function dengan density parameter"""
    station_coords = (
        float(station_row['Easting']),
        float(station_row['Northing']),
        float(station_row['Elev'])
    )
    
    corrector = OSSTerrainCorrector(dem_df, station_coords, params, density)
    tc_value, _ = corrector.calculate_terrain_correction()
    
    return tc_value

# ============================================================
# KOREKSI BAGIAN UTAMA PEMROSESAN
# ============================================================

# ... [bagian awal kode (hash, login, UTM, load_dem, compute_drift) tetap sama] ...

# Di bagian pemrosesan utama, GANTI:

if run:
    # ... [validasi awal tetap sama] ...
    
    # BACA DATA EXCEL
    try:
        xls = pd.ExcelFile(grav)
    except Exception as e:
        st.error(f"Gagal baca Excel gravitasi: {e}")
        st.stop()
    
    all_dfs = []
    t0 = time.time()
    
    for sh in xls.sheet_names:
        df = pd.read_excel(grav, sheet_name=sh)
        required = {"Nama","Time","G_read (mGal)","Lat","Lon","Elev"}
        if not required.issubset(set(df.columns)):
            st.warning(f"Sheet {sh} dilewati (kolom tidak lengkap).")
            continue
        
        # UTM conversion
        E,N,_,_ = latlon_to_utm_redfearn(df["Lat"].to_numpy(), df["Lon"].to_numpy())
        df["Easting"] = E
        df["Northing"] = N
        
        # Drift correction
        Gmap, D = compute_drift(df, G_base)
        df["G_read (mGal)"] = df["Nama"].map(Gmap)
        
        # ============================================================
        # KOREKSI KRITIS: PERHITUNGAN YANG BENAR
        # ============================================================
        
        # 1. Koreksi dasar (DIPERBAIKI)
        df["Latitude_Correction"] = latitude_correction(df["Lat"])  # dalam mGal
        df["Free_Air_Correction"] = free_air(df["Elev"])           # dalam mGal
        
        # 2. FAA yang BENAR: TIDAK kurangi dengan gravitasi normal!
        # FAA = G_observed + Free_Air_Correction
        df["FAA"] = df["G_read (mGal)"] + df["Free_Air_Correction"]
        
        # 3. Hitung terrain correction
        tc_list = []
        nstations = len(df)
        
        # Parameter OSS
        oss_params = {
            'max_radius': max_radius,
            'tolerance_nGal': 1.0,
            'threshold_mGal': 1.0,
            'theta_step': 1.0,
            'r_step_near': 10.0,
            'r_step_far': 50.0,
            'use_optimized_elevation': True,
            'debug': False
        }
        
        for i in range(nstations):
            station_data = df.iloc[i]
            
            if method == "OSS (Algorithm from Paper)":
                if dem is not None:
                    # PASS density ke OSS
                    tc_val = calculate_oss_correction(dem, station_data, density, oss_params)
                else:
                    tc_val = 0.0
            elif method == "HAMMER (Legacy)":
                e0 = float(station_data["Easting"])
                n0 = float(station_data["Northing"])
                z0 = float(station_data["Elev"])
                tc_val = hammer_tc(e0, n0, z0, dem) if dem is not None else 0.0
            elif method == "NAGY Prism (Reference)":
                e0 = float(station_data["Easting"])
                n0 = float(station_data["Northing"])
                z0 = float(station_data["Elev"])
                if dem is not None:
                    tc_val = simple_nagy_prism(e0, n0, z0, dem, density, max_radius)
                else:
                    tc_val = 0.0
            else:
                tc_val = 0.0
            
            tc_list.append(tc_val)
        
        df["Koreksi_Medan"] = tc_list
        
        # 4. Hitung X-Parasnis dan Y-Parasnis
        df["X-Parasnis"] = 0.04192 * df["Elev"] - df["Koreksi_Medan"]
        df["Y-Parasnis"] = df["Free_Air_Correction"]  # Y = FAC
        
        df["Hari"] = sh
        all_dfs.append(df)
    
    # ============================================================
    # ANALISIS PARASNIS YANG BENAR
    # ============================================================
    if len(all_dfs) == 0:
        st.error("No valid sheets processed.")
        st.stop()
    
    df_all = pd.concat(all_dfs, ignore_index=True)
    elapsed = time.time() - t0
    
    st.write(f"Processed {len(df_all)} rows in {elapsed:.1f} s")
    
    # Regresi Parasnis
    mask = df_all[["X-Parasnis","Y-Parasnis"]].notnull().all(axis=1)
    if mask.sum() >= 2:
        X = df_all.loc[mask, "X-Parasnis"]
        Y = df_all.loc[mask, "Y-Parasnis"]
        slope, intercept = np.polyfit(X, Y, 1)
        
        # Hitung R-squared
        y_pred = slope * X + intercept
        y_actual = Y
        r_squared = 1 - np.sum((y_actual - y_pred)**2) / np.sum((y_actual - np.mean(y_actual))**2)
        
        # Hitung density dari slope Parasnis
        # slope = 0.04192 × (ρ_bouguer / 2670)
        density_bouguer = slope * 2670.0 / 0.04192
        
        st.success(f"**Hasil Analisis Parasnis:**")
        st.write(f"• Slope (K) = {slope:.6f}")
        st.write(f"• Densitas Bouguer = {density_bouguer:.1f} kg/m³")
        st.write(f"• R² = {r_squared:.3f}")
        
        # Validasi density
        if density_bouguer < 1500 or density_bouguer > 3500:
            st.warning(f"Density Bouguer tidak realistis ({density_bouguer:.1f} kg/m³)")
            st.write("Menggunakan density dari input sidebar")
            density_bouguer = density
        
        # Hitung Bouguer Correction dengan density dari Parasnis
        density_factor = density_bouguer / 2670.0
        
    else:
        slope = np.nan
        density_bouguer = density
        density_factor = density_bouguer / 2670.0
        st.warning("Tidak cukup data untuk analisis Parasnis")
    
    # ============================================================
    # PERHITUNGAN BOUGUER YANG BENAR
    # ============================================================
    
    # Bouguer Correction dengan density yang sesuai
    df_all["Bouguer_Correction"] = 0.04192 * df_all["Elev"] * density_factor
    
    # Simple Bouguer Anomaly
    df_all["Simple_Bouguer_Anomaly"] = df_all["FAA"] - df_all["Bouguer_Correction"]
    
    # Complete Bouguer Anomaly
    df_all["Complete_Bouguer_Anomaly"] = df_all["Simple_Bouguer_Anomaly"] + df_all["Koreksi_Medan"]
    
    # ============================================================
    # VALIDASI HASIL
    # ============================================================
    st.subheader("🩺 Validasi Hasil")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean G_read", f"{df_all['G_read (mGal)'].mean():.2f} mGal")
    with col2:
        st.metric("Mean FAA", f"{df_all['FAA'].mean():.2f} mGal")
    with col3:
        st.metric("Mean SBA", f"{df_all['Simple_Bouguer_Anomaly'].mean():.2f} mGal")
    with col4:
        st.metric("Mean CBA", f"{df_all['Complete_Bouguer_Anomaly'].mean():.2f} mGal")
    
    # Validasi range yang realistic
    st.write("**Range yang harusnya realistis:**")
    st.write("- G_read: ±200 mGal (relatif ke base)")
    st.write("- FAA: ±300 mGal")
    st.write("- SBA/CBA: ±200 mGal")
    st.write("- TC: 0-50 mGal (topografi kasar)")
    
    if df_all["FAA"].abs().max() > 1000:
        st.error("❌ FAA terlalu besar! Pastikan:")
        st.write("1. G_read dalam mGal (bukan Gal)")
        st.write("2. TIDAK kurangi dengan gravitasi normal absolut")
    
    # ============================================================
    # TAMPILKAN HASIL
    # ============================================================
    st.success("✅ Processing completed!")
    st.dataframe(df_all.head(10))
    
    # Plot Parasnis
    st.subheader("Plot Parasnis X–Y")
    if mask.sum() >= 2:
        X = df_all.loc[mask, "X-Parasnis"].values
        Y = df_all.loc[mask, "Y-Parasnis"].values
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X, Y, s=25, color="blue", label="Data Parasnis", alpha=0.7)
        
        X_line = np.linspace(min(X), max(X), 100)
        Y_line = slope * X_line + intercept
        ax.plot(X_line, Y_line, color="red", linewidth=2,
                label=f"Regresi: Y = {slope:.5f}X + {intercept:.2f}")
        
        ax.set_xlabel("X-Parasnis (mGal)")
        ax.set_ylabel("Y-Parasnis (mGal)")
        ax.set_title(f"Diagram Parasnis\nρ = {density_bouguer:.1f} kg/m³")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        st.pyplot(fig)
    
    # Download
    csv = df_all.to_csv(index=False)
    st.download_button(
        "Download CSV", 
        csv.encode("utf-8"),
        "Hasil_Perhitungan_Benar.csv"
    )
