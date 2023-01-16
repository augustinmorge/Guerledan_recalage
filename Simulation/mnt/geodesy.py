#!/usr/bin/python3
import numpy as np
from scipy.integrate import quad
import pyproj as pj

## Global variables : parameters for ellipsoid GRS80
mas_to_rad = np.pi / (180 * 3600 * 1000)

a = 6378137
e2 = 0.006694380022
omega_eura = np.array([-0.085, -0.531, 0.770]) * mas_to_rad
t0 = 2000.0
ti = 2019.0

## Variable for reference evolutions and velocities
Tx0, Ty0, Tz0 = 53.7e-3, 51.2e-3, -55.1e-3
Tx0_dot, Ty0_dot, Tz0_dot = 0.1e-3, 0.1e-3, -1.9e-3
d0 = 1.02e-9
d0_dot = 0.11e-9
eps_x0, eps_y0, eps_z0 = 0.891 * mas_to_rad, 5.390 * mas_to_rad, -8.712 * mas_to_rad
eps_x0_dot, eps_y0_dot, eps_z0_dot = 0.081 * mas_to_rad, 0.490 * mas_to_rad, -0.792 * mas_to_rad

# UTM parameters
k0 = 0.9996

## Conversion functions

def geod2ecef(lon, lat, hgt):
    v = np.sqrt(1 - e2 * np.sin(lat) ** 2)
    N = a / v
    x = (N + hgt) * np.cos(lon) * np.cos(lat)
    y = (N + hgt) * np.sin(lon) * np.cos(lat)
    z = (N * (1 - e2) + hgt) * np.sin(lat)
    return x, y, z

def ecef2geod(x, y, z):
    b = np.sqrt(z ** 2 / (1 - x ** 2 / a ** 2))
    f = (a - b) / a
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    mu = np.arctan((z * (1 - f + a * e2 / r)) / np.sqrt(x ** 2 + y ** 2))
    lon = np.arctan(y / x)
    lat = np.arctan((z * (1 - f) + e2 * a * np.sin(mu) ** 3) / ((1 - f) * np.sqrt(x ** 2 + y ** 2) - e2 * a * np.cos(mu) ** 3))
    hgt = np.sqrt(x ** 2 + y ** 2) * np.cos(lat) + z * np.sin(lat) - a * np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return lon, lat, hgt

def enu2geod(lon0, lat0, hgt0, e, n, u):
    M_loc = np.array([[e], [n], [u]])
    x0, y0, z0 = geod2ecef(lon0, lat0, hgt0)
    M0 = np.array([[x0], [y0], [z0]])
    R = np.array([-np.sin(lon0), np.cos(lon0), 0, -np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0), np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)])
    R = np.reshape(R, (3, 3))
    M = M0 + np.linalg.inv(R) @ M_loc
    x, y, z = M[0, 0], M[1, 0], M[2, 0]
    lon, lat, hgt = ecef2geod(x, y, z)
    return lon, lat, hgt

def geod2enu(lon0, lat0, hgt0, lon, lat, hgt):
    R = np.array([-np.sin(lon0), np.cos(lon0), 0, -np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0), np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)])
    R = np.reshape(R, (3, 3))
    x, y, z = geod2ecef(lon, lat, hgt)
    x0, y0, z0 = geod2ecef(lon0, lat0, hgt0)
    delta_M = np.array([[x - x0], [y - y0], [z - z0]])
    ENU = R @ delta_M
    e, n, u = ENU[0, 0], ENU[1, 0], ENU[2, 0]
    return e, n, u

### UTM projection

def utm_mu(n, lon, lat):
    b = np.sqrt(a ** 2 * (1 - e2))
    e_second2 = (a ** 2 - b ** 2) / b ** 2
    v_second = np.sqrt(1 + e_second2 * np.cos(lat) ** 2)
    lambda0 = (6 * (n - 31) + 3) * np.pi / 180
    mu = k0 * (1 + (v_second ** 2 / 2) * (lon - lambda0) ** 2 * np.cos(lat) ** 2)
    return mu

def utm_gamma(n, lon, lat):
    lambda0 = (6 * (n - 31) + 3) * np.pi / 180
    gamma = (lon - lambda0) * np.sin(lat) * (1 + (lon - lambda0) ** 2 * np.cos(lat) ** 2 / 3)
    return gamma

def utm_geod2map(n, lon, lat):
    v = np.sqrt(1 - e2 * np.sin(lat) ** 2)
    lambda0 = (6 * (n - 31) + 3) * np.pi / 180
    b = np.sqrt(a ** 2 * (1 - e2))
    e_second2 = (a ** 2 - b ** 2) / b ** 2
    v_second = np.sqrt(1 + e_second2 * np.cos(lat) ** 2)
    rho = a * (1 - e2) / v ** 3
    n1 = np.sqrt(1 + e_second2 * np.cos(lat) ** 4)
    X0 = 500000
    Y0 = 0
    N = a / v
    if lat < 0:
        Y0 = 10000000
    X = X0 + k0 * (rho * N) ** 0.5 / 2 * np.log((n1 + v_second * np.cos(lat) * np.sin(n1 * (lon - lambda0))) / (
                n1 - v_second * np.cos(lat) * np.sin(n1 * (lon - lambda0))))
    Y = Y0 + k0 * beta(lat) + k0 * (rho * N) ** 0.5 * (
                np.arctan((np.tan(lat)) / (v_second * np.cos(n1 * (lon - lambda0)))) - np.arctan(
            np.tan(lat) / v_second))
    return X, Y

def utm_map2geod(n, X, Y):
    lon, lat = np.nan, np.nan
    # ...
    return lon, lat

def beta(lat):
    def toIntegrate(theta):
        result = (1 - e2 * np.sin(theta) ** 2) ** (-1.5)
        return result

    integral, err = quad(toIntegrate, 0, lat)
    beta = a * (1 - e2) * integral
    return beta

def geod2iso(lat):
    lat_iso = np.ln(np.tan(np.pi / 4 + lat / 2) * ((1 - e2 ** 0.5 * np.sin(lat)) / (1 + e2 ** 0.5 * np.sin(lat))) ** (
                e2 ** 0.5 / 2))
    return lat_iso

def iso2geod(lat_iso):
    lat = np.nan
    # ...
    return lat

def DegDecToLambert93(lat, lon):
    """Convertit des degrés décimaux en Lambert 93. @author: Léo Pham-Van"""
    lat_lamb, lon_lamb = [], []
    inProj = pj.Proj(init='epsg:4171')
    outProj = pj.Proj(init='epsg:2154')
    for i in range(len(lat)):
        lat_lamb.append(pj.transform(inProj, outProj, lon[i], lat[i])[0])
        lon_lamb.append(pj.transform(inProj, outProj, lon[i], lat[i])[1])
    return lat_lamb, lon_lamb

def Lambert93ToDegDec(E, N):
    """Convertit des Eastings/Northings RGF93 en lat/lon WGS84. @author: Léo Pham-Van"""
    lat_wgs84, lon_wgs84 = [], []
    WGS84Proj = pj.Proj(init='epsg:4326')
    RGF93Proj = pj.Proj(init='epsg:2154')
    for i in range(len(E)):
        lat_wgs84.append(pj.transform(RGF93Proj, WGS84Proj, E[i], N[i])[0])
        lon_wgs84.append(pj.transform(RGF93Proj, WGS84Proj, E[i], N[i])[1])
    return lat_wgs84, lon_wgs84

### From a reference to another

def get_vel(X):
    """Get velocity in ITRF2014 from a station located located on the Eurasian Plate."""
    velocity = np.cross(omega_eura, X)
    return velocity

def apply_vel(P0, Pdot, t0, ti):
    """Apply velocityto the parameter P0 at date t0 for date ti."""
    Pi = P0 + (ti - t0) * Pdot
    return Pi

def get_similitude(t_i):
    """Get similitude parameters at time ti knowing parameters at time t0."""
    Txi = apply_vel(Tx0, Tx0_dot, t0, t_i)
    Tyi = apply_vel(Ty0, Ty0_dot, t0, t_i)
    Tzi = apply_vel(Tz0, Tz0_dot, t0, t_i)

    di = apply_vel(d0, d0_dot, t0, t_i)

    eps_xi = apply_vel(eps_x0, eps_x0_dot, t0, t_i)
    eps_yi = apply_vel(eps_y0, eps_y0_dot, t0, t_i)
    eps_zi = apply_vel(eps_z0, eps_z0_dot, t0, t_i)

    Ti = np.array([[Txi], [Tyi], [Tzi]])
    Ri = np.zeros((3, 3))
    Ri[:, 0] = [1, eps_zi, -eps_yi]
    Ri[:, 1] = [-eps_zi, 1, eps_xi]
    Ri[:, 2] = [eps_yi, -eps_xi, 1]

    return Ti, Ri, di

def itrf2rgf(X_i, t_i):
    """Compute the coordinates Xr at date ti knowing coordinates Xi."""
    X_i = np.reshape(X_i, (3, 1))
    Ti, Ri, di = get_similitude(t_i)
    Xr = Ti + (di * np.eye(3) + Ri) @ X_i
    return Xr

            print("Xr_tj :", Xr_tj)
            print("Xr_geod_tj :", lon_rgf, lat_rgf, hgt_rgf)
