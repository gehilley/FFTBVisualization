import numpy as np
import matplotlib.pyplot as plt
from obspy.imaging.beachball import beach
def strike_dip_rake_to_mt(strike, dip, rake, M0=1.0):
    """
    Convert strike, dip, rake to moment tensor components for ObsPy beach function.
    Parameters
    ----------
    strike : float
        Strike angle in degrees (0-360), measured clockwise from north
    dip : float
        Dip angle in degrees (0-90), measured down from horizontal
    rake : float
        Rake angle in degrees (-180 to 180)
    M0 : float
        Scalar moment (default=1.0)
    Returns
    -------
    mt : list
        Moment tensor components [M11, M22, M33, M12, M13, M23] in ObsPy convention
    """
    # Convert angles to radians
    phi = np.deg2rad(strike)
    delta = np.deg2rad(dip)
    lambd = np.deg2rad(rake)
    # Calculate moment tensor components in Aki & Richards (NED) convention
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_2phi = np.sin(2 * phi)
    cos_2phi = np.cos(2 * phi)
    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)
    sin_2delta = np.sin(2 * delta)
    cos_2delta = np.cos(2 * delta)
    sin_lambd = np.sin(lambd)
    cos_lambd = np.cos(lambd)
    # Aki & Richards (North-East-Down) convention
    Mxx = -M0 * (sin_delta * cos_lambd * sin_2phi + sin_2delta * sin_lambd * sin_phi**2)
    Myy = M0 * (sin_delta * cos_lambd * sin_2phi - sin_2delta * sin_lambd * cos_phi**2)
    Mzz = M0 * sin_2delta * sin_lambd
    Mxy = M0 * (sin_delta * cos_lambd * cos_2phi + 0.5 * sin_2delta * sin_lambd * sin_2phi)
    Mxz = -M0 * (cos_delta * cos_lambd * cos_phi + cos_2delta * sin_lambd * sin_phi)
    Myz = -M0 * (cos_delta * cos_lambd * sin_phi - cos_2delta * sin_lambd * cos_phi)
    # Convert to ObsPy (Up-South-East) convention
    M11 = Mzz  # Mrr
    M22 = Mxx  # Mtt
    M33 = Myy  # Mpp
    M12 = Mxz  # Mrt
    M13 = -Myz # Mrp
    M23 = -Mxy # Mtp
    return [M11, M22, M33, M12, M13, M23]
def rotate_moment_tensor(strike, dip, rake, v, theta, use_upper_hemisphere=False):
    """
    Rotate a moment tensor about an arbitrary axis.
    Parameters
    ----------
    strike : float
        Strike angle in degrees
    dip : float
        Dip angle in degrees
    rake : float
        Rake angle in degrees
    v : array-like
        Unit vector defining the rotation axis [vx, vy, vz]
        In geographic coordinates: x=North, y=East, z=Down
    theta : float
        Rotation angle in degrees (positive = counterclockwise when looking along v)
    use_upper_hemisphere : bool
        If True, convert to upper hemisphere before rotation (for map view to cross-section)
    Returns
    -------
    mt_rotated : list
        Rotated moment tensor components [M11, M22, M33, M12, M13, M23] for ObsPy
    Notes
    -----
    Uses Rodrigues' rotation formula to rotate the moment tensor.
    The rotation is performed in the North-East-Down coordinate system,
    then converted to ObsPy's Up-South-East convention.
    """
    # First, get the moment tensor from strike/dip/rake
    mt_obspy = strike_dip_rake_to_mt(strike, dip, rake)
    # If requested, convert to upper hemisphere before rotation
    if use_upper_hemisphere:
        mt_obspy = convert_to_upper_hemisphere(mt_obspy)
    # Convert from ObsPy (USE) to NED convention for rotation
    # ObsPy: 1=Up, 2=South, 3=East
    # NED: x=North, y=East, z=Down
    # Mrr=Mzz, Mtt=Mxx, Mpp=Myy, Mrt=Mxz, Mrp=-Myz, Mtp=-Mxy
    Mzz = mt_obspy[0]  # M11 = Mrr
    Mxx = mt_obspy[1]  # M22 = Mtt
    Myy = mt_obspy[2]  # M33 = Mpp
    Mxz = mt_obspy[3]  # M12 = Mrt
    Myz = -mt_obspy[4] # M13 = Mrp, so Myz = -Mrp
    Mxy = -mt_obspy[5] # M23 = Mtp, so Mxy = -Mtp
    # Build the moment tensor matrix in NED coordinates
    M_ned = np.array([[Mxx, Mxy, Mxz],
                      [Mxy, Myy, Myz],
                      [Mxz, Myz, Mzz]])
    # Normalize the rotation axis
    v = np.array(v, dtype=float)
    v = v / np.linalg.norm(v)
    # Special handling for cross-section view
    if use_upper_hemisphere and abs(v[2]) < 0.1:  # Horizontal rotation axis
        # Step 1: Calculate azimuth of the horizontal axis
        axis_azimuth = np.degrees(np.arctan2(v[1], v[0]))  # Azimuth from North
        # Step 2: Rotate about vertical (z-axis) by -axis_azimuth
        # This aligns the rotation axis with the E-W direction
        theta_vertical = np.radians(axis_azimuth)
        R_vertical = np.array([[np.cos(theta_vertical), -np.sin(theta_vertical), 0],
                               [np.sin(theta_vertical), np.cos(theta_vertical), 0],
                               [0, 0, 1]])
        # Apply vertical rotation
        M_ned = np.dot(np.dot(R_vertical, M_ned), R_vertical.T)
        # Step 3: Now rotate about E-W axis (y-axis in NED) by theta
        # Using right-hand rule: positive rotation tips northward side down
        theta_rad = np.deg2rad(theta)
        R_ew = np.array([[np.cos(theta_rad), 0, np.sin(theta_rad)],
                         [0, 1, 0],
                         [-np.sin(theta_rad), 0, np.cos(theta_rad)]])
        # Apply E-W rotation
        M_rotated_ned = np.dot(np.dot(R_ew, M_ned), R_ew.T)
    else:
        # Standard rotation using Rodrigues' formula for non-cross-section cases
        # Convert rotation angle to radians
        theta_rad = np.deg2rad(theta)
        # Build rotation matrix using Rodrigues' formula
        # R = I + sin(θ)K + (1-cos(θ))K²
        # where K is the skew-symmetric matrix of v
        I = np.eye(3)
        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        R = I + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
        # Rotate the moment tensor: M' = R * M * R^T
        M_rotated_ned = np.dot(np.dot(R, M_ned), R.T)
    # Extract components
    Mxx_rot = M_rotated_ned[0, 0]
    Myy_rot = M_rotated_ned[1, 1]
    Mzz_rot = M_rotated_ned[2, 2]
    Mxy_rot = M_rotated_ned[0, 1]
    Mxz_rot = M_rotated_ned[0, 2]
    Myz_rot = M_rotated_ned[1, 2]
    # Convert back to ObsPy (USE) convention
    M11_rot = Mzz_rot  # Mrr
    M22_rot = Mxx_rot  # Mtt
    M33_rot = Myy_rot  # Mpp
    M12_rot = Mxz_rot  # Mrt
    M13_rot = -Myz_rot # Mrp
    M23_rot = -Mxy_rot # Mtp
    return [M11_rot, M22_rot, M33_rot, M12_rot, M13_rot, M23_rot]
def convert_to_upper_hemisphere(mt_obspy):
    """
    Convert moment tensor to upper hemisphere projection.
    This flips the sign of components that involve the vertical (Up) direction
    to simulate viewing the beach ball from above instead of below.
    Parameters
    ----------
    mt_obspy : list
        Moment tensor components [M11, M22, M33, M12, M13, M23] in ObsPy convention
        where 1=Up, 2=South, 3=East
    Returns
    -------
    mt_upper : list
        Moment tensor components for upper hemisphere projection
    """
    # In ObsPy convention: 1=Up, 2=South, 3=East
    # For upper hemisphere, we flip components involving the Up (1) direction
    M11 = mt_obspy[0]  # Mrr (Up-Up) - no change
    M22 = mt_obspy[1]  # Mtt (South-South) - no change
    M33 = mt_obspy[2]  # Mpp (East-East) - no change
    M12 = -mt_obspy[3] # Mrt (Up-South) - flip sign
    M13 = -mt_obspy[4] # Mrp (Up-East) - flip sign
    M23 = mt_obspy[5]  # Mtp (South-East) - no change
    return [M11, M22, M33, M12, M13, M23]
def vector_from_trend_plunge(trend, plunge=0):
    """
    Convert trend and plunge to a unit vector in NED coordinates.
    Parameters
    ----------
    trend : float
        Azimuth in degrees, measured clockwise from North (0-360)
    plunge : float
        Plunge in degrees, positive downward (0-90)
    Returns
    -------
    v : array
        Unit vector [North, East, Down]
    """
    trend_rad = np.deg2rad(trend)
    plunge_rad = np.deg2rad(plunge)
    # In NED coordinates
    v_north = np.cos(trend_rad) * np.cos(plunge_rad)
    v_east = np.sin(trend_rad) * np.cos(plunge_rad)
    v_down = np.sin(plunge_rad)
    return np.array([v_north, v_east, v_down])
