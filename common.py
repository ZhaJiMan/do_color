import numpy as np
import pandas as pd

def XYZ_to_xyz(XYZ):
    '''将XYZ转换为xyz. X+Y+Z=0时xyz为NaN.'''
    XYZ = np.asarray(XYZ)
    S = XYZ.sum(axis=-1, keepdims=True)
    S = np.where(np.isclose(S, 0), np.nan, S)
    xyz = XYZ / S

    return xyz

def XYZ_to_xyY(XYZ):
    '''将XYZ转换为xyY. X+Y+Z=0时xy为NaN.'''
    XYZ = np.asarray(XYZ)
    xyY = XYZ_to_xyz(XYZ)
    xyY[..., 2] = XYZ[..., 1]

    return xyY

def xyY_to_XYZ(xyY):
    '''将xyY转换为XYZ.'''
    xyY = np.asarray(xyY)
    x, y, Y = (xyY[..., i] for i in range(3))
    Y_y = Y / y
    X = x * Y_y
    Z = (1 - x - y) * Y_y
    XYZ = np.stack((X, Y, Z), axis=-1)

    return XYZ

def gamma_encoding(RGB):
    '''对线性sRGB做编码得到sRGB.'''
    RGB = np.asarray(RGB).astype(float)
    mask = RGB > 0.0031308
    RGB[~mask] *= 12.92
    RGB[mask] = 1.055 * RGB[mask]**(1 / 2.4) - 0.055

    return RGB

def gamma_decoding(RGB):
    '''对sRGB做解码得到线性sRGB.'''
    RGB = np.array(RGB, float)
    mask = RGB > 0.04045
    RGB[~mask] /= 12.92
    RGB[mask] = ((RGB[mask] + 0.055) / 1.055)**2.4

    return RGB

def normalize_by_maximum(RGB):
    '''分别用RGB里三个分量的最大值做归一化.'''
    RGB = np.array(RGB, float)
    RGB /= RGB.max(axis=-1, keepdims=True)

    return RGB

def XYZ_to_sRGB(XYZ):
    '''将XYZ转换为线性sRGB.'''
    M = np.array([
        [+3.2406, -1.5372, -0.4986],
        [-0.9689, +1.8758, +0.0415],
        [+0.0557, -0.2040, +1.0570]
    ])
    RGB = np.tensordot(XYZ, M, (-1, 1))

    return RGB

def sRGB_to_XYZ(RGB):
    '''将线性sRGB转换为XYZ.'''
    M = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]
    ])
    XYZ = np.tensordot(RGB, M, (-1, 1))

    return XYZ

def XYZ_to_RGB(XYZ):
    '''将XYZ转换为RGB(CIE 1931).'''
    M = np.array([
        [+0.41846, -0.15866, -0.08283],
        [-0.09117, +0.25243, +0.01571],
        [+0.00092, -0.00255, +0.17860]
    ])
    RGB = np.tensordot(XYZ, M, (-1, 1))

    return RGB

def RGB_to_XYZ(RGB):
    '''将RGB(CIE 1931)转换为XYZ.'''
    M = np.array([
        [2.76888, 1.75175, 1.13016],
        [1.00000, 4.59070, 0.06010],
        [0.00000, 0.05651, 5.59427]
    ])
    XYZ = np.tensordot(RGB, M, (-1, 1))

    return XYZ

def RGB_to_rgb(RGB):
    '''将RGB转换为rgb.'''
    return XYZ_to_xyz(RGB)

def load_xyz_cmf():
    '''读取1931 XYZ CMF.'''
    return pd.read_csv('./data/cie_1931_2deg_xyz_cmf.csv', index_col=0)

def load_xyz_cc():
    '''读取1931 XYZ CC.'''
    return pd.read_csv('./data/cie_1931_2deg_xyz_cc.csv', index_col=0)

def load_rgb_cmf(from_xyz=True):
    '''读取Wright-Guild RGB CMF.'''
    if from_xyz:
        xyz_cmf = load_xyz_cmf()
        rgb_cmf = pd.DataFrame(
            XYZ_to_RGB(xyz_cmf),
            index=xyz_cmf.index,
            columns=['r', 'g', 'b']
        )
    else:
        rgb_cmf = pd.read_csv(
            './data/wright_guild_1931_2deg_rgb_cmf.csv',
            index_col=0
        )

    return rgb_cmf

def load_rgb_cc(from_xyz=True):
    '''读取Wright-Guild RGB CC.'''
    if from_xyz:
        rgb_cmf = load_rgb_cmf(from_xyz=True)
        rgb_cc = pd.DataFrame(
            RGB_to_rgb(rgb_cmf),
            index=rgb_cmf.index,
            columns=rgb_cmf.columns
        )
    else:
        rgb_cc = pd.read_csv(
            './data/wright_guild_1931_2deg_rgb_cc.csv',
            index_col=0
        )

    return rgb_cc

def load_lef():
    '''读取1924 LEF.'''
    return pd.read_csv(
        './data/cie_1924_photopic_lef.csv',
        index_col=0
    )['lef']

def load_lms():
    '''读取Stiles & Burch LMS.'''
    return pd.read_csv(
        './data/stiles_burch_2deg_lms.csv',
        index_col=0
    ).fillna(0)