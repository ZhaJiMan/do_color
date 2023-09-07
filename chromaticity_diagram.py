import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from common import *

xyz_cc = load_xyz_cc()
rgb_cc = load_rgb_cc()
N = 256

class BaseDiagram:
    '''色度图基类.'''
    def __init__(self, ax=None):
        '''没有ax时创建fig和ax.'''
        if ax is None:
            _, self.ax = plt.subplots()
        else:
            self.ax = ax

    def save_fig(self, filepath):
        '''保存图片.'''
        self.ax.figure.savefig(filepath, dpi=300, bbox_inches='tight')

class xyDiagram(BaseDiagram):
    @staticmethod
    def generate_RGB():
        '''生成传给imshow的RGB数组.'''
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N).clip(1e-3, 1)
        x, y = np.meshgrid(x, y)
        Y = np.ones_like(x)
        xyY = np.dstack((x, y, Y))
        XYZ = xyY_to_XYZ(xyY)
        RGB = XYZ_to_sRGB(XYZ)
        RGB = normalize_by_maximum(RGB)
        RGB = gamma_encoding(RGB.clip(0, 1))

        return RGB

    def set_axes(self):
        '''设置ax的范围, 刻度和标签.'''
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-0.1, 0.9)
        self.ax.set_ylim(-0.1, 1.0)
        self.ax.xaxis.set_major_locator(MultipleLocator(0.1))
        self.ax.yaxis.set_major_locator(MultipleLocator(0.1))
        self.ax.tick_params(labelsize='small')
        self.ax.grid(ls='--', lw=0.5, c='k')
        self.ax.set_xlabel('x', fontsize='large')
        self.ax.set_ylabel('y', fontsize='large', rotation=0)

    def add_chromaticity(self):
        '''添加填色后的色度区域.'''
        patch = Polygon(
            xy=xyz_cc[['x', 'y']],
            transform=self.ax.transData,
            ec='k', fc='none', lw=1
        )
        self.ax.add_patch(patch)

        self.ax.imshow(
            self.generate_RGB(),
            origin='lower',
            extent=[0, 1, 0, 1],
            interpolation='bilinear',
            clip_path=patch
        )

    def add_sRGB_gamut(self):
        '''添加三角形的sRGB色域.'''
        xr, yr = 0.64, 0.33
        xg, yg = 0.30, 0.60
        xb, yb = 0.15, 0.06
        xw, yw = 0.3127, 0.3290

        patch = Polygon(
            xy=np.array([[xr, yr], [xg, yg], [xb, yb]]),
            transform=self.ax.transData,
            ec='k', fc='none', lw=1
        )
        self.ax.add_patch(patch)

        self.ax.text(xr, yr - 0.01, 'R', ha='center', va='top')
        self.ax.text(xg, yg + 0.01, 'G', ha='center', va='bottom')
        self.ax.text(xb, yb + 0.01, 'B', ha='right', va='bottom')
        self.ax.text(xw, yw - 0.02, 'D65', ha='center', va='top')
        self.ax.scatter(
            [xr, xg, xb, xw],
            [yr, yg, yb, yw],
            c='k', ec='k', fc='w', s=10
        )

        line = Line2D(
            [], [],
            ls='', marker='^',
            mec='k', mfc='none', ms=10,
            label='sRGB'
        )

        self.ax.legend(
            handles=[line],
            fontsize='medium',
            handletextpad=0.2,
            fancybox=False,
            framealpha=1
        )

    def add_ticks(self):
        '''添加波长刻度.'''
        major_len = 0.03
        minor_len = 0.5 * major_len
        label_len = 1.4 * major_len
        major_ticks = [380, *range(460, 601, 10), 620, 700]
        minor_ticks = range(380, 781, 5)

        xy = xyz_cc[['x', 'y']].to_numpy()
        dc = np.zeros_like(xy)
        dc[0] = xy[1] - xy[0]
        dc[-1] = xy[-1] - xy[-2]
        dc[1:-1] = xy[2:] - xy[:-2]
        dc = pd.DataFrame(dc, index=xyz_cc.index, columns=['dx', 'dy'])
        dc['dl'] = np.hypot(dc['dx'], dc['dy'])
        dc.loc[(dc.index < 430) | (dc.index > 660)] = np.nan
        dc = dc.ffill().bfill()
        dc['cos'] = -dc['dy'] / dc['dl']
        dc['sin'] = dc['dx'] / dc['dl']

        tick_df = pd.DataFrame({
            'x0': xyz_cc['x'],
            'x1': xyz_cc['x'] + major_len * dc['cos'],
            'x2': xyz_cc['x'] + minor_len * dc['cos'],
            'x3': xyz_cc['x'] + label_len * dc['cos'],
            'y0': xyz_cc['y'],
            'y1': xyz_cc['y'] + major_len * dc['sin'],
            'y2': xyz_cc['y'] + minor_len * dc['sin'],
            'y3': xyz_cc['y'] + label_len * dc['sin']
        })

        major_df = tick_df.loc[major_ticks]
        minor_df = tick_df.loc[minor_ticks]

        for row in major_df.itertuples():
            self.ax.plot(
                [row.x0, row.x1],
                [row.y0, row.y1],
                c='k', lw=0.6
            )
            self.ax.text(
                row.x3, row.y3, row.Index,
                ha='left' if row.Index > 520 else 'right',
                va='center',
                fontsize='x-small'
            )

        for row in minor_df.itertuples():
            self.ax.plot(
                [row.x0, row.x2],
                [row.y0, row.y2],
                c='k', lw=0.6
            )

    def draw(self):
        '''绘制色度图.'''
        self.set_axes()
        self.add_chromaticity()
        self.add_sRGB_gamut()
        self.add_ticks()

class rgDiagram(BaseDiagram):
    @staticmethod
    def generate_RGB():
        '''生成传给imshow的RGB数组.'''
        r = np.linspace(-1.5, 2, N)
        g = np.linspace(-1.5, 2, N).clip(1e-3, 2)
        r, g = np.meshgrid(r, g)
        G = np.ones_like(r)
        rgG = np.dstack((r, g, G))
        RGB = xyY_to_XYZ(rgG)
        RGB = XYZ_to_sRGB(RGB_to_XYZ(RGB))
        RGB = normalize_by_maximum(RGB)
        RGB = gamma_encoding(RGB.clip(0, 1))

        return RGB

    def set_axes(self):
        '''设置ax的范围, 刻度和标签.'''
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-2, 1.5)
        self.ax.set_ylim(-0.5, 3)
        self.ax.xaxis.set_major_locator(MultipleLocator(0.5))
        self.ax.yaxis.set_major_locator(MultipleLocator(0.5))
        self.ax.tick_params(labelsize='small')
        self.ax.grid(ls='--', lw=0.5, c='k')
        self.ax.set_xlabel('x', fontsize='large')
        self.ax.set_ylabel('y', fontsize='large', rotation=0)

    def add_chromaticity(self):
        '''添加填色后的色度区域.'''
        patch = Polygon(
            xy=rgb_cc[['r', 'g']],
            transform=self.ax.transData,
            ec='k', fc='none', lw=1
        )
        self.ax.add_patch(patch)

        self.ax.imshow(
            self.generate_RGB(),
            origin='lower',
            extent=[-1.5, 2, -1.5, 2],
            interpolation='bilinear',
            clip_path=patch
        )

    def add_XYZ_triangle(self):
        '''添加XYZ三角连线.'''
        rx, gx = 1.2749, -0.2777
        ry, gy = -1.7400, 2.7677
        rz, gz = -0.7430, 0.1408
        patch = Polygon(
            xy=np.array([[rx, gx], [ry, gy], [rz, gz]]),
            transform=self.ax.transData,
            ec='k', fc='none', lw=0.8
        )
        self.ax.add_patch(patch)

        self.ax.text(rx + 0.05, gx, 'X', ha='left', va='top')
        self.ax.text(ry - 0.05, gy, 'Y', ha='right', va='bottom')
        self.ax.text(rz - 0.05, gz, 'Z', ha='right', va='top')
        self.ax.scatter(
            [rx, ry, rz], [gx, gy, gz],
            c='k', ec='k', fc='w', s=10
        )

    def add_ticks(self):
        '''添加波长刻度.'''
        major_len = 0.08
        minor_len = 0.5 * major_len
        label_len = 1.4 * major_len
        major_ticks = [380, *range(480, 581, 10), 600, 700]
        minor_ticks = range(380, 701, 5)

        xy = rgb_cc[['r', 'g']].to_numpy()
        dc = np.zeros_like(xy)
        dc[0] = xy[1] - xy[0]
        dc[-1] = xy[-1] - xy[-2]
        dc[1:-1] = xy[2:] - xy[:-2]
        dc = pd.DataFrame(dc, index=rgb_cc.index, columns=['dx', 'dy'])
        dc['dl'] = np.hypot(dc['dx'], dc['dy'])
        dc.loc[(dc.index < 430) | (dc.index > 660)] = np.nan
        dc = dc.ffill().bfill()
        dc['cos'] = -dc['dy'] / dc['dl']
        dc['sin'] = dc['dx'] / dc['dl']

        tick_df = pd.DataFrame({
            'x0': rgb_cc['r'],
            'x1': rgb_cc['r'] + major_len * dc['cos'],
            'x2': rgb_cc['r'] + minor_len * dc['cos'],
            'x3': rgb_cc['r'] + label_len * dc['cos'],
            'y0': rgb_cc['g'],
            'y1': rgb_cc['g'] + major_len * dc['sin'],
            'y2': rgb_cc['g'] + minor_len * dc['sin'],
            'y3': rgb_cc['g'] + label_len * dc['sin']
        })

        major_df = tick_df.loc[major_ticks]
        minor_df = tick_df.loc[minor_ticks]

        for row in major_df.itertuples():
            self.ax.plot(
                [row.x0, row.x1],
                [row.y0, row.y1],
                c='k', lw=0.6
            )
            self.ax.text(
                row.x3, row.y3, row.Index,
                ha='left' if row.Index > 510 else 'right',
                va='center',
                fontsize='x-small'
            )

        for row in minor_df.itertuples():
            self.ax.plot(
                [row.x0, row.x2],
                [row.y0, row.y2],
                c='k', lw=0.6
            )

    def draw(self):
        '''绘制色度图.'''
        self.set_axes()
        self.add_chromaticity()
        self.add_XYZ_triangle()
        self.add_ticks()

if __name__ == '__main__':
    xy_diagram = xyDiagram()
    xy_diagram.draw()
    xy_diagram.save_fig('./fig/xy_chromaticity_diagram.png')

    rg_diagram = rgDiagram()
    rg_diagram.draw()
    rg_diagram.save_fig('./fig/rg_chromaticity_diagram.png')