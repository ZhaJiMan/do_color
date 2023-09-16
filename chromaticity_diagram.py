import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from common import *

xyz_cc = load_xyz_cc()
rgb_cc = load_rgb_cc()

def normal_direction(x, y):
    '''计算离散曲线每一点的法向cos和sin值.'''
    xy = np.column_stack((x, y))
    dxdy = np.zeros_like(xy)
    dxdy[0] = xy[1] - xy[0]
    dxdy[-1] = xy[-1] - xy[-2]
    dxdy[1:-1] = xy[2:] - xy[:-2]
    dx, dy = dxdy[:, 0], dxdy[:, 1]
    dl = np.hypot(dx, dy)
    dl[dl <= 0] = np.nan
    cos = -dy / dl
    sin = dx / dl

    return cos, sin

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
    def get_grid():
        '''生成RGB数组基于的网格.'''
        N = 256
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N).clip(1e-3, 1)
        x, y = np.meshgrid(x, y)

        return x, y

    def get_RGB(self):
        '''生成传给imshow的RGB数组.'''
        x, y = self.get_grid()
        Y = np.ones_like(y)
        xyY = np.dstack((x, y, Y))
        XYZ = xyY_to_XYZ(xyY)
        RGB = XYZ_to_sRGB(XYZ)
        RGB = move_toward_white(RGB)
        RGB = normalize_by_maximum(RGB)
        RGB = gamma_encoding(RGB)

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
            self.get_RGB(),
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

        x, y = xyz_cc['x'], xyz_cc['y']
        cos, sin = normal_direction(x, y)
        cs = pd.DataFrame(
            data=np.column_stack((cos, sin)),
            index=xyz_cc.index,
            columns=['cos', 'sin']
        )
        cs.loc[(cs.index < 430) | (cs.index > 660)] = np.nan
        cs = cs.ffill().bfill()
        cos, sin = cs['cos'], cs['sin']

        tick_df = pd.DataFrame({
            'x0': x,
            'x1': x + major_len * cos,
            'x2': x + minor_len * cos,
            'x3': x + label_len * cos,
            'y0': y,
            'y1': y + major_len * sin,
            'y2': y + minor_len * sin,
            'y3': y + label_len * sin
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
    def get_grid():
        '''生成RGB数组基于的网格.'''
        N = 256
        x = np.linspace(-1.5, 2, N)
        y = np.linspace(-1.5, 2, N).clip(1e-3, 2)
        x, y = np.meshgrid(x, y)

        return x, y

    def get_RGB(self):
        '''生成传给imshow的RGB数组.'''
        r, g = self.get_grid()
        G = np.ones_like(g)
        rgG = np.dstack((r, g, G))
        RGB = xyY_to_XYZ(rgG)
        RGB = XYZ_to_sRGB(RGB_to_XYZ(RGB))
        RGB = move_toward_white(RGB)
        RGB = normalize_by_maximum(RGB)
        RGB = gamma_encoding(RGB)

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
            self.get_RGB(),
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

        x, y = rgb_cc['r'], rgb_cc['g']
        cos, sin = normal_direction(x, y)
        cs = pd.DataFrame(
            data=np.column_stack((cos, sin)),
            index=rgb_cc.index,
            columns=['cos', 'sin']
        )
        cs.loc[(cs.index < 430) | (cs.index > 660)] = np.nan
        cs = cs.ffill().bfill()
        cos, sin = cs['cos'], cs['sin']

        tick_df = pd.DataFrame({
            'x0': x,
            'x1': x + major_len * cos,
            'x2': x + minor_len * cos,
            'x3': x + label_len * cos,
            'y0': y,
            'y1': y + major_len * sin,
            'y2': y + minor_len * sin,
            'y3': y + label_len * sin
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