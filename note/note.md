- 三基色的原理：基于 cone fundamentals，可以用三种颜色线性组合出任意的 LMS 三刺激值。
- lumiosity efficiency function
- 辐射学和光度学的物理量、单位定义
- luminance、brightness、lightness 的区别
- 格拉斯曼定律指出颜色的叠加是线性的
- 颜色匹配实验导出 CMF 和色度坐标，意义和单位
- RGB 空间里离原点的距离正比于三基色的辐射强度（缩放因子）
- RGB 空间里两点连线上一点对应于 rg 平面里两点连线上一点。注意占连线的比例并不直接相等，并且要求两点的 R + G + B 同时为正或为负。选取合适的三基色以满足这一条件，好让色度图有几何意义。
- 为了使 CMF 为正数，利用上 V，利用亮度关系、切线和零亮度面将 RGB 空间线性变换为 XYZ 空间。
- XYZ、RGB 和 LMS 三刺激值都可以用线性变换得出，确保颜色的唯一性。
- 证明两种颜色按功率比例混合的结果在色度图上为连线上一点。
- 实际光源不能产生负数光，色域怎么确定。

具体亮度并不是特别重要，因此色度图采用色度坐标，缺少了亮度信息。

定义 XYZ 空间的目的：
- 不需要像 RGB 空间那样必须确定三基色的波长。
- 人类可感颜色的坐标值都是非负数。

那为什么不直接用 LMS 空间呢？因为 1931 年那会儿不能得出 cone fundamentals，近些年才有实验测量。

本来 Y 和 RGB 的关系应该通过拟合得到。但如果使用的是同一套数据（1921 V 和 1931 CMF），按 CMF 的定义（匹配光谱光的亮度），就可以直接决定变换矩阵第二行的系数为亮度系数。总结一下如何得到矩阵：

- Y 对应 V
- XYZ 对波长的积分（和）相等，或者说 RGB (1, 1, 1) 映射到 XYZ (k, k, k)。
- rg 平面上 X 和 Z 点位于零亮度面上，其它边与可视范围相切。

颜色空间里所有能物理实现的颜色被称为色域（gamut）。现实世界里光源不能直接发出负值的基色光。

## XYZ 空间

制定 XYZ 空间的原则：

- 认为格拉斯曼定律成立：认为颜色可以由三基色混合表示，两种颜色混合后的颜色等于对应三基色量的线性叠加。
- 亮度系数比例为 `1:4.5907:0.0601`，由 CMF 对 LEF 的最小二乘拟合得到。
- 所有真实颜色的坐标都为正数。
- 新基色的单位能让 EEW 的色度坐标都相等。
- 让新的色度坐标限于 `(1, 0, 0)`、`(0, 1, 0)`、`(0, 0, 1)` 第一象限的等边三角形中。
- 让 z 色匹配函数的长波区域数值为零，减少计算负担。

精简一下：

- 让 y 色匹配函数等于 LEF，将 X 和 Z 基色的矢量位于 alychne 面上。目的是减少计算。
- rg 色度图上 XYZ 基矢量的点构成的三角形刚好包住 spectrum locus 和 purple line，使 XYZ 坐标值均为正数。（色度图上包住就能保证三维空间也包住，想象一下）。同时这个三角形不要太大，浪费空间。求解变换的过程主要就是确定这个三角形的色度坐标。
- Guild 系统里选取 700 nm 作为红基色能使 r 色匹配函数在 660 nm 后数值均为零。在色度图上体现为 650 nm 后光谱轨迹近乎为直线 r + g = 1。让 XY 的连线几乎与该线重合，那么长波区域的 z 坐标就会出现很多零，减少计算量。实际上这个区域会出现 b 值为特别小的负数的情况，导致 r + g 略大于 1。因此也要让 XY 连线的斜率略大于 1。

## 参考链接

CIE:

- Color Vision and Colorimetry THEORY and APPLICATIONS
- https://en.wikipedia.org/wiki/CIE_1931_color_space
- https://yuhaozhu.com/blog/cmf.html

XYZ:

- https://horizon-lab.org/colorvis/xyz.html
- https://philservice.typepad.com/f_optimum/2016/04/the-wright-guild-experiments-and-the-development-of-the-cie-1931-rgb-and-xyz-color-spaces.html
- How the CIE 1931 Color-Matching Functions Were Derived from Wright–Guild Data
- The Heights of the CIE Colour-Matching Functions
- CIE Method for Calculatina Tristimulus Values
- http://yamatyuu.net/other/color/cie1931xyzrgb/xyz.html
- http://brucelindbloom.com/index.htm

sRGB:

- https://en.wikipedia.org/wiki/SRGB
- https://fujiwaratko.sakura.ne.jp/infosci/colorspace/colorspace2_e.html
- https://fujiwaratko.sakura.ne.jp/infosci/colorspace/rgb_xyz_e.html
- https://www.bilibili.com/video/BV1wC4y1p7Xr
- https://www.w3.org/Graphics/Color/sRGB.html
- https://www.w3.org/Graphics/Color/srgb
- https://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf
- 色度図の着色

数据来源：

- http://www.cvrl.org/
- https://github.com/colour-science/colour

画图：
- https://colour.readthedocs.io/en/develop/generated/colour.plotting.plot_chromaticity_diagram_CIE1931.html
- https://blog.csdn.net/weixin_43194305/article/details/115468614
- https://scipython.com/blog/converting-a-spectrum-to-a-colour/