# =============================================================================
# Imports 
# =============================================================================
import numpy as np
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature as cnef
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from pathlib import Path
# =============================================================================

# =============================================================================
# Functions
# =============================================================================
# ref: https://scitools.org.uk/cartopy/docs/latest/gallery/lines_and_polygons/always_circular_stereo.html#sphx-glr-gallery-lines-and-polygons-always-circular-stereo-py
def add_circle_boundary(ax):
    # Compute a circle in axes coordinates, which will be used as a boundary
    # for the map
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
# =============================================================================


# =============================================================================
# Plot Map
# =============================================================================
#define map projection
proj = ccrs.NearsidePerspective(central_longitude=-60, 
                                central_latitude=-65, 
                                satellite_height=1500250)

#set up figure space
fig = plt.figure(figsize=(3,3), dpi=1200)
ax = plt.axes(projection=proj)

#plot map features
ax.add_feature(cnef('physical', 'land', '50m', edgecolor='face', linewidth=0.5, 
                    facecolor='grey', zorder=0), zorder=10, edgecolor='black')

ax.add_feature(cnef('physical', 'antarctic_ice_shelves_polys', '50m', edgecolor='face', 
                    linewidth=0.5, facecolor='#fafafa', zorder=0), zorder=1, edgecolor='black')    
    
ax.add_feature(cnef('physical', 'ocean', '50m', edgecolor='face', linewidth=0.5,
                    facecolor='#83B1DA', zorder=0, alpha=0.8), zorder=0, edgecolor='black')        


#add gridlines
gl = ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False, 
                  linewidth=0.5, color='grey', alpha=.75, linestyle='--', zorder=9)

#adjust styling
gl.xlabel_style = {'size': 10, 'weight': 'normal'}
gl.ylabel_style = {'size': 10, 'weight': 'normal', 'rotation': 'horizontal'}
gl.xlocator = mticker.FixedLocator(np.arange(-180,180,20))
gl.ylocator = mticker.FixedLocator(np.arange(-90,90,10))
# gl.xlocator = mticker.FixedLocator([20, 0,-20,-40,-60,-80,-100,-120,-140])
# gl.ylocator = mticker.FixedLocator([-40,-50,-60,-70])
# gl.xlocator = mticker.FixedLocator([-60,-62.5,-65,-67.5,-70,-72.5,-75])
# gl.ylocator = mticker.FixedLocator([-63,-64,-65,-66,-67,-68,-69])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
# =============================================================================


# =============================================================================
# Save Figure as PDF for Compiling in Adobe Illustrator
# =============================================================================
current_directory = Path.cwd()
absolute_path = Path("analysis/publication figures/Figure 1 - Map/")
filename = Path("Fig1a_WAPGlobalContextMap.pdf")
savepath = str(current_directory / absolute_path / filename)

plt.savefig(savepath, transparent=True)
# =============================================================================









