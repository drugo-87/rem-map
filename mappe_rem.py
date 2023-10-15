import numpy as np
import math
import xarray as xr
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap, LogNorm
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

## Feeding

parser = argparse.ArgumentParser()

parser.add_argument("-v", "--variable", help="the current variable implemented are ['temp_val', 'rain_rate_max', 'slp', 'wind_speed', 'rel_hum']")
parser.add_argument("-d", "--date", help="Use bash command 'date +%Y%m%d_%H%M' ")
parser.add_argument("-i", "--input", required=False, help="the data-input folder", default='data/')
parser.add_argument("-o", "--output", required=False, help="the output folder", default='maps/')
#parser.print_help()
args = parser.parse_args()


#exit()
print(args.input, args.output)
# Defining variables
variabili =['temp_val', 'rain_rate_max', 'slp', 'wind_speed', 'rel_hum']

if args.variable not in variabili:
    print(f'{args.variable} is not present in the variable list')
    exit()

## Defining functions
def compute_distance_weight(coord_stations, coord_grid):
    dist = abs(scipy.spatial.distance.cdist(coord_stations,coord_grid).T[0])/10000 # pick the appropriate distance metric 
    dist_w = np.exp(-dist)
    return dist_w
def compute_elevation_weight(z_stations, z_grid):
    if np.isnan(z_grid)==False:
        disl = abs(z_stations - z_grid)/500 # pick the appropriate distance metric 
        dist_z = np.exp(-disl[0])
    else:
        dist_z=1    
    return dist_z

def compute_weighted_var(d_w, z_w, T):
    weights=d_w*z_w*T
    # for i in range(len(T)):
    #     T_weighted = d_w[i]*z_w[i]*T[i]
    weighted_var=np.nansum(weights)/(np.nansum(d_w*z_w))
    # print(T_weighted.sum(),np.sum(d_w.T*z_w),T_mean)
    return weighted_var

##########
# Reading input
try:
  gf=pd.read_csv(f'{args.input}/geo-data/coordinate_stazione.csv')
except:
  print("Something went wrong, can't find coordinates file")
  exit()
try:
  dem = xr.open_dataset(f'{args.input}/geo-data/dem500.nc')
except:
  print("Something went wrong, can't find dem file")
  exit()
try:
  df = pd.read_csv(f'{args.input}/dati_{args.date}.csv',na_values='NA')
except:
  print("Something went wrong, can't find data file")
  exit()
  
#gf=pd.read_csv('/home/drugo/Projects/rem/coordinate_stazione.csv')
#dem = xr.open_dataset('/home/drugo/Projects/rem/dem100.nc')
#df = pd.read_csv('/home/drugo/Projects/rem/dati_20230518_1820_temporale.csv',na_values='NA')


## Extract coordinates and variables
ids = np.array([df.id[:]])
lon = np.array([gf.lon_32[:]])
lat = np.array([gf.lat_32[:]])
elev=np.array([df.quota[:]])
grid_lat = dem.y.values
grid_lon = dem.x.values
if args.variable == 'temp_val':
    var2int = np.array([df[args.variable][:]])[0]+273
else:
    var2int = np.array([df[args.variable][:]])[0]
    
# coordinates station
station_coord = np.dstack([lon[:], lat[:]])[0]
print(station_coord.shape)
station_coord[:3,:]
interp_var = np.full([dem.Band1.values.shape[0],dem.Band1.values.shape[1]],np.nan)

# Interpolation
for i in range(dem.Band1.values.shape[0]):
    for j in range(dem.Band1.values.shape[1]):
        if np.isnan(dem.Band1.values[i][j])==False:
            d_w=compute_distance_weight(station_coord,np.array([[grid_lon[j],grid_lat[i]]]))
            if args.variable=='temp_val':
                z_w=compute_elevation_weight(elev,dem.Band1.values[i][j])
            else:
                z_w=1
            interp_var[i,j]=compute_weighted_var(d_w, z_w, var2int)
        else:
            interp_var[i,j]=np.nan
    

# Plotting stage
if args.variable == 'temp_val':
    #levels = np.linspace(-20, 40, 31)
    cmap_T = [
    '#B400B4', '#9600C8', '#A064DC', '#BE8CC8', '#E1AFC3', '#99DDB4', '#26B6A2', '#1EC5B6',
    '#16D5CE', '#0EE1E5', '#05E5F6', '#00DE89', '#00C638', '#04B400', '#21C000', '#42CC00', 
    '#67D700', '#8FE300', '#BAEF00', '#E9FA00', '#FDF000', '#F7D500', '#F2BB00', '#EDA100', 
    '#E88900', '#E16600', '#D93E00', '#D21800', '#C8000F', '#B8003C', '#A90062', '#990080'
    ]
    levels = np.array([-20,-18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 
            12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40])
    palette = LinearSegmentedColormap.from_list('palette_temp', cmap_T, N=len(cmap_T))
    norm = mpl.colors.BoundaryNorm(levels, palette.N, extend='both')
    #palette="gist_ncar"
    interp_var=interp_var-273
elif args.variable == 'rain_rate_max':
    cmap_prp = 'BuPu'
    #cmap_prp = ['#C0C0C0', '#D6E2FF', '#B5C9FF', '#8EB2FF', '#7F96FF', '#6370F7', '#009E1E' , '#3CBC3D','#B3D16E', '#B9F96E', 
    #'#FEFEA0', '#FFF914', '#FFA30A', '#E50000', '#BD0000', '#D464C3',]# '#B5199D', '#840094', '#B4B4B4', '#8C8C8C', '#5A5A5A', '#323232']
#    levels = np.array([0.2, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 125, 150, 175, 200, 250, 300])
    #levels = np.array([0.2, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 125, 150, 175, 200, 250, 300])
    levels = np.array([1, 2, 3, 4,  5,  7.5, 10, 15, 20,  ])#  30,  35, 40,  50,  75,  100]) #,  90, 100, 125, 150, 175, 200, 250, 300
    #palette = LinearSegmentedColormap.from_list('palette_prp', cmap_prp, N=len(cmap_prp))
    palette = cmap_prp
    #norm =LogNorm(vmin=levels[0], vmax=levels[-1], clip=1) #= BoundaryNorm(levels, palette.N)
    #orm = mpl.colors.BoundaryNorm(levels, palette.N, extend='both')
    #norm = mpl.colors.BoundaryNorm(levels, palette.N, extend='both')

#    levels = np.linspace(0, 20, 21)
#    palette="Blues"
elif args.variable == 'rel_hum':
    cmap_hr = ['#C0C0C0', '#C8000F', '#E16600', '#F2BB00', '#E9FA00', '#67D700', '#04B400', '#00F5E9', '#16D5CE', '#2CA890', '#1D6DA5', '#4C00B3']
    levels = np.array([1,10,20,30,40,50,60,70,80,90,100])
    palette = LinearSegmentedColormap.from_list('palette_hr', cmap_hr, N=len(cmap_hr))
    norm = mpl.colors.BoundaryNorm(levels, palette.N, extend='both')
       
#    levels = np.linspace(0, 100, 11)
#    palette="Blues"
elif args.variable == 'wind_speed':
    cmap_wind = ['#C0C0C0', '#EEFFFF', '#96D2FA', '#50A5F5', '#196EE1', '#00D278', '#00A000', '#E11400', '#A50000', '#FF00FF', '#FFAAFF', '#FF9600', '#AAAAAA', '#777777']
    #levels = np.array([0.5,2.5,5,11,19,30,39,50,61,74,87,102,117])
    levels = np.array([2.5,5,10,15,20,25,30,35,40,45,50,55,60])
    palette = LinearSegmentedColormap.from_list('palette_wind', cmap_wind, N=len(cmap_wind))
    norm = mpl.colors.BoundaryNorm(levels, palette.N, extend='both')
    
    
#    levels = np.linspace(0, 25, 26)
#    palette="BuPu"
    # levels = np.linspace(-20, 40, 31)
Novellara = gf[['stazione','lon_32','lat_32']].loc[(gf['stazione']=='Novellara')].values[0]
ReggioEmilia = gf[['stazione','lon_32','lat_32']].loc[(gf['stazione']=='Reggio Emilia')].values[0]
CMonti = gf[['stazione','lon_32','lat_32']].loc[(gf['stazione']=='Carnola - C.Monti')].values[0]
fname = f'{args.input}/geo-data/comuni_reggio_Emilia.shp'
print(Novellara)


def main():
    fig = plt.figure(figsize=(dem.Band1.shape[0]/75*5,dem.Band1.shape[1]/75*5))
    shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                ccrs.epsg(32632), edgecolor='k',alpha=0.1)

    # Setup a global EckertIII map with faint coastlines.
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.epsg(32632))
    ax.add_feature(shape_feature)

    # ax.set_extent([10, 11, 44.2,45], crs=ccrs.PlateCarree())
    # ax.coastlines('110m', alpha=0.1)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    
   


#    # Add colourful filled contours.
#    filled_c = ax.contourf(grid_lon, grid_lat, interp_var, levels=levels, cmap='Blues',vmin=levels[0], vmax=levels[-1],extend='both')
    

    print(type(interp_var))
    
    # And black line contours.
    line_c = ax.contour(grid_lon, grid_lat, (interp_var), levels=levels, vmin=levels[0], vmax=levels[-1],#filled_c.levels,
                        colors=['black'],linewidths=0.1,extend='both')
    # Add colourful filled contours.
    filled_c = ax.contourf(grid_lon, grid_lat, interp_var,  cmap=palette,levels=levels, vmin=levels[0], vmax=levels[-1],extend='both')
    filled_c.cmap.set_under('white',alpha=0)
    #CS3.cmap.set_over('cyan')
    
    # Add a colorbar for the filled contour.
    fig.colorbar(filled_c, orientation='vertical',extend='both')
#    fig.colorbar(filled_c, cax=sub_ax1)
   # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=palette),
    #         orientation='vertical',)
             #label="Discrete intervals with extend='both' keyword")


                        
    ax.scatter(Novellara[1], Novellara[2],color='k', edgecolor='k')
    ax.scatter(ReggioEmilia[1], ReggioEmilia[2],color='k', edgecolor='k')
    ax.scatter(CMonti[1], CMonti[2],color='k', edgecolor='k')

    plt.text(Novellara[1]-1000, Novellara[2]-1000, Novellara[0],
         horizontalalignment='right',
         transform=ccrs.epsg(32632))
    
    plt.text(ReggioEmilia[1], ReggioEmilia[2]+750, ReggioEmilia[0],
         horizontalalignment='center',
         transform=ccrs.epsg(32632))
    plt.text(CMonti[1], CMonti[2]+500, CMonti[0][10:],
         horizontalalignment='left',
         transform=ccrs.epsg(32632))
         
         


    # Use the line contours to place contour labels.
    ax.clabel(
        line_c,  # Typically best results when labelling line contours.
        colors=['black'],
        manual=False,  # Automatic placement vs manual placement.
        inline=True,  # Cut the line where the label will be placed.
        fmt=' {:.0f} '.format,  # Labes as integers, with some extra space.
    )
    fig.savefig(f'{args.output}/mappa_{args.variable}_{args.date}.png',bbox_inches='tight', transparent=False) #pad_inches=0.1,
        # facecolor='auto', edgecolor='auto',
        # backend=None,)

    # plt.show()
    
    # return filled_c.levels


if __name__ == '__main__':
    main()
