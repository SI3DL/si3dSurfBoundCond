# surfBoundCond.py
# This code creates the sourface boundary conditions file for numerical simulations using SI3D. Follow description of functions in si3dInputs.py
# Copy Right Sergio A. Valbuena 2021
# UC Davis - TERC
# February 2021


# Library Import
import sys
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import datetime as Dt
import pytz
import pandas as pd
import time
import pickle
import math

# Declaration of mathematical constants
pi = math.pi

# Import functions
root = "C://Users/"
user = "SV"
func = "/Documents/Github/si3dInputs/"
FuncPath = root + user + func
sys.path.append(FuncPath)
from si3dInputs import surfbc4si3d
# from si3dInputs import LayerGenerator
del root, user, func, FuncPath

# --------------------- User variables declaration ----------------------------
# Specify the type of surface boundary condition desired
# Options are 'Preprocess', 'RunTime1', and 'RunTime2'
surfbcType = 'RunTime2'
# Specify the name of the site for the simulations
LakeName = 'LakeTahoe'
SimFolder = 'rho1L20H100f39W15'
# Specify the paths for existent data and where the file will be saved
root = "G:/My Drive/Lake_Tahoe/Projects/"
PathSave = root+"RotationalWeddeburnNumber/si3D/"+SimFolder
PathFile = root+"Upwelling_3DModel/Met_data/LakeTahoe_metData"
PathSecchiData = root+"Upwelling_3DModel/Secchi_data/Processed_0"
# Fields of data
Fields = ['WindSpeed','WindDir','P','AirTemp','RH','LWin','LWout','SWin','SWout','WaTemp']
# Chose between heatbudget methods, Only applicable if 'Preprocess' is chosen as surfbcType
# Options are: 'Chapra1995', 'AirSea', and 'TERC'
HeatBudgetMethod = 'Chapra1995'
# Define the starting and ending dates of simulation
StartDate = '2018-05-26 00:00:00'
EndDate = '2018-06-13 06:00:00'

# Define geographical location of site
DataTzone = 'UTC'
LocalTzone = 'US/Pacific'
lakeLatitude = 39.0968      # Latitude of Lake Tahoe N
lakeLongitude = 122.0324    # Longitude of Lake Tahoe W
LongHemis = 'West'          # West globe
offset_hr = 7.3             # [hrs] Offset of local time from UTC

# Define the height for temperature measurements
Ta_height = 4   # [m] from ground
# Define height for wind measurements
W_height = 4    # [m] from ground
# Define height for RH (relative humidity) measurements
RH_height = 4   # [m] from ground
# Define conversion method due to wind height
# Options are 1, 2 or 3.
wspd_conv = 2
# Degine the wind drag coefficient
WDragFlag = 'constant'
cw = 1.1*10**-3
# Define the method for the light attenuation coefficient eta
# Options are 1 for Martin & McCutcheon (1999) and 2 for TERC common method
etaMethod = 1
# Define secchidepth coefficient.
# Common practice is 1.18 for etaMethod = 1 and 1.7 for etaMethod = 2
secchiCoeff = 1.18
# Define if secchi depth is 'variable' or 'constant' during simulation.
secchiMethod = 'variable'
secchiDepth = 20
# Assume an attenuation coefficient of radiation in atmosphere
# Reference value is 0.76 from (from p. 359 Martin & McCutcheon, 1999)
att = 0.805
# Define longwave method. Where 1 is to use data and 2 is to use approximation from equation based on Air temperature.
LWinMethod = 2
LWoutMethod = 2
# Define if albedo is used or data from Metdata for shortwave out [SWout] is used
ialbedo = 'YES'
# Define albedo method to use IF ialbedo = 'YES'
# Use 1 for Pivovarov, (1972)
# Use 2 for Martin & McCutcheon, (1999)
# Use 3 for Neumann & Pierson, (1966). This option uses the matlab function
# Albedo.m, thus it is important to have it within the same folder.
albedo_eq = 2;
# Define ratio of Pa/P
Pa_P = 0.81        # Pa/P for 3149 m is 0.69; Linear interpolation says 0.81 for 1895 m.
# Define method for the estimation of vapor pressure [mbar] from RH and AirTemp
# Options are:
# 1) es = 6.11 * exp(17.3 * AirTemp / ( AirTemp + 237.3))
# 2) es = 6.11 * exp(7.5 * Lake.AirTemp / ( Lake.AirTemp + 237.3))
# 3 es = 10**(9.286-(2322.38/(AirTemp+273.15)))
esMethod = 1

# Desired time interval for numerical simulation
dt = 10  # [min]

if HeatBudgetMethod =='AirSea':
    # Please Define if clarke or bunker is used in cloudcor. This is the script for the correction for bulk long wave flux. Clarke only works for latitues < 50.
    opt_cloudcor = 'clarke'
    # Please define the method for the bulk longwave heat flux. Options are 'brunt' 'clark' 'hastenrath','efimova' 'bunker' 'anderson' 'swinbank' 'anderson' and 'berliand'. The last one considered to be the bestone.
    opt_blwhf = {'berliand'}
    dt_airsea = 1


# ----------------------------- CODE SECTION ----------------------------------
# Time zone management
LakeTzone = pytz.timezone(LocalTzone)
DatTzone = pytz.timezone(DataTzone)
# Varibles to limit data extent
StartDate = Dt.datetime.strptime(StartDate,'%Y-%m-%d %H:%M:%S')
EndDate = Dt.datetime.strptime(EndDate,'%Y-%m-%d %H:%M:%S')
StartDate = LakeTzone.localize(StartDate)
EndDate = LakeTzone.localize(EndDate)
StartDate = StartDate.astimezone(DatTzone)
EndDate = EndDate.astimezone(DatTzone)
StartDateNum = Dt.datetime.timestamp(StartDate)
EndDateNum = Dt.datetime.timestamp(EndDate)

dtsec = dt*60
Delta = EndDate - StartDate
Deltasec = Delta.days*24*60*60 + Delta.seconds
TimeSim = [StartDate + Dt.timedelta(0,t) for t in range(0,Deltasec+dtsec,dtsec)]
TimeNumSim = [Dt.datetime.timestamp(x) for x in TimeSim]
# Management of secchi disk method
if secchiMethod == 'constant':
    secchi = secchiDepth*np.ones(len(TimeNumSim))
else:
    os.chdir(PathSecchiData)
    data1 = pickle.load(open(LakeName+'_secchi','rb'))
    data2 = {}
    data3 = {}
    data2['Time'] = data1['Time']
    data2['TimeNum'] = [Dt.datetime.timestamp(x) for x in data1['Time']]
    data2['depth'] = data1['depth']
    dtsec = 1*60
    Delta = data2['Time'][-1] - data2['Time'][0]
    Deltasec = Delta.days*24*60*60 + Delta.seconds
    data3['Time'] = [data2['Time'][0] + Dt.timedelta(0,t) for t in range(0,Deltasec+dtsec,dtsec)]
    data3['TimeNum'] = [Dt.datetime.timestamp(x) for x in data3['Time']]
    data3['depth'] = np.interp(data3['TimeNum'],data2['TimeNum'],data2['depth'])
    secchi = np.interp(TimeNumSim,data3['TimeNum'],data3['depth'])
    del data1,data2, data3

# Trimming data for simulation period
os.chdir(PathFile)
data = pickle.load(open(LakeName,'rb'))
data['Time'] = [DatTzone.localize(x) for x in data['Time']]
del data['TimeNum']
data['Time'] = np.array(data['Time'])
idata1 = data['Time'] < StartDate
idata2 = data['Time'] > EndDate
for i in Fields:
    if np.isnan(data[i]).sum() != 0:
        print('Field '+ i + ' is not complete')
    else:
        data[i][idata1] = np.nan
        data[i][idata2] = np.nan
        data[i] = data[i][~np.isnan(data[i])]

# Change to local timezone
dateSim = [x.astimezone(LakeTzone) for x in TimeSim]

# Shotwave Radiation component of the HeatBudget
# Estimation of cloudiness on a daily basis
dateSimpd = pd.DatetimeIndex(dateSim)
year = dateSimpd[0].year
hr = dateSimpd.hour
mins = dateSimpd.minute
hr += mins/60
doy = dateSimpd.dayofyear
days = doy + hr/24
StdMeridian = offset_hr*15
Lat = lakeLatitude      # [Degrees] Local Latitude
Lon = lakeLongitude     # [Degrees] Local Longitude
Lsm = StdMeridian       #[Degrees] Local standard meridian
Hsc = 1390              # [W/m2] Constant solar radiation
theta = Lat*pi/180
hr = hr.values
mins = mins.values
doy = doy.values
days = days.values
r = 1 + 0.017 * np.cos((2*pi/365)*(186+doy))          # Relative earth-sun distance
d = 23.45 * pi/180 * np.cos((2*pi/365)*(172-doy))     # Declination of sun
if LongHemis == 'West':
    dts = -1/15 * (Lsm - Lon)
elif LongHemis == 'East':
    dts = 1/15 * (Lsm - Lon)
value = np.sin(theta)*np.sin(d)
value = value/(np.cos(theta)*np.cos(d))
tss = 12/pi * np.arccos(-value) + dts + 12
tsu = -tss + 2*dts + 24
gamma = np.zeros(len(hr))
idaytime = np.logical_and(hr > tsu, hr < tss)
gamma[idaytime] = 1
ibnoon = hr <= 12
ianoon = hr > 12
hbl = pi/12*(hr-1-dts)
hbl[ibnoon] = hbl[ibnoon] + pi
hbl[ianoon] = hbl[ianoon] - pi
hb = hbl
idat = hbl > 2*pi
hb[idat] = hb[idat] + 2*pi
idat = hbl < 0
hb[idat] = hb[idat] - 2*pi
he1 = pi/12*(hr-dts)
he1[ibnoon] = he1[ibnoon] + pi
he1[ianoon] = he1[ianoon] - pi
he = he1
idat = he1 > 2*pi
he[idat] = he[idat] + 2*pi
idat = he1 < 0
he[idat] = he[idat] - 2*pi
Ho = Hsc/(r**2)*(np.sin(theta)*np.sin(d)+12/pi*np.cos(theta)*np.cos(d)*(np.sin(he)-np.sin(hb)))*gamma
Ho = att*Ho;
idata = Ho < 0.0;
Ho[idata] = 0;

cc = np.zeros(len(data['SWin']))
ccl = cc
for i in range(doy.min(),doy.max()+1):
    iday = doy == i
    sro = np.mean(data['SWin'][iday])
    sra = np.mean(Ho[iday])
    if sro/sra > 1:
        print('The solar radiation of data is greater than the terrestrial, please check data. The ratio is  = '+ str(sro/sra))
    else:
        cc[iday] = math.sqrt((1-sro/sra)/0.67)
    ccl[iday] = 1 - sro/sra
cc[cc > 1] = 1
cc[cc < 0] = 0
ccl[ccl > 1] = 1
ccl[ccl < 0] = 0

Cl = ccl
# Estimation of albedo
if ialbedo == 'YES':
    if albedo_eq == 1:
        a0 = 0.02 + 0.01 * (0.5-Cl) * (1 - np.sin(pi*(days-81)/183))
        declination = 0.4093 * np.sin(2*pi * (days-79.75)/365)
        sin_alpha = np.sin(Lat*pi/180) * np.sin(declination) + np.cos(Lat*pi/180) * np.cos(declination) * np.cos(pi/12 * abs(hr - 12))
        sin_alpha = np.maximum(0,sin_alpha)
        albedo = a0 / (a0 + sin_alpha);
    elif albedo_eq == 2:
        A = 1.18
        B = -0.77
        declination = 0.4093 * np.sin( 2*pi*(days-79.75)/ 365)
        sin_alpha = np.sin(Lat*pi/180) * np.sin(declination) + np.cos(Lat*pi/180) * np.cos(declination) * np.cos(pi/12 * abs(hr - 12))
        sin_alpha = np.maximum(0.0,sin_alpha)
        asa = np.maximum(0.01, np.arcsin(sin_alpha))
        albedo = A * (57.3 * asa)**B
        dum = np.logical_and(Cl >= 0.1, Cl < 0.5)
        albedo[dum] = 2.20 * (57.3 * asa[dum])**(-0.97)
        del dum
        dum = np.logical_and(Cl >= 0.5, Cl < 0.9)
        albedo[dum] = 0.95 * (57.3 * asa[dum] )**(-0.75)
        del dum
        dum = Cl >=0.9
        albedo[dum] = 0.33 * (57.3 * asa[dum])**(-0.45)
        del dum
        albedo = np.minimum(1., albedo)
    elif albedo_eq == 3:
        [albedo,RefAng] = Albedo(days,Lon)
        print('This section requires the use of a matlab function and is still under development')
    Hswn = data['SWin']*(1-albedo)
elif ialbedo == 'NO':
    Hswn = data['SWin'] - data['SWout']

# Attenuation coefficient of ligth penetration based on the secchi depth
# Correlation shown in p. 386 in Martin and McCutcheon (1999)
eta = np.zeros(len(secchi))
if etaMethod == 1:
    eta = secchiCoeff/secchi**0.73
elif etaMethod == 2:
    eta = secchiCoeff/secchi

# Correction for wind speed height
if wspd_conv == 1:
    zo = 0.0002
    Wspd = data['WindSpeed']*(math.log(10/zo)/math.log(W_height/zo))
elif wspd_conv == 2:
    vonkart = 0.41
    slope = 0.0504
    Wspd = data['WindSpeed']*1/(1-slope/vonkart*math.log(10/W_height))
elif wspd_conv == 3:
    print()
     # Monin & Obukhov expression
     # U_10 - U_1 = u_*/kappa*(ln(z10/z1)-gamma(z10/L)-gamma(z1/L))
     # u_* = sqrt(\tau/\rho)^0.5
u = -Wspd * np.sin(data['WindDir']*pi/180)
v = -Wspd * np.cos(data['WindDir']*pi/180)

# Wind drag coefficient
if WDragFlag == 'constant':
    cw = cw*np.ones(len(u))
elif WDragFlag == 'variable':
    cw = 0.0015*(1/(1+np.exp((12.5-Wspd)/1.56)))+0.00104

# Longwave radiation in
if LWinMethod == 1:
    Hlwin = data['LWin']
elif LWinMethod == 2:
    Hlwin = 0.937e-5 * 0.97 * 5.67e-8 * ((data['AirTemp'] + 273.16)**6) * (1 + 0.17*Cl)

# Longwave radiation out
if LWoutMethod == 1:
    Hlwout = data['LWout']
elif LWoutMethod == 2:
    Hlwout = 0.97 * 5.67e-8 * (data['WaTemp'] + 273.15)**4

# Relative Humidity
RH = data['RH']
# Air Temperature
Ta = data['AirTemp']
# Barotropic pressure
Pa = data['P']*100
# Surface Water Temperature
WaTemp = data['WaTemp']

if surfbcType == 'Preprocess':
    surfbc4si3d(LakeName,surfbcType,days,hr,mins,year,dt,PathSave,HeatBudgetMethod,eta,Hswn,Hlwin,Hlwout,Ta,Pa,RH,Cl,cw,u,v,WaTemp,Pa_P,esMethod)
elif surfbcType == 'RunTime1':
    surfbc4si3d(LakeName,surfbcType,days,hr,mins,year,dt,PathSave,eta,Hswn,Ta,Pa,RH,Cl,cw,u,v)
elif surfbcType == 'RunTime2':
    surfbc4si3d(LakeName,surfbcType,days,hr,mins,year,dt,PathSave,eta,Hswn,Ta,Pa,RH,Hlwin,cw,u,v)


secSim = EndDate - StartDate
secSim = secSim.days*24*60*60 + secSim.seconds
print('The duration of the simulations is ',secSim,' seconds')
