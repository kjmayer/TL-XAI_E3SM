import xarray as xr
import pandas as pd
import numpy as np
from numpy.polynomial import polynomial

import datetime as dt
import calendar
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.dpi'] = 150
dpiFig = 300.

#%% >>>>> Plot Accuracy & Loss during Training >>>>>
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
            ax.xaxis.set_ticks([])

def plot_results(history, exp_info, showplot=True):
    
    n_epochs, hiddens, lr_init, batch_size, network_seed, patience, ridge = exp_info
    
    trainColor = 'k'
    valColor = (141/255,171/255,127/255,1.)
    FS = 14
    plt.figure(figsize=(15, 7))
    
    #---------- plot loss -------------------
    ax = plt.subplot(2,2,1)
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
    ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35)

    plt.plot(history.history['sparse_categorical_accuracy'], 'o', color=trainColor, label='Training',alpha=0.6)
    plt.plot(history.history['val_sparse_categorical_accuracy'], 'o', color=valColor, label='Validation',alpha=0.6)
    plt.vlines(len(history.history['val_sparse_categorical_accuracy'])-(patience+1),-10,np.max(history.history['loss']),'k',linestyle='dashed',alpha=0.4)

    plt.title('ACCURACY')
    plt.xlabel('EPOCH')
    plt.xticks(np.arange(0,n_epochs+20,20),labels=np.arange(0,n_epochs+20,20))
    plt.yticks(np.arange(.4,1.1,.1),labels=[0.4,0.5,0.6,0.7,0.8,0.9,1.0]) # 0,0.1,0.2,0.3,
    plt.grid(True)
    plt.legend(frameon=True, fontsize=FS)
    plt.xlim(-2, n_epochs)
    plt.ylim(.4,1)
    
    # ---------- plot accuracy -------------------
    ax = plt.subplot(2,2,2)
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
    ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35)
    
    plt.plot(history.history['loss'], 'o', color=trainColor, label='Training',alpha=0.6)
    plt.plot(history.history['val_loss'], 'o', color=valColor, label='Validation',alpha=0.6)
    plt.vlines(len(history.history['val_loss'])-(patience+1),0,1,'k',linestyle='dashed',alpha=0.4)
    plt.title('PREDICTION LOSS')
    plt.xlabel('EPOCH')
    plt.legend(frameon=True, fontsize=FS)
    plt.xticks(np.arange(0,n_epochs+20,20),labels=np.arange(0,n_epochs+20,20))
    plt.yticks(np.arange(0,1.1,.1),labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.ylim(0,1)
    plt.grid(True)
    plt.xlim(-2, n_epochs)
    
    # ---------- report parameters -------------------
    plt.subplot(2, 2, 3)
    plt.ylim(0, 1)
    
    text = (
            "\n"
            + f"NETWORK PARAMETERS\n"
            + f"  Number of Epochs     = {n_epochs}\n"
            + f"  Hidden Layers        = {hiddens}\n"
            + f"  Learning Rate        = {lr_init}\n"
            + f"  Network Seed   = {network_seed}\n"
            + f"  Batch Size     = {batch_size}\n"
            + f"  Ridge          = {ridge}\n"
            )

    plt.text(0.01, 0.95, text, fontfamily='monospace', fontsize=FS, va='top')
    
    plt.axis('off')

    # ---------- Make the plot -------------------
    #plt.tight_layout()
    if showplot==False:
        plt.close('all')
    else:
        plt.show()
        

def is_month(data, months):
    i_timedim = np.where(np.asarray(data.dims) == 'time')[0][0]
    if i_timedim == 0:
        data = data[data.time.dt.month.isin(months)]
    elif i_timedim == 1:
        data = data[:,data.time.dt.month.isin(months)]
    return data


def detrend_members(data, ensmean_data, npoly=3):
    '''
    detrend ensemble member using polynomial fit (for each doy) to the ensemble mean
    
    data: [member, time, lat, lon] or [member, time]
        ensemble members to detrend 
    
    ensmean_data: [time, lat, lon] or [time]
        ensemble mean 
    
    npoly: [int] 
        order of polynomial, default = 3rd order
    '''
    
    # stack lat and lon of ensemble mean data
    if len(ensmean_data.shape) == 3:
        ensmean_data = ensmean_data.stack(z=('lat', 'lon'))
 
    # stack lat and lon of member data & grab doy information
    if len(data.shape) >= 3:
        data = data.stack(z=('lat', 'lon'))
    temp = data['time.dayofyear']
    
    # grab every Xdoy from ensmean, fit npoly polynomial
    # subtract polynomial from every Xdoy from members
    detrend = []
    for label,ens_group in ensmean_data.groupby('time.dayofyear'):
        Xgroup = data.where(temp == label, drop = True)
        
        curve = polynomial.polyfit(np.arange(0, ens_group.shape[0]), ens_group, npoly)
        trend = polynomial.polyval(np.arange(0, ens_group.shape[0]), curve, tensor=True)
        if len(ensmean_data.shape) >= 2: #combined lat and lon, so now 2-3
            trend = np.swapaxes(trend,0,1) #only need to swap if theres a space dimension

        diff = Xgroup - trend
        detrend.append(diff)

    detrend_xr = xr.concat(detrend,dim='time').unstack()
    detrend_xr = detrend_xr.sortby('time')
    
    return detrend_xr


def balance_classes(Xdata, Ydata):
    # subset z500val (& precipval) to equal 0s and 1s
    nzero = np.shape(np.where(Ydata==0)[0])[0]
    none  = np.shape(np.where(Ydata==1)[0])[0]
    izero = np.where(Ydata==0)[0]
    ione  = np.where(Ydata==1)[0]

    if none != nzero:
        if none > nzero:
            isubset_one = np.random.choice(ione,size=nzero,replace=False)
            inew = np.sort(np.append(izero,isubset_one))

        elif none < nzero:
            isubset_zero = np.random.choice(izero,size=none,replace=False)
            inew = np.sort(np.append(isubset_zero,ione))

        Ydata  = Ydata.isel(time = inew,drop=True) 
        Xdata  = Xdata.isel(time = inew,drop=True)
        
    else:
        inew = []
        
    return Xdata,Ydata,inew




def split_SDbias(trainmems, valmem, testmem, months, lead):
    zpath = '/glade/derecho/scratch/kjmayer/DATA/E3SMv2/Z500/'
    ppath = '/glade/derecho/scratch/kjmayer/DATA/E3SMv2/PRECT/'
    biasstr = '60Eshift'

    print('files do not exist - loading data & saving')
    # ----------- LOAD data & SHIFT time -------------
    # open detrended & running mean-ed data

    # ------ X LOAD & SHIFT --------
    if len(trainmems) > 1:
        for m,mem in enumerate(trainmems):
            pfiname = 'PRECT'+biasstr+'_mem'+str(mem)+'_7daymean_1950-2014_20S-20N_regrid2.5x2.5_polydetrend_allmems.nc'
            X1xr = xr.open_dataarray(ppath+pfiname)[:-6][:-lead]
    
            if m == 0:
                X1trainxr_NDJF = X1xr[X1xr.time.dt.month.isin(months)]
            else:
                X1trainxr_NDJF = xr.concat([X1trainxr_NDJF,X1xr[X1xr.time.dt.month.isin(months)]],dim='mem')
    
            del X1xr
    
            # ------ Y LOAD & SHIFT --------
            zfiname = 'Z500_mem'+str(mem)+'_7daymean_1950-2014_30-60Nx170-240E_regrid2.5x2.5_polydetrend_allmems.nc'
            Y1xr = xr.open_dataarray(zpath+zfiname)
    
            if m == 0:
                # ----- grab precip time + LEAD -----
                # ----------- (CESM doesn't include Feb 29th for leap years)
                Y1_leadtime = []
                for d in np.arange(len(X1trainxr_NDJF.time)):
                    temp = pd.to_datetime(X1trainxr_NDJF.time)[d] + dt.timedelta(days = lead)
                    # if leap year and the day is after feburary 28th and before april (lead is okay for Nov-Dec)
                    if calendar.isleap(temp.year) and temp.month in [2,3] and ((temp - pd.to_datetime(str(temp.year)+'-02-28')) > dt.timedelta(days = 0)):
                        temp = pd.to_datetime(X1trainxr_NDJF.time)[d] + dt.timedelta(days = lead+1)
                    Y1_leadtime.append(temp)
                Y1_leadtimexr = xr.DataArray(np.array(Y1_leadtime),dims='time',coords={'time':np.array(Y1_leadtime)})
                # ----------------
                Y1trainxr_NDJFM = Y1xr.where(Y1xr['time'] == Y1_leadtimexr, drop=True)
            else:
                Y1trainxr_NDJFM = xr.concat([Y1trainxr_NDJFM, Y1xr.where(Y1xr['time'] == Y1_leadtimexr, drop=True)],dim='mem')
    
            del Y1xr

        # get training & standardize
        X1trainxr_NDJF = X1trainxr_NDJF.stack(s=('mem','time')).transpose('s','lat','lon')
        X1trainxr_NDJF = X1trainxr_NDJF.reset_index(['s'])
    
        X1train_mean = X1trainxr_NDJF.mean('s')#['s','lat','lon'])
        X1train_std  = X1trainxr_NDJF.std('s')#['s','lat','lon'])

        # get training & standardize
        Y1trainxr_NDJFM = Y1trainxr_NDJFM.stack(s=('mem','time'))
        Y1train_med = Y1trainxr_NDJFM.quantile(q=0.5,dim='s',keep_attrs=True)
        
    elif len(trainmems) == 1:
        print('made it!')
        pfiname = 'PRECT'+biasstr+'_mem'+trainmems[0]+'_7daymean_1950-2014_20S-20N_regrid2.5x2.5_polydetrend_allmems.nc'
        X1xr = xr.open_dataarray(ppath+pfiname)[:-6][:-lead]
        X1trainxr_NDJF = X1xr[X1xr.time.dt.month.isin(months)]            

        # ------ Y LOAD & SHIFT --------
        zfiname = 'Z500_mem'+trainmems[0]+'_7daymean_1950-2014_30-60Nx170-240E_regrid2.5x2.5_polydetrend_allmems.nc'
        Y1xr = xr.open_dataarray(zpath+zfiname)

        # ----- grab precip time + LEAD -----
        # ----------- (E3SM doesn't include Feb 29th for leap years)
        Y1_leadtime = []
        for d in np.arange(len(X1trainxr_NDJF.time)):
            temp = pd.to_datetime(X1trainxr_NDJF.time)[d] + dt.timedelta(days = lead)
            # if leap year and the day is after feburary 28th and before april (lead is okay for Nov-Dec)
            if calendar.isleap(temp.year) and temp.month in [2,3] and ((temp - pd.to_datetime(str(temp.year)+'-02-28')) > dt.timedelta(days = 0)):
                temp = pd.to_datetime(X1trainxr_NDJF.time)[d] + dt.timedelta(days = lead+1)
            Y1_leadtime.append(temp)
        Y1_leadtimexr = xr.DataArray(np.array(Y1_leadtime),dims='time',coords={'time':np.array(Y1_leadtime)})
        # ----------------
        Y1trainxr_NDJFM = Y1xr.where(Y1xr['time'] == Y1_leadtimexr, drop=True)
        
        # get training & standardize
        X1train_mean = X1trainxr_NDJF.mean('time')
        X1train_std  = X1trainxr_NDJF.std('time')
        
        # get training & standardize\
        Y1train_med = Y1trainxr_NDJFM.quantile(q=0.5,dim='time',keep_attrs=True)


    X1train = (X1trainxr_NDJF - X1train_mean)/X1train_std
    Y1train = Y1trainxr_NDJFM - Y1train_med

    # get validation & standardize
    pfiname = 'PRECT'+biasstr+'_mem'+str(valmem)+'_7daymean_1950-2014_20S-20N_regrid2.5x2.5_polydetrend_allmems.nc'
    X1xr = xr.open_dataarray(ppath+pfiname)[:-6][:-lead]
    X1valxr_NDJF = X1xr[X1xr.time.dt.month.isin(months)]
    del X1xr
    X1val = (X1valxr_NDJF - X1train_mean)/X1train_std

    # get testing & standardize
    pfiname = 'PRECT'+biasstr+'_mem'+str(testmem)+'_7daymean_1950-2014_20S-20N_regrid2.5x2.5_polydetrend_allmems.nc'
    X1xr = xr.open_dataarray(ppath+pfiname)[:-6][:-lead]
    X1testxr_NDJF = X1xr[X1xr.time.dt.month.isin(months)]
    del X1xr
    X1test = (X1testxr_NDJF - X1train_mean)/X1train_std


    # ----------------- Y split -------------------
    # get validation & standardize
    zfiname = 'Z500_mem'+str(valmem)+'_7daymean_1950-2014_30-60Nx170-240E_regrid2.5x2.5_polydetrend_allmems.nc'
    Y1xr = xr.open_dataarray(zpath+zfiname)
    Y1valxr_NDJFM = Y1xr.where(Y1xr['time'] == Y1_leadtimexr, drop=True)
    del Y1xr
    Y1val = Y1valxr_NDJFM - Y1train_med

    # get testing & standardize
    zfiname = 'Z500_mem'+str(testmem)+'_7daymean_1950-2014_30-60Nx170-240E_regrid2.5x2.5_polydetrend_allmems.nc'
    Y1xr = xr.open_dataarray(zpath+zfiname)
    Y1testxr_NDJFM = Y1xr.where(Y1xr['time'] == Y1_leadtimexr, drop=True)
    del Y1xr
    Y1test = Y1testxr_NDJFM - Y1train_med

    # ---------- BALANCE classes ----------
    Y1train[Y1train <= 0] = 0
    Y1train[Y1train > 0] = 1

    Y1val[Y1val <= 0] = 0
    Y1val[Y1val > 0] = 1

    Y1test[Y1test <= 0] = 0
    Y1test[Y1test > 0] = 1

    X1val, Y1val, _   = balance_classes(Xdata=X1val, Ydata=Y1val)
    X1test, Y1test, inew = balance_classes(Xdata=X1test, Ydata=Y1test)
    # ---------------------------------------------------

    # ---------- NUMPY ----------
    X1train = X1train.values
    X1val   = X1val.values
    X1test  = X1test.values

    Y1train = Y1train.values
    Y1val   = Y1val.values
    Y1test  = Y1test.values
    # ---------------------------------------------------
    
    return X1train, X1val, X1test, Y1train, Y1val, Y1test, inew


def split_retrain(trainmems, valmem, testmem, months, lead):
    zpath = '/glade/derecho/scratch/kjmayer/DATA/E3SMv2/Z500/'
    ppath = '/glade/derecho/scratch/kjmayer/DATA/E3SMv2/PRECT/'

    print('loading data & saving')
    # ----------- LOAD data & SHIFT time -------------
    # open detrended & running mean-ed data

    # ------ X LOAD & SHIFT --------
    if len(trainmems) > 1:
        for m,mem in enumerate(trainmems):
            pfiname = 'PRECT_mem'+str(mem)+'_7daymean_1950-2014_20S-20N_regrid2.5x2.5_polydetrend_allmems.nc'
            X1xr = xr.open_dataarray(ppath+pfiname)[:-6][:-lead]

            if m == 0:
                X1trainxr_NDJF = X1xr[X1xr.time.dt.month.isin(months)]
            else:
                X1trainxr_NDJF = xr.concat([X1trainxr_NDJF,X1xr[X1xr.time.dt.month.isin(months)]],dim='mem')

            del X1xr

            # ------ Y LOAD & SHIFT --------
            zfiname = 'Z500_mem'+str(mem)+'_7daymean_1950-2014_30-60Nx170-240E_regrid2.5x2.5_polydetrend_allmems.nc'
            Y1xr = xr.open_dataarray(zpath+zfiname)

            if m == 0:
                # ----- grab precip time + LEAD -----
                # ----------- (E3SM doesn't include Feb 29th for leap years)
                Y1_leadtime = []
                for d in np.arange(len(X1trainxr_NDJF.time)):
                    temp = pd.to_datetime(X1trainxr_NDJF.time)[d] + dt.timedelta(days = lead)
                    # if leap year and the day is after feburary 28th and before april (lead is okay for Nov-Dec)
                    if calendar.isleap(temp.year) and temp.month in [2,3] and ((temp - pd.to_datetime(str(temp.year)+'-02-28')) > dt.timedelta(days = 0)):
                        temp = pd.to_datetime(X1trainxr_NDJF.time)[d] + dt.timedelta(days = lead+1)
                    Y1_leadtime.append(temp)
                Y1_leadtimexr = xr.DataArray(np.array(Y1_leadtime),dims='time',coords={'time':np.array(Y1_leadtime)})
                # ----------------
                Y1trainxr_NDJFM = Y1xr.where(Y1xr['time'] == Y1_leadtimexr, drop=True)
            else:
                Y1trainxr_NDJFM = xr.concat([Y1trainxr_NDJFM, Y1xr.where(Y1xr['time'] == Y1_leadtimexr, drop=True)],dim='mem')

            del Y1xr
        # ---------------------------------------------------


        # ----------- SPLIT data --------------------------------- 
        # ----------------- X split -------------------
        # get training & standardize
        X1trainxr_NDJF = X1trainxr_NDJF.stack(s=('mem','time')).transpose('s','lat','lon')
        X1trainxr_NDJF = X1trainxr_NDJF.reset_index(['s'])

        X1train_mean = X1trainxr_NDJF.mean('s')#['s','lat','lon'])
        X1train_std  = X1trainxr_NDJF.std('s')#['s','lat','lon'])
        
        # get training & standardize
        Y1trainxr_NDJFM = Y1trainxr_NDJFM.stack(s=('mem','time'))
        Y1train_med = Y1trainxr_NDJFM.quantile(q=0.5,dim='s',keep_attrs=True)

        
    elif len(trainmems) == 1:
        print('made it!')
        pfiname = 'PRECT_mem'+str(trainmems[0])+'_7daymean_1950-2014_20S-20N_regrid2.5x2.5_polydetrend_allmems.nc'
        X1xr = xr.open_dataarray(ppath+pfiname)[:-6][:-lead]
        X1trainxr_NDJF = X1xr[X1xr.time.dt.month.isin(months)]            

        # ------ Y LOAD & SHIFT --------
        zfiname = 'Z500_mem'+str(trainmems[0])+'_7daymean_1950-2014_30-60Nx170-240E_regrid2.5x2.5_polydetrend_allmems.nc'
        Y1xr = xr.open_dataarray(zpath+zfiname)

        # ----- grab precip time + LEAD -----
        # ----------- (E3SM doesn't include Feb 29th for leap years)
        Y1_leadtime = []
        for d in np.arange(len(X1trainxr_NDJF.time)):
            temp = pd.to_datetime(X1trainxr_NDJF.time)[d] + dt.timedelta(days = lead)
            # if leap year and the day is after feburary 28th and before april (lead is okay for Nov-Dec)
            if calendar.isleap(temp.year) and temp.month in [2,3] and ((temp - pd.to_datetime(str(temp.year)+'-02-28')) > dt.timedelta(days = 0)):
                temp = pd.to_datetime(X1trainxr_NDJF.time)[d] + dt.timedelta(days = lead+1)
            Y1_leadtime.append(temp)
        Y1_leadtimexr = xr.DataArray(np.array(Y1_leadtime),dims='time',coords={'time':np.array(Y1_leadtime)})
        # ----------------
        Y1trainxr_NDJFM = Y1xr.where(Y1xr['time'] == Y1_leadtimexr, drop=True)
        
        # get training & standardize
        X1train_mean = X1trainxr_NDJF.mean('time')
        X1train_std  = X1trainxr_NDJF.std('time')
        
        # get training & standardize\
        Y1train_med = Y1trainxr_NDJFM.quantile(q=0.5,dim='time',keep_attrs=True)
    
    
    X1train = (X1trainxr_NDJF - X1train_mean)/X1train_std
    Y1train = Y1trainxr_NDJFM - Y1train_med
    
    # ---------------------------------------------------
    # get validation & standardize
    pfiname = 'PRECT_mem'+str(valmem)+'_7daymean_1950-2014_20S-20N_regrid2.5x2.5_polydetrend_allmems.nc'
    X1xr = xr.open_dataarray(ppath+pfiname)[:-6][:-lead]
    X1valxr_NDJF = X1xr[X1xr.time.dt.month.isin(months)]
    del X1xr
    X1val = (X1valxr_NDJF - X1train_mean)/X1train_std

    # get testing & standardize
    pfiname = 'PRECT_mem'+str(testmem)+'_7daymean_1950-2014_20S-20N_regrid2.5x2.5_polydetrend_allmems.nc'
    X1xr = xr.open_dataarray(ppath+pfiname)[:-6][:-lead]
    X1testxr_NDJF = X1xr[X1xr.time.dt.month.isin(months)]
    del X1xr
    X1test = (X1testxr_NDJF - X1train_mean)/X1train_std


    # ----------------- Y split -------------------
    # get validation & standardize
    zfiname = 'Z500_mem'+str(valmem)+'_7daymean_1950-2014_30-60Nx170-240E_regrid2.5x2.5_polydetrend_allmems.nc'
    Y1xr = xr.open_dataarray(zpath+zfiname)
    Y1valxr_NDJFM = Y1xr.where(Y1xr['time'] == Y1_leadtimexr, drop=True)
    del Y1xr
    Y1val = Y1valxr_NDJFM - Y1train_med

    # get testing & standardize
    zfiname = 'Z500_mem'+str(testmem)+'_7daymean_1950-2014_30-60Nx170-240E_regrid2.5x2.5_polydetrend_allmems.nc'
    Y1xr = xr.open_dataarray(zpath+zfiname)
    Y1testxr_NDJFM = Y1xr.where(Y1xr['time'] == Y1_leadtimexr, drop=True)
    del Y1xr
    Y1test = Y1testxr_NDJFM - Y1train_med
    
    # ---------- BALANCE classes ----------
    Y1train[Y1train <= 0] = 0
    Y1train[Y1train > 0] = 1

    Y1val[Y1val <= 0] = 0
    Y1val[Y1val > 0] = 1

    Y1test[Y1test <= 0] = 0
    Y1test[Y1test > 0] = 1

    X1val, Y1val, _   = balance_classes(Xdata=X1val, Ydata=Y1val)
    X1test, Y1test, inew = balance_classes(Xdata=X1test, Ydata=Y1test)
    # ---------------------------------------------------

    # ---------- NUMPY ----------
    X1train = X1train.values
    X1val   = X1val.values
    X1test  = X1test.values

    Y1train = Y1train.values
    Y1val   = Y1val.values
    Y1test  = Y1test.values
    # ---------------------------------------------------
    
    return X1train, X1val, X1test, Y1train, Y1val, Y1test, inew
