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


def detrend_obs(data, train_data, npoly=3):
    '''
    detrend reanalysis using polynomial fit (for each doy) to the training mean
    
    data: [time, lat, lon] or [member, time]
        reanalysis to detrend 
    
    train_data: [time, lat, lon] or [time]
        ensemble mean 
    
    npoly: [int] 
        order of polynomial, default = 3rd order
    '''
    
    # stack lat and lon of ensemble mean data
    if len(train_data.shape) == 3:
        train_data = train_data.stack(z=('latitude', 'longitude'))
 
    # stack lat and lon of member data & grab doy information
    if len(data.shape) == 3:
        data = data.stack(z=('latitude', 'longitude'))
    temp = data['time.dayofyear']
    
    # grab every Xdoy from ensmean, fit npoly polynomial
    # subtract polynomial from every Xdoy from members
    detrend = []
    for label,ens_group in train_data.groupby('time.dayofyear'):
        Xgroup = data.where(temp == label, drop = True)
        
        curve = polynomial.polyfit(np.arange(0, ens_group.shape[0]), ens_group, npoly)
        trend = polynomial.polyval(np.arange(0, ens_group.shape[0]), curve, tensor=True)
        if len(train_data.shape) == 2: #combined lat and lon, so now 2
            trend = np.swapaxes(trend,0,1) #only need to swap if theres a space dimension

        diff = Xgroup - trend
        detrend.append(diff)
        
        # if label == 1:
        #if len(Xgroup.shape) == 2:
        #    plt.plot(Xgroup[:,100],'teal')
        #    plt.plot(trend[:,100],'k')
        #    plt.show()
        #else:
        #    plt.plot(Xgroup,'teal')
        #    plt.plot(trend,'k')
        #    plt.show()

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
        
        return Xdata,Ydata,inew
        
    else:
        inew = []
        return Xdata,Ydata,inew



def split_e3sm(trainmems, valmem, testmem, months, lead, extrainfo=False):
    z500_path = '/glade/derecho/scratch/kjmayer/DATA/E3SMv2/Z500/'
    precip_path = '/glade/derecho/scratch/kjmayer/DATA/E3SMv2/PRECT/'
    
    # ----------- LOAD data & SHIFT time -------------
    # open detrended & running mean-ed data

    # ------ X LOAD & SHIFT --------

    for m,mem in enumerate(trainmems):
        precip_finame = 'PRECT_mem'+str(mem)+'_7daymean_1950-2014_20S-20N_regrid2.5x2.5_polydetrend_allmems.nc'
        X1xr = xr.open_dataarray(precip_path+precip_finame)[:-6][:-lead]

        if m == 0:
            X1trainxr_NDJF = X1xr[X1xr.time.dt.month.isin(months)]
        else:
            X1trainxr_NDJF = xr.concat([X1trainxr_NDJF,X1xr[X1xr.time.dt.month.isin(months)]],dim='mem')

        del X1xr

        # ------ Y LOAD & SHIFT --------
        z500_finame = 'Z500_mem'+str(mem)+'_7daymean_1950-2014_30-60Nx170-240E_regrid2.5x2.5_polydetrend_allmems.nc'
        Y1xr = xr.open_dataarray(z500_path+z500_finame)

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
    # ---------------------------------------------------


    # ----------- SPLIT data --------------------------------- 
    # ----------------- X split -------------------
    # get training & standardize
    X1trainxr_NDJF = X1trainxr_NDJF.stack(s=('mem','time')).transpose('s','lat','lon')
    X1trainxr_NDJF = X1trainxr_NDJF.reset_index(['s'])

    X1train_mean = X1trainxr_NDJF.mean('s')
    X1train_std  = X1trainxr_NDJF.std('s')

    X1train = (X1trainxr_NDJF - X1train_mean)/X1train_std

    # get validation & standardize
    precip_finame = 'PRECT_mem'+str(valmem)+'_7daymean_1950-2014_20S-20N_regrid2.5x2.5_polydetrend_allmems.nc'
    X1xr = xr.open_dataarray(precip_path+precip_finame)[:-6][:-lead]
    X1valxr_NDJF = X1xr[X1xr.time.dt.month.isin(months)]
    del X1xr
    X1val = (X1valxr_NDJF - X1train_mean)/X1train_std

    # get testing & standardize
    precip_finame = 'PRECT_mem'+str(testmem)+'_7daymean_1950-2014_20S-20N_regrid2.5x2.5_polydetrend_allmems.nc'
    X1xr = xr.open_dataarray(precip_path+precip_finame)[:-6][:-lead]
    X1testxr_NDJF = X1xr[X1xr.time.dt.month.isin(months)]
    del X1xr
    X1test = (X1testxr_NDJF - X1train_mean)/X1train_std


    # ----------------- Y split -------------------
    # get training & standardize
    Y1trainxr_NDJFM = Y1trainxr_NDJFM.stack(s=('mem','time'))

    Y1train_med = Y1trainxr_NDJFM.quantile(q=0.5,dim='s',keep_attrs=True)
    Y1train = Y1trainxr_NDJFM - Y1train_med

    # get validation & standardize
    z500_finame = 'Z500_mem'+str(valmem)+'_7daymean_1950-2014_30-60Nx170-240E_regrid2.5x2.5_polydetrend_allmems.nc'
    Y1xr = xr.open_dataarray(z500_path+z500_finame)
    Y1valxr_NDJFM = Y1xr.where(Y1xr['time'] == Y1_leadtimexr, drop=True)
    del Y1xr
    Y1val = Y1valxr_NDJFM - Y1train_med

    # get testing & standardize
    z500_finame = 'Z500_mem'+str(testmem)+'_7daymean_1950-2014_30-60Nx170-240E_regrid2.5x2.5_polydetrend_allmems.nc'
    Y1xr = xr.open_dataarray(z500_path+z500_finame)
    Y1testxr_NDJFM = Y1xr.where(Y1xr['time'] == Y1_leadtimexr, drop=True)
    del Y1xr
    Y1test = Y1testxr_NDJFM - Y1train_med
    Y1test_time = Y1test.time
    Y1test_vals = Y1test.copy(deep=True)

    # ---------- BALANCE classes ----------
    Y1train[Y1train <= 0] = 0
    Y1train[Y1train > 0] = 1

    Y1val[Y1val <= 0] = 0
    Y1val[Y1val > 0] = 1

    Y1test[Y1test <= 0] = 0
    Y1test[Y1test > 0] = 1

    X1val, Y1val,_   = balance_classes(Xdata=X1val, Ydata=Y1val)
    X1test, Y1test, inew = balance_classes(Xdata=X1test, Ydata=Y1test)
    # ---------------------------------------------------
    

    # ---------- SAVE ----------
    X1train = X1train.values
    X1val   = X1val.values
    X1test  = X1test.values

    Y1train = Y1train.values
    Y1val   = Y1val.values
    Y1test  = Y1test.values
    
    # ---------------------------------------------------
    if extrainfo:
        return X1train, X1val, X1test, Y1train, Y1val, Y1test, inew, Y1test_vals, Y1test_time
    elif not extrainfo:
        return X1train, X1val, X1test, Y1train, Y1val, Y1test, inew

    

    

    

def split_obs(trainyrs, valyrs, testyrs, months, lead, latpt, lonpt):
        
    precip_obs_path = '/glade/derecho/scratch/kjmayer/DATA/GPCP/PRECT/daily/NN_data/'
    z500_obs_path = '/glade/derecho/scratch/kjmayer/DATA/ERA5/z500/daily/NN_data/'
    
    # ----------- LOAD data & SHIFT time -------------
    # open detrended & running mean-ed data

    # ------ X LOAD --------
    precip_obs_finame = 'precip_gpcp_7dayrunmean_1996-2023_20S-20N_regrid2.5x2.5_finetunetrain_polydetrendyrs96-23.nc'
    X2xr = xr.open_dataarray(precip_obs_path+precip_obs_finame)

    # ------ Y LOAD --------
    z500_obs_finame = 'z500_daily_era5_7daymean_1996-2023_20-90N_regrid2.5x2.5_finetunetrain_polydetrendyrs96-23.nc'
    Y2xr = xr.open_dataarray(z500_obs_path+z500_obs_finame)
    #Y2xr = Y2xr.where((Y2xr['latitude'] == latpt) & (Y2xr['longitude'] == lonpt),drop = True).squeeze()
    Y2xr = Y2xr.where((Y2xr['latitude'] >= latpt[0]) & (Y2xr['latitude'] <= latpt[1]) & (Y2xr['longitude'] >= lonpt[0]) & (Y2xr['longitude'] <= lonpt[1]), drop = True).mean(['latitude','longitude'])
    # ---------------------------------------------------


    # ----------- SPLIT data --------------------------------- 
    # split training & (calculate) standardization data

    # X2train
    X2trainxr = X2xr.sel(time=slice(str(trainyrs[0])+'-11-01',
                                    str(trainyrs[1])+'-2-28'))[:-lead]

    X2trainxr_NDJF = X2trainxr[X2trainxr.time.dt.month.isin(months)]

    # Y2train
    Y2trainxr = Y2xr.sel(time=slice(str(trainyrs[0])+'-11-01',
                                    str(trainyrs[1])+'-2-28'))[lead:]

    Y2train_leadtime = pd.to_datetime(X2trainxr_NDJF.time) + dt.timedelta(days = lead)
    Y2train_leadtimexr = xr.DataArray(np.array(Y2train_leadtime),dims='time',coords={'time':np.array(Y2train_leadtime)})
    Y2trainxr_NDJFM = Y2trainxr.where(Y2trainxr['time'] == Y2train_leadtimexr,drop = True).squeeze()

    
    X2train_mean = X2trainxr_NDJF.mean('time')
    X2train_std  = X2trainxr_NDJF.std('time')
    Y2train_med = Y2trainxr_NDJFM.quantile(q=0.5,dim='time',keep_attrs=True)


    # ----------------- X split -------------------    
    X2train = (X2trainxr_NDJF - X2train_mean)/X2train_std

    # get validation & standardize
    X2valxr = X2xr.sel(time=slice(str(valyrs[0])+'-11-01',
                                  str(valyrs[1])+'-2-28'))[:-lead]

    X2valxr_NDJF = X2valxr[X2valxr.time.dt.month.isin(months)]
    X2val = (X2valxr_NDJF - X2train_mean)/X2train_std

    # get testing & standardize
    X2testxr = X2xr.sel(time=slice(str(testyrs[0])+'-11-01',
                                   str(testyrs[1])+'-2-28'))[:-lead]

    X2testxr_NDJF = X2testxr[X2testxr.time.dt.month.isin(months)]
    X2test = (X2testxr_NDJF - X2train_mean)/X2train_std


    # ----------------- Y split -------------------
    Y2train = Y2trainxr_NDJFM - Y2train_med

    # get validation & standardize
    Y2valxr = Y2xr.sel(time=slice(str(valyrs[0])+'-11-01',
                                  str(valyrs[1])+'-2-28'))[lead:]

    Y2val_leadtime = pd.to_datetime(X2valxr_NDJF.time) + dt.timedelta(days = lead)
    Y2val_leadtimexr = xr.DataArray(np.array(Y2val_leadtime),dims='time',coords={'time':np.array(Y2val_leadtime)})
    Y2valxr_NDJFM = Y2valxr.where(Y2valxr['time'] == Y2val_leadtimexr,drop = True).squeeze()

    Y2val = Y2valxr_NDJFM - Y2train_med

    # get testing & standardize
    Y2testxr = Y2xr.sel(time=slice(str(testyrs[0])+'-11-01',
                                   str(testyrs[1])+'-2-28'))[lead:]

    Y2test_leadtime = pd.to_datetime(X2testxr_NDJF.time) + dt.timedelta(days = lead)
    Y2test_leadtimexr = xr.DataArray(np.array(Y2test_leadtime),dims='time',coords={'time':np.array(Y2test_leadtime)})
    Y2testxr_NDJFM = Y2testxr.where(Y2testxr['time'] == Y2test_leadtimexr,drop = True).squeeze()

    Y2test = Y2testxr_NDJFM - Y2train_med
    # --------------------------------------------------------------


    # ---------- BALANCE classes ----------
    Y2train[Y2train <= 0] = 0
    Y2train[Y2train > 0] = 1

    Y2val[Y2val <= 0] = 0
    Y2val[Y2val > 0] = 1

    Y2test[Y2test <= 0] = 0
    Y2test[Y2test > 0] = 1

    X2val, Y2val, _   = balance_classes(Xdata=X2val, Ydata=Y2val)
    X2test, Y2test, _ = balance_classes(Xdata=X2test, Ydata=Y2test)
    # ---------------------------------------------------


    # ---------- SAVE ----------
    X2train = X2train.values
    X2val   = X2val.values
    X2test  = X2test.values

    Y2train = Y2train.values
    Y2val   = Y2val.values
    Y2test  = Y2test.values
    
    # ---------------------------------------------------
    
    return X2train, X2val, X2test, Y2train, Y2val, Y2test
