def get_hp(experiment_name):
    
    experiments = {
        # ---------------------------------------------------
        ## ------Experiment 2:------
        # Dropout & no ridge
        'exp2': {
            'train_mems': ['0101',
                           '0111',
                           '0121',
                           '0131',
                           '0141',
                           '0151',
                           '0161',
                           '0171',
                           '0181'],
            'val_mems': ['0191'],
            'test_mems': ['0301'],
            'region': ['30-60','170-240'], #N,E 
            'input': 'PRECT',
            'output': 'Z500',
            'LEAD': 14,
            'RUNMEAN': 7,
            'HIDDENS': [128], 
            'BATCH_SIZE': 256,
            'LR_INIT': 0.0001,
            'RIDGE': 0.0,
            'DROPOUT_RATE': 0.9,
            'PATIENCE': 10,
            'GLOBAL_SEED': 99,
        },

        
        'exp2_retrain': {
            'train_yrs': [1996,2015],
            'val_yrs': [2015,2020],
            'test_yrs': [2020,2023],
            'region': [[30,60],[170,240]], # [lat,lat,lon,lon] 
            'input': 'obs_precip',
            'output': 'obs_z500',
            'LEAD': 14,
            'RUNMEAN': 7,
            'HIDDENS': [128,8], 
            'BATCH_SIZE': 64,
            'LR_INIT': 0.01,
            'RIDGE': 0.1,
            'DROPOUT_RATE': 0.9,
            'PATIENCE': 30, 
            'GLOBAL_SEED': 99,
        },

        'exp2_retrain.1': {
            'train_yrs': [1999,2018], # 19 years
            'val_yrs': [2018,2023], # 5 years
            'test_yrs': [1996,1999], # 3 years
            'region': [[30,60],[170,240]], # [lat,lat,lon,lon] 
            'input': 'obs_precip',
            'output': 'obs_z500',
            'LEAD': 14,
            'RUNMEAN': 7,
            'HIDDENS': [128], 
            'BATCH_SIZE': 64,
            'LR_INIT': 0.001,
            'RIDGE': 3.0,
            'DROPOUT_RATE': 0.95,
            'PATIENCE': 10, 
            'GLOBAL_SEED': 99,
        },

        'exp2_allobs': {
            'train_yrs': [1996,2023],
            'val_yrs': [2020,2023], # filler so code doesnt error
            'test_yrs': [1996,2023],
            'region': [[30,60],[170,240]], # [lat,lat,lon,lon] 
            'input': 'obs_precip',
            'output': 'obs_z500',
            'LEAD': 14,
            'RUNMEAN': 7,
            'HIDDENS': [128,8], 
            'BATCH_SIZE': 64,
            'LR_INIT': 0.01,
            'RIDGE': 0.1,
            'DROPOUT_RATE': 0.9,
            'PATIENCE': 30, # doesn't do anything here.
            'GLOBAL_SEED': 99,
        },

    }

    return experiments[experiment_name]
