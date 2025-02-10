def get_hp(experiment_name):
    
    experiments = {
        # ---------------------------------------------------
        ## ------Experiment 1:------
        # ridge & no dropout & ridge then dropout
        'exp_data': {
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
        },

        'exp_data_retrain': {
            'train_yrs': [1996,2015],
            'val_yrs': [2015,2020],
            'test_yrs': [2020,2023],
            'region': [[30,60],[170,240]], # [lat,lat,lon,lon] 
            'input': 'obs_precip',
            'output': 'obs_z500',
            'LEAD': 14,
            'RUNMEAN': 7,
        },
            

        'exp_2.BO_retrain': {
            'HIDDENS': [8, 16, 32, 128],
            'BATCH_SIZES': [16, 64],
            'LR_INITS':  [1e-2, 1e-3, 1e-4],
            'RIDGES': [0.0, 0.5, 1.0],
            'DROPOUT_RATES': [0.0, 0.5, 0.9], 
            'PATIENCE': 10,
            'GLOBAL_SEED': 99,
            'BO_ALPHA': 0.0001,
            'BO_BETA': 4,
        },

        'exp_2.1.BO_retrain': {
            'HIDDENS': [8, 16, 32, 128],
            'BATCH_SIZES': [16, 64],
            'LR_INITS':  [1e-2, 1e-3, 1e-4],
            'RIDGES': [0.0, 0.5, 1.0],
            'DROPOUT_RATES': [0.0, 0.5, 0.9], 
            'PATIENCE': 10,
            'GLOBAL_SEED': 99,
            'BO_ALPHA': 0.001,
            'BO_BETA': 10,
        },
    }

    return experiments[experiment_name]
