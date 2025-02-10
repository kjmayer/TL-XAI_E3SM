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
            'train_mems': ['0201',
                           '0211',
                           '0221',
                           '0231',
                           '0241',
                           '0251',
                           '0261',
                           '0271',
                           '0281'],
            'val_mems': ['0291'],
            'test_mems': ['0301'],
            'region': ['30-60','170-240'], #N,E
            'input': 'PRECT',
            'output': 'Z500',
            'LEAD': 14,
            'RUNMEAN': 7,
        },
            

        'exp_1': {
            'HIDDENS': [8, 16, 32, 128, 256],
            'BATCH_SIZES': [32, 64, 128, 256, 512],
            'LR_INITS':  [1e-2, 1e-3, 1e-4],
            'RIDGES': [0.1,0.5,1.0],
            'DROPOUT_RATE': 0.0, 
            'PATIENCE': 10,
            'GLOBAL_SEED': 99,
        },


        'exp_2': {
            'HIDDENS': [8, 16, 32, 128, 256],
            'BATCH_SIZES': [32, 64, 128, 256, 512],
            'LR_INITS':  [1e-2, 1e-3, 1e-4],
            'RIDGES': 0,
            'DROPOUT_RATES': [0.85,0.9,0.95], 
            'PATIENCE': 10,
            'GLOBAL_SEED': 99,
        },

        'exp_2.BO': {
            'HIDDENS': [8, 16, 32, 128, 256],
            'BATCH_SIZES': [64, 256, 1028],
            'LR_INITS':  [1e-2, 1e-3, 1e-4],
            'RIDGES': 0,
            'DROPOUT_RATES': [0.2,0.5,0.9], 
            'PATIENCE': 10,
            'GLOBAL_SEED': 99,
        },

        
        'exp_2_retrain': {
            'HIDDENS': [8, 16, 32, 128, 256],  
            'BATCH_SIZES': [32, 64, 128, 256, 512],
            'LR_INITS':[1e-3, 1e-4, 1e-5],
            'RIDGES': 0,
            'DROPOUT_RATES': [0.85,0.9,0.95],
            'PATIENCE': 30,
            'GLOBAL_SEED': 99,
        },

        'exp_2.BO_retrain': {
            'HIDDENS': [8, 16, 32, 128, 256],  
            'BATCH_SIZES': [64, 256, 1028],
            'LR_INITS':[1e-3, 1e-4, 1e-5],
            'RIDGES': 0,
            'DROPOUT_RATES': [0.2,0.5,0.95],
            'PATIENCE': 30,
            'GLOBAL_SEED': 99,
        },

    }

    return experiments[experiment_name]
