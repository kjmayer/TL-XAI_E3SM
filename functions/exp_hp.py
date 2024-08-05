def get_hp(experiment_name):
    
    experiments = {
        # ---------------------------------------------------
        ## Experiment 1:
        'exp1': {
            'train_mems': ['0101','0111','0121','0131','0141','0151','0161','0171','0181'],
            'val_mems': ['0191'],
            'test_mems': ['0301'],
            'region': ['30-60','170-240'], # [lat,lon] 
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
        
        'exp1_retrain': {
            'train_mems': ['0201','0211','0221','0231','0241','0251','0261','0271','0281'],
            'val_mems': ['0291'],
            'test_mems': ['0301'],
            'region': ['30-60','170-240'], # [lat,lon] 
            'input': 'PRECT',
            'output': 'Z500',
            'LEAD': 14,
            'RUNMEAN': 7,
            'HIDDENS': [128], 
            'BATCH_SIZE': 256,
            'LR_INIT': 0.0001,
            'RIDGE': 0,
            'DROPOUT_RATE': 0.9,
            'PATIENCE': 10,
            'GLOBAL_SEED': 99,
        },
    }

    return experiments[experiment_name]
