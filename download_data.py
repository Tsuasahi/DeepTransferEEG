import tl.utils.numpy_fix as np
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import moabb

from moabb.datasets import (BNCI2014001, BNCI2014002, BNCI2015001, Nakanishi2015, Wang2016, MAMEM1, MAMEM2, MAMEM3, Lee2019_SSVEP)
from moabb.paradigms import MotorImagery, P300, SSVEP


def download_subject(paradigm, dataset, subject):
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    return (X, labels, meta)


def dataset_to_file(dataset_name, data_save=True):
    moabb.set_log_level("ERROR")
    if dataset_name == 'BNCI2014001':
        dataset = BNCI2014001()
        paradigm = MotorImagery(n_classes=4)
        # (5184, 22, 1001) (5184,) 250Hz 9subjects * 4classes * (72+72)trials for 2sessions
    elif dataset_name == 'BNCI2014002':
        dataset = BNCI2014002()
        paradigm = MotorImagery(n_classes=2)
        # (2240, 15, 2561) (2240,) 512Hz 14subjects * 2classes * (50+30)trials * 2sessions(not namely separately)
    elif dataset_name == 'BNCI2015001':
        dataset = BNCI2015001()
        paradigm = MotorImagery(n_classes=2)
        # (5600, 13, 2561) (5600,) 512Hz 12subjects * 2 classes * (200 + 200 + (200 for Subj 8/9/10/11)) trials * (2/3)sessions
    elif dataset_name == 'Nakanishi2015':
        dataset = Nakanishi2015()
        paradigm = SSVEP(n_classes=12)

    # maybe Connection refused
    elif dataset_name == 'Wang2016':
        dataset = Wang2016()
        paradigm = SSVEP(n_classes=40)

    # 403 error at moabb 0.4.6
    elif dataset_name == 'MAMEM1':
        dataset = MAMEM1()
        paradigm = SSVEP(n_classes=5)
    elif dataset_name == 'MAMEM2':
        dataset = MAMEM2()
        paradigm = SSVEP(n_classes=5)
    elif dataset_name == 'MAMEM3':
        dataset = MAMEM3()
        paradigm = SSVEP(n_classes=4)

    elif dataset_name == 'Lee2019_SSVEP':
        dataset = Lee2019_SSVEP()
        paradigm = SSVEP(n_classes=4)

    if data_save:
        print(f'preparing ' + (dataset_name) + ' data...')
        results = []
        subjects = dataset.subject_list[:]
        
        # multi threads download
        with ThreadPoolExecutor(max_workers=1) as executor:
            tasks = [executor.submit(download_subject, paradigm, dataset, sub) for sub in subjects]
            
            for task in tasks:
                res = task.result()
                if res is not None:
                    results.append(res)

        xs_list, labels_list, meta_list = zip(*results)
        
        X = np.concatenate(xs_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        meta = pd.concat(meta_list, ignore_index=True)
        
        if not os.path.exists('./data/'):
            os.makedirs('./data/')
        if not os.path.exists('./data/' + dataset_name + '/'):
            os.makedirs('./data/' + dataset_name + '/')
        np.save('./data/' + dataset_name + '/X', X)
        np.save('./data/' + dataset_name + '/labels', labels)
        meta.to_csv('./data/' + dataset_name + '/meta.csv')
        print(f'{dataset_name} done!')

    else:
        if isinstance(paradigm, MotorImagery):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[dataset.subject_list[0]], return_epochs=True)
            return X.info
        elif isinstance(paradigm, P300):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[dataset.subject_list[0]], return_epochs=True)
            return X.info
        elif isinstance(paradigm, SSVEP):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[dataset.subject_list[0]], return_epochs=True)
            return X.info

if __name__ == '__main__':

    datasets = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001', 'Nakanishi2015','Lee2019_SSVEP']
    
    for dataset_name in datasets:
        dataset_to_file(dataset_name, data_save=True)
    

    '''
    BNCI2014001
    <Info | 8 non-empty values
     bads: []
     ch_names: 'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
     chs: 22 EEG
     custom_ref_applied: False
     dig: 25 items (3 Cardinal, 22 EEG)
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 22
     projs: []
     sfreq: 250.0 Hz
    >

    BNCI2014002
    <Info | 7 non-empty values
     bads: []
     ch_names: 'EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8', 'EEG9', 'EEG10', 'EEG11', 'EEG12', 'EEG13', 'EEG14', 'EEG15'
     chs: 15 EEG
     custom_ref_applied: False
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 15
     projs: []
     sfreq: 512.0 Hz
    >

    BNCI2015001
    <Info | 8 non-empty values
     bads: []
     ch_names: 'FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CPz', 'CP4'
     chs: 13 EEG
     custom_ref_applied: False
     dig: 16 items (3 Cardinal, 13 EEG)
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 13
     projs: []
     sfreq: 512.0 Hz
    >
    '''
