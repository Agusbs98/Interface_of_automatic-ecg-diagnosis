from libs import *
import configVars
from tools import tools
from data import ECGDataset

def procesar_archivo(format,number,unit,frec,file):
    prepare_data(format,number,unit,frec,file)
    antonior92 = predict_antonior92()
    #antonior92 = []
    CPSC = predict_CPSC_2018()
    Chapman = predict_Chapman()
    result = pd.DataFrame(data = [['Antonior92',antonior92],['CPSC-2018',CPSC],['Chapman',Chapman]],columns=['Red','Predicción'])
    ##return []
    return result


def predict_CPSC_2018():
    config = {
    "ecg_leads":[
        0, 1, 
        6, 
    ], 
    "ecg_length":5000, 
    "is_multilabel":True, 
    }

    train_loaders = {
        "pred":torch.utils.data.DataLoader(
            ECGDataset(
                df_path = f"{configVars.pathCasos}pred.csv", data_path = f"{configVars.pathCasos}",
                config = config, 
                augment = False, 
            ), 
            timeout=0
        )
    }
    save_ckp_dir = f"{configVars.pathModel}CPSC-2018"

    pred = tools.LightX3ECG(
                train_loaders, 
                config,
                save_ckp_dir, 
                )
    return pred

def predict_Chapman():
    config = {
    "ecg_leads":[
        0, 1, 
        6, 
    ], 
    "ecg_length":5000, 
    "is_multilabel":False, 
    }

    train_loaders = {
        "pred":torch.utils.data.DataLoader(
            ECGDataset(
                df_path = f"{configVars.pathCasos}pred.csv", data_path = f"{configVars.pathCasos}",
                config = config, 
                augment = False, 
            ), 
            timeout=0
        )
    }
    save_ckp_dir = f"{configVars.pathModel}Chapman"

    pred = tools.LightX3ECG(
                train_loaders, 
                config,
                save_ckp_dir, 
                )
    return pred

def predict_antonior92():
    threshold = np.array([0.124, 0.07, 0.05, 0.278, 0.390, 0.174])
    f = h5py.File(f"{configVars.pathCasos}pred.hdf5", 'r')
    model = load_model(f"{configVars.pathModel}/antonior92/model.hdf5", compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    pred = model.predict(f['tracings'],  verbose=1)
    optimal_thresholds = pd.read_csv(f"{configVars.pathThresholds}antonior92/optimal_thresholds_best.csv")
    result = optimal_thresholds[optimal_thresholds["Threshold"]<=pred[0]]
    result = result['Pred'].values.tolist()
    print(result)
    enfermedades = np.array(['Normal','1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST'])
    selected = [True,False,True,False,False,True,False]
    #result = enfermedades[selected]
    f.close()
    
    return result

def prepare_data(format,number,unit,frec,file):
    names=['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    units = {
        'V':1000,
        'miliV':1,
        'microV':0.001,
        'nanoV':0.000001
    }
    if(format == 'XML'):
        f = read_file(file.name)
        df = pd.DataFrame(columns=names)
        for lead in f.leads:
            df[lead.label]=lead.samples
        data = df
    elif(format == 'CSV'):
        data = pd.read_csv(file.name,header = None)
    else:
        data = pd.read_csv(file,header = None,names = names)
        
    data = data[:-200]
    data = data.T
    leads = len(data)
    frec = frec if frec>0 else 1
    scale = 1/(number*units[unit])
    print(scale)
    ecg_preprocessed = tools.preprocess_ecg(data, frec, leads,
                                            scale=scale,######### modificar para que segun la unidad introducida se pueda convertir los datos
                                            use_all_leads=True,
                                            remove_baseline=True)
    tools.generateH5(ecg_preprocessed,
             "pred.hdf5",new_freq=400,new_len=4096,
             scale=20,sample_rate = frec)

    np.save(f"{configVars.pathCasos}pred.npy",ecg_preprocessed )