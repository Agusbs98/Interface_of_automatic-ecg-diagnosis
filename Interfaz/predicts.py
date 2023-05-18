from libs import *
import configVars
from tools import tools
from data import ECGDataset

def procesar_archivo(format,number,unit,frec,file):
    try:
        prepare_data(format,number,unit,frec,file)
        antonior92 = predict_antonior92()
        CPSC = predict_CPSC_2018()
        Chapman = predict_Chapman()
        result = pd.DataFrame(data = [['Antonior92',antonior92],['CPSC-2018',CPSC],['Chapman',Chapman]],columns=['Red','Predicción'])
        tools.ecgPlot("./datasets/pred.npy",500)
        return result, "ecg.png"
    except:
        return pd.DataFrame(data = ["Se ha producido un error inesperado.","Compruebe que los datos de entrada sean correctos"],columns = ["ERROR."]), "error.jpg"


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
    return pred if len(pred) != 0 else ['El archivo introducido no satisface ninguno de los criterios de clasificación']

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
    pred = model.predict(f['tracings'],  verbose=0)
    optimal_thresholds = pd.read_csv(f"{configVars.pathThresholds}antonior92/optimal_thresholds_best.csv")
    result = optimal_thresholds[optimal_thresholds["Threshold"]<=pred[0]]
    result = result['Pred'].values.tolist()
    f.close()
    
    return result if len(result) != 0 else ['Normal']

def prepare_data(format,number,unit,frec,file):
    units = {
        'V':0.001,
        'miliV':1,
        'microV':1000,
        'nanoV':1000000
    }
    if(format == 'XML'):
        f = read_file(file.name)
        df = pd.DataFrame()
        for lead in f.leads:
            df[lead.label]=lead.samples
        data = df
    elif(format == 'CSV'):
        data = pd.read_csv(file.name,header = None)
        
    data = data[:-200]
    data = data.T
    leads = len(data)
    frec = frec if frec>0 else 1
    scale = 1/(number*units[unit])
    ecg_preprocessed = tools.preprocess_ecg(data, frec, leads,
                                            scale=scale,######### modificar para que segun la unidad introducida se pueda convertir los datos
                                            use_all_leads=True,
                                            remove_baseline=True)
    tools.generateH5(ecg_preprocessed,
             "pred.hdf5",new_freq=400,new_len=4096,
             scale=2,sample_rate = frec)

    np.save(f"{configVars.pathCasos}pred.npy",ecg_preprocessed )