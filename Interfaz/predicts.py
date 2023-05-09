from libs import *
import configVars
from tools import tools
from data import ECGDataset

def procesar_archivo(form,sex,age,file):
    prepare_data(form,sex,age,file)
    antonior92 = predict_antonior92()
    CPSC = predict_CPSC_2018()
    Chapman = predict_Chapman()
    result = pd.DataFrame(data = [['Antonior92',antonior92],['CPSC-2018',CPSC],['Chapman',Chapman]],columns=['Red','Predicción'])
    #return []
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
                training_verbose = True, 
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
    print(train_loaders['pred'])
    print(save_ckp_dir)
    pred = tools.LightX3ECG(
                train_loaders, 
                config,
                save_ckp_dir, 
                training_verbose = True, 
                )
    return pred

def predict_antonior92():

    #f = h5py.File("ecg.hdf5", 'r')
    #model = load_model(f"{configVars.pathModel}/antonior92/model.hdf5", compile=False)
    #model.compile(loss='binary_crossentropy', optimizer=Adam())
    #result = model.predict(ecg,  verbose=1)
    
    enfermedades = np.array(['Normal','1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST'])
    selected = [True,False,True,False,False,True,False]
    result = enfermedades[selected]
    #f.close()
    
    return result.tolist()

def prepare_data(form,sex,age,file):
    names=['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    if(form == 'XML'):
        data = pd.read_csv(file,header = None,names = names)
    else:
        data = pd.read_csv(file.name,header = None)
    data = data.T
    tools.generateH5(data,
             "ecg.hdf5",new_freq=400,new_len=4096,
             scale=2,powerline=None,use_all_leads=True,
             remove_baseline=False,root_dir=None,fmt='wfdb')
    pred = pd.read_csv(f"{configVars.pathCasos}pred.csv",)
    pred['age'][0] = age
    pred['sex'][0] = (0 if sex == 'Hombre'else 1)
    pred['r_count'][0] = 19 ##### tools.get_r_count(data.T)
    pred['length'][0] = len(data[0]) 
    pred.to_csv(f"{configVars.pathCasos}pred.csv",index = False)
    np.save(f"{configVars.pathCasos}pred.npy",data.values)