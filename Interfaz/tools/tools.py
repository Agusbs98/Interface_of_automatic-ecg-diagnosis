from libs import *
import configVars

def remove_baseline_filter(sample_rate):
    fc = 0.8  # [Hz], cutoff frequency
    fst = 0.2  # [Hz], rejection band
    rp = 0.5  # [dB], ripple in passband
    rs = 40  # [dB], attenuation in rejection band
    wn = fc / (sample_rate / 2)
    wst = fst / (sample_rate / 2)

    filterorder, aux = sgn.ellipord(wn, wst, rp, rs)
    sos = sgn.iirfilter(filterorder, wn, rp, rs, btype='high', ftype='ellip', output='sos')

    return sos

reduced_leads = ['DI', 'DII', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
all_leads = ['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def preprocess_ecg(ecg, sample_rate, leads, new_freq=None, new_len=None, scale=1,
                   use_all_leads=True, remove_baseline=False, powerline=None):
    print("entrada")
    print(ecg.shape)
    # Remove baseline
    if remove_baseline:
        sos = remove_baseline_filter(sample_rate)
        ecg_nobaseline = sgn.sosfiltfilt(sos, ecg, padtype='constant', axis=-1)
    else:
        ecg_nobaseline = ecg
    print("Remove baseline")
    print(ecg_nobaseline.shape)

    # Remove powerline
    if powerline is None:
        ecg_nopowerline = ecg_nobaseline
    else:
        # Design notch filter
        q = 30.0  # Quality factor
        b, a = sgn.iirnotch(powerline, q, fs=sample_rate)
        ecg_nopowerline = sgn.filtfilt(b, a, ecg_nobaseline)
    print("remove powerline")
    print(ecg_nopowerline.shape)

    # Resample
    #if new_freq is not None:
    #    print("none")
    #    ecg_resampled = sgn.resample_poly(ecg_nopowerline, up=new_freq, down=sample_rate, axis=-1)
    #else:
    #    print("else")
    #    ecg_resampled = ecg_nopowerline
    #    new_freq = sample_rate
    ecg_resampled = ecg_nopowerline
    n_leads, length = ecg_resampled.shape
    print("resample")
    print(ecg_resampled.shape)
    print(n_leads)

    # Rescale
    ecg_rescaled = scale * ecg_resampled
    print("rescale")
    print(ecg_rescaled.shape)

    # Add leads if needed
    target_leads = all_leads if use_all_leads else reduced_leads
    n_leads_target = len(target_leads)
    l2p = dict(zip(target_leads, range(n_leads_target)))
    ecg_targetleads = np.zeros([n_leads_target, length])
    print(l2p)
    print(ecg_rescaled.shape)
    print(ecg_targetleads.shape)
    ecg_targetleads = ecg_rescaled
    #for i, l in enumerate(leads):
    #    if l in target_leads:
    #        #ecg_targetleads[l2p[l], :] = ecg_rescaled[i, :]
    #        ecg_targetleads[l2p[l], :] = ecg_rescaled[:]
    if n_leads_target >= n_leads and use_all_leads:
        ecg_targetleads[l2p['DIII'], :] = ecg_targetleads[l2p['DII'], :] - ecg_targetleads[l2p['DI'], :]
        ecg_targetleads[l2p['AVR'], :] = -(ecg_targetleads[l2p['DI'], :] + ecg_targetleads[l2p['DII'], :]) / 2
        ecg_targetleads[l2p['AVL'], :] = (ecg_targetleads[l2p['DI'], :] - ecg_targetleads[l2p['DIII'], :]) / 2
        ecg_targetleads[l2p['AVF'], :] = (ecg_targetleads[l2p['DII'], :] + ecg_targetleads[l2p['DIII'], :]) / 2
    
    ###################################
    #ecg_targetleads = ecg_rescaled
    ###################################

    # Reshape
    if new_len is None or new_len == length:
        ecg_reshaped = ecg_targetleads
    elif new_len > length:
        ecg_reshaped = np.zeros([n_leads_target, new_len])
        pad = (new_len - length) // 2
        ecg_reshaped[..., pad:length+pad] = ecg_targetleads
    else:
        extra = (length - new_len) // 2
        ecg_reshaped = ecg_targetleads[:, extra:new_len + extra]

    return ecg_reshaped, new_freq, target_leads


def generateH5(input_file,out_file,new_freq=None,new_len=None,scale=1,powerline=None,use_all_leads=True,remove_baseline=False,root_dir=None,fmt='wfdb'):
    n = len(input_file)  # Get length
    try:
      h5f = h5py.File(f"{configVars.pathCasos}{out_file}", 'r+')
      h5f.clear()
    except:
      h5f = h5py.File(f"{configVars.pathCasos}{out_file}", 'w')

    ecg = input_file 
    sample_rate, leads = input_file.shape
    #ecg_preprocessed, new_rate, new_leads = preprocess_ecg(ecg, sample_rate, leads,
    #                                                                  new_freq=new_freq,
    #                                                                  new_len=new_len,
    #                                                                  scale=scale,
    #                                                                  powerline=powerline,
    #                                                                  use_all_leads=use_all_leads,
    #                                                                  remove_baseline=remove_baseline)    
    ecg_preprocessed = ecg
    n_leads, n_samples = ecg_preprocessed.shape
    x = h5f.create_dataset('tracings', (1, n_samples, n_leads), dtype='f8')
    x[0, :, :] = ecg_preprocessed.T

    h5f.close()
    
    
def LightX3ECG(
    train_loaders, 
    config,
    save_ckp_dir, 
    training_verbose = True, 
):
    model = torch.load(f"{save_ckp_dir}/best.ptl", map_location='cpu')
    #model = torch.load(f"{save_ckp_dir}/best.ptl", map_location = "cuda")
    model.to(torch.device('cpu'))
    with torch.no_grad():
        model.eval()
        running_preds = []
    
        for ecgs, labels in tqdm(train_loaders["pred"], disable = not training_verbose):  
            ecgs, labels = ecgs.cpu(), labels.cpu()
            #ecgs, labels = ecgs.cuda(), labels.cuda()
            logits = model(ecgs)
            preds = list(torch.max(logits, 1)[1].detach().cpu().numpy()) if not config["is_multilabel"] else list(torch.sigmoid(logits).detach().cpu().numpy())
            running_preds.extend(preds)
    
    if config["is_multilabel"]:
        running_preds =  np.array(running_preds)
        optimal_thresholds = pd.read_csv(f"{configVars.pathThresholds}CPSC-2018/optimal_thresholds_best.csv")
        preds = optimal_thresholds[optimal_thresholds["Threshold"]<=running_preds[0]]
        preds = preds['Pred'].values.tolist()
    else:   
        enfermedades = ['AFIB','GSVT','SB','SR']
        running_preds =  np.array(running_preds)
        #running_preds=np.reshape(running_preds, (len(running_preds),-1))
        preds = enfermedades[running_preds[0]]
    return preds


def getPredictedProbability(labels,preds):
    #labels debería ser las labels reales, que no hacen falta aquí en un principio
    # aux=np.array([])
    probs = []
    predLabels = []
    print(labels)
    for label, pred in zip(labels, preds):
        # aux = np.append(aux,pred[label])
        predLabel=np.array(pred).argmax()
        predLabels.append(predLabel)
        probs.append(pred[predLabel])
    print(predLabels)
    print(probs)
    return predLabels,probs


def get_r_count(ecg):
    counts = []
    for i in range(ecg.shape[0]):
        try:
            count = len(nk.ecg_peaks(ecg[i, :], sampling_rate=500)[1]['ECG_R_Peaks'].tolist())
        except:
            count = 0
        counts.append(count)
    print("####################")
    print(ecg)
    print(counts)
    print("####################")

    return max(set(counts), key = counts.count)