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

def preprocess_ecg(ecg, sample_rate, leads, scale=1,
                   use_all_leads=True, remove_baseline=False):
    # Remove baseline
    if remove_baseline:
        sos = remove_baseline_filter(sample_rate)
        ecg_nobaseline = sgn.sosfiltfilt(sos, ecg, padtype='constant', axis=-1)
    else:
        ecg_nobaseline = ecg

    # Rescale
    ecg_rescaled = scale * ecg_nobaseline
    
     # Resample
    if sample_rate != 500:
        ecg_resampled = sgn.resample_poly(ecg_rescaled, up=500, down=sample_rate, axis=-1)
    else:
        ecg_resampled = ecg_rescaled
    length = len(ecg_resampled[0])
    
    # Add leads if needed
    target_leads = all_leads if use_all_leads else reduced_leads
    n_leads_target = len(target_leads)
    l2p = dict(zip(target_leads, range(n_leads_target)))
    ecg_targetleads = np.zeros([n_leads_target, length])
    ecg_targetleads = ecg_rescaled
    if n_leads_target >= leads and use_all_leads:
        ecg_targetleads[l2p['DIII'], :] = ecg_targetleads[l2p['DII'], :] - ecg_targetleads[l2p['DI'], :]
        ecg_targetleads[l2p['AVR'], :] = -(ecg_targetleads[l2p['DI'], :] + ecg_targetleads[l2p['DII'], :]) / 2
        ecg_targetleads[l2p['AVL'], :] = (ecg_targetleads[l2p['DI'], :] - ecg_targetleads[l2p['DIII'], :]) / 2
        ecg_targetleads[l2p['AVF'], :] = (ecg_targetleads[l2p['DII'], :] + ecg_targetleads[l2p['DIII'], :]) / 2

    return ecg_targetleads


def generateH5(input_file,out_file,new_freq=None,new_len=None,scale=1,sample_rate=None):
    n = len(input_file)  # Get length
    try:
      h5f = h5py.File(f"{configVars.pathCasos}{out_file}", 'r+')
      h5f.clear()
    except:
      h5f = h5py.File(f"{configVars.pathCasos}{out_file}", 'w')

    # Resample
    if new_freq is not None:
        ecg_resampled = sgn.resample_poly(input_file, up=new_freq, down=sample_rate, axis=-1)
    else:
        ecg_resampled = input_file
        new_freq = sample_rate
    n_leads, length = ecg_resampled.shape
    
    # Rescale
    ecg_rescaled = scale * ecg_resampled
    
    # Reshape
    if new_len is None or new_len == length:
        ecg_reshaped = ecg_rescaled
    elif new_len > length:
        ecg_reshaped = np.zeros([n_leads, new_len])
        pad = (new_len - length) // 2
        ecg_reshaped[..., pad:length+pad] = ecg_rescaled
    else:
        extra = (length - new_len) // 2
        ecg_reshaped = ecg_rescaled[:, extra:new_len + extra]
    
    n_leads, n_samples = ecg_reshaped.shape
    x = h5f.create_dataset('tracings', (1, n_samples, n_leads), dtype='f8')
    x[0, :, :] = ecg_reshaped.T
    h5f.close()
    
def LightX3ECG(
    train_loaders, 
    config,
    save_ckp_dir, 
):
    model = torch.load(f"{save_ckp_dir}/best.ptl", map_location='cpu')
    #model = torch.load(f"{save_ckp_dir}/best.ptl", map_location = "cuda")
    model.to(torch.device('cpu'))
    with torch.no_grad():
        model.eval()
        running_preds = []
    
        for ecgs in train_loaders["pred"]:  
            ecgs = ecgs.cpu()
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
