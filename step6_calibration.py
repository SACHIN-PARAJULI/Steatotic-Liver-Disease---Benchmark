"""
SLD Risk Prediction Benchmark — Step 6: Calibration Evaluation
Input : splits.pkl, all_models.pkl
Output: calibration.pkl, calib_metrics.json, plots/calibration_*.png
"""
import pickle, json, warnings, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
warnings.filterwarnings('ignore')

os.makedirs('plots', exist_ok=True)
results     = pickle.load(open('all_models.pkl','rb'))
splits_data = pickle.load(open('splits.pkl','rb'))
splits      = splits_data['splits']
MODELS  = ['LogisticRegression','EBM','RandomForest','LightGBM']
TARGETS = ['y_steatosis','y_fibrosis']
N_BINS  = 10

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0,1,n_bins+1); ece_val = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() == 0: continue
        ece_val += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece_val / len(y_true)

def platt_calibrate(proba_tr, y_tr, proba_te):
    lr = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    lr.fit(proba_tr.reshape(-1,1), y_tr)
    return lr.predict_proba(proba_te.reshape(-1,1))[:,1]

def isotonic_calibrate(proba_tr, y_tr, proba_te):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(proba_tr, y_tr); return iso.predict(proba_te)

calib_metrics = {}; all_calibrated = {}
for target in TARGETS:
    s = splits[target]
    X_test,y_test = s['X_test'],s['y_test']
    X_ft_s,y_ft_s = s['X_ft_s'],s['y_ft_s']
    calib_metrics[target] = {}; all_calibrated[target] = {}
    fig, axes = plt.subplots(2, len(MODELS), figsize=(16,8))
    fig.suptitle(f'Calibration — {target}', fontsize=13, fontweight='bold')
    for col, mname in enumerate(MODELS):
        model    = results[target][mname]['model']
        proba_tr = model.predict_proba(X_ft_s)[:,1]
        proba_te = model.predict_proba(X_test)[:,1]
        brier_pre = round(brier_score_loss(y_test, proba_te),4)
        ece_pre   = round(compute_ece(y_test, proba_te),4)
        fp0,mp0   = calibration_curve(y_test, proba_te, n_bins=N_BINS, strategy='quantile')
        ax = axes[0,col]
        ax.plot(mp0,fp0,'s-',label='Before',color='steelblue')
        ax.plot([0,1],[0,1],'k--',lw=1,label='Perfect')
        ax.set_title(f'{mname}\nECE={ece_pre}  Brier={brier_pre}',fontsize=8)
        ax.set_xlabel('Mean predicted'); ax.set_ylabel('Fraction positive')
        ax.legend(fontsize=7); ax.set_xlim(0,1); ax.set_ylim(0,1)
        proba_platt = platt_calibrate(proba_tr,y_ft_s,proba_te)
        proba_iso   = isotonic_calibrate(proba_tr,y_ft_s,proba_te)
        brier_platt = round(brier_score_loss(y_test,proba_platt),4)
        brier_iso   = round(brier_score_loss(y_test,proba_iso),4)
        ece_platt   = round(compute_ece(y_test,proba_platt),4)
        ece_iso     = round(compute_ece(y_test,proba_iso),4)
        fp_p,mp_p   = calibration_curve(y_test,proba_platt,n_bins=N_BINS,strategy='quantile')
        fp_i,mp_i   = calibration_curve(y_test,proba_iso,  n_bins=N_BINS,strategy='quantile')
        ax2 = axes[1,col]
        ax2.plot(mp_p,fp_p,'s-',label=f'Platt ECE={ece_platt}',color='darkorange')
        ax2.plot(mp_i,fp_i,'s-',label=f'Isotonic ECE={ece_iso}',color='green')
        ax2.plot([0,1],[0,1],'k--',lw=1)
        ax2.set_title('After Calibration',fontsize=8)
        ax2.set_xlabel('Mean predicted'); ax2.set_ylabel('Fraction positive')
        ax2.legend(fontsize=7); ax2.set_xlim(0,1); ax2.set_ylim(0,1)
        calib_metrics[target][mname] = {
            'before':   {'ECE':ece_pre,   'Brier':brier_pre},
            'platt':    {'ECE':ece_platt, 'Brier':brier_platt},
            'isotonic': {'ECE':ece_iso,   'Brier':brier_iso}}
        all_calibrated[target][mname] = {
            'proba_raw':proba_te,'proba_platt':proba_platt,'proba_iso':proba_iso}
        print(f"[{target}] {mname:22}  Before ECE={ece_pre:.4f}  Platt={ece_platt:.4f}  Iso={ece_iso:.4f}")
    plt.tight_layout()
    plt.savefig(f'plots/calibration_{target}.png',dpi=150,bbox_inches='tight'); plt.close()

pickle.dump({'calib_metrics':calib_metrics,'calibrated':all_calibrated},open('calibration.pkl','wb'))
json.dump(calib_metrics, open('calib_metrics.json','w'), indent=2)
print("✅ Step 6 done")
