from pathlib import Path
import gc, json, time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import OneClassSVM

# ============================================================
# ZeroShield AI - FINAL 6-MODEL TRAIN/EVALUATE SCRIPT
# Models: AE, VAE, DAE, CAE, Deep SVDD, One-Class SVM
# Training: benign only
# Validation: held-out benign + regular attacks
# Zero-day: held-out Mirai + Web-Based attacks
# Large CSVs are always evaluated in chunks.
# ============================================================

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "train_benign_only.csv"
VALID = ROOT / "validation_set.csv"
ZERO = ROOT / "zero_day_test_scaled_v2.csv"
OUT = ROOT / "model_training" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
THRESHOLD_PERCENTILE = 95   # matches the user's current AE run; use one value for all 6
TRAIN_SAMPLE_SIZE = 500_000
TRAIN_SCAN_CHUNK = 100_000
EVAL_CHUNK = 50_000
BATCH = 2048
EPOCHS = 8
LR = 1e-3
LATENT = 16
NOISE_STD = 0.05
CAE_LAMBDA = 1e-4
OCSVM_ROWS = 10_000
OCSVM_NU = 0.05
OCSVM_GAMMA = "scale"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
torch.manual_seed(SEED)

SAMPLE_FILE = OUT / "zero_shield_benign_sample_500k.npy"
FEATURE_FILE = OUT / "zero_shield_feature_columns.json"
MASTER_FILE = OUT / "six_model_master_results.csv"


def section(s):
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)


def get_features():
    if FEATURE_FILE.exists():
        return json.loads(FEATURE_FILE.read_text(encoding="utf-8"))
    h = pd.read_csv(TRAIN, nrows=0)
    f = [c for c in h.columns if c not in {"label", "binary_label"}]
    FEATURE_FILE.write_text(json.dumps(f, indent=2), encoding="utf-8")
    return f


def get_benign_sample(features):
    if SAMPLE_FILE.exists():
        x = np.load(SAMPLE_FILE).astype(np.float32, copy=False)
        if x.ndim != 2 or x.shape[1] != len(features):
            raise ValueError("Saved benign sample does not match the current feature list.")
        print(f"Using existing common benign sample: {x.shape[0]:,} x {x.shape[1]}")
        return x

    print(f"Creating common benign sample ({TRAIN_SAMPLE_SIZE:,} rows)...")
    rng = np.random.default_rng(SEED)
    reservoir = np.empty((TRAIN_SAMPLE_SIZE, len(features)), dtype=np.float32)
    seen = 0
    for chunk in pd.read_csv(TRAIN, usecols=features, chunksize=TRAIN_SCAN_CHUNK, low_memory=True):
        a = chunk.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32, copy=False)
        for row in a:
            seen += 1
            if seen <= TRAIN_SAMPLE_SIZE:
                reservoir[seen - 1] = row
            else:
                j = rng.integers(0, seen)
                if j < TRAIN_SAMPLE_SIZE:
                    reservoir[j] = row
        print(f"  scanned {seen:,} benign rows", end="\r", flush=True)
    n = min(seen, TRAIN_SAMPLE_SIZE)
    reservoir = reservoir[:n]
    np.save(SAMPLE_FILE, reservoir)
    print(f"\nSaved: {SAMPLE_FILE}")
    return reservoir


def labels(chunk):
    y = pd.to_numeric(chunk["binary_label"], errors="coerce")
    miss = y.isna()
    if miss.any():
        fallback = chunk.loc[miss, "label"].astype(str).str.strip().ne("BenignTraffic").astype(np.int8)
        y.loc[miss] = fallback.to_numpy()
    if y.isna().any():
        raise ValueError("Could not resolve all binary_label values.")
    y = y.astype(np.int8).to_numpy()
    bad = np.setdiff1d(np.unique(y), np.array([0, 1], dtype=np.int8))
    if len(bad):
        raise ValueError(f"Unexpected binary_label values: {bad.tolist()}")
    return y


def clean_df(chunk, features):
    return chunk[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)


def metrics(tp, tn, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    fpr = fp / (fp + tn) if fp + tn else None
    spec = tn / (tn + fp) if tn + fp else None
    return {"precision": p, "recall": r, "f1": f1, "accuracy": acc, "fpr": fpr,
            "specificity": spec, "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def eval_large(path, features, scorer, threshold, show_categories=False):
    tp = tn = fp = fn = processed = 0
    cats = {}
    usecols = features + ["binary_label", "label"]
    for i, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=EVAL_CHUNK, low_memory=True), 1):
        xdf = clean_df(chunk, features)
        xnp = xdf.to_numpy(dtype=np.float32, copy=False)
        y = labels(chunk)
        score = scorer(xdf, xnp)
        pred = (score >= threshold).astype(np.int8)
        tp += int(((pred == 1) & (y == 1)).sum())
        tn += int(((pred == 0) & (y == 0)).sum())
        fp += int(((pred == 1) & (y == 0)).sum())
        fn += int(((pred == 0) & (y == 1)).sum())
        if show_categories:
            lab = chunk["label"].astype(str).to_numpy()
            for c in np.unique(lab):
                m = lab == c
                yy, pp = y[m], pred[m]
                d = cats.setdefault(c, {"tp":0,"tn":0,"fp":0,"fn":0})
                d["tp"] += int(((pp==1)&(yy==1)).sum())
                d["tn"] += int(((pp==0)&(yy==0)).sum())
                d["fp"] += int(((pp==1)&(yy==0)).sum())
                d["fn"] += int(((pp==0)&(yy==1)).sum())
        processed += len(chunk)
        if i == 1 or i % 20 == 0:
            print(f"  processed {processed:,} rows...", flush=True)
    cat_metrics = {c: metrics(**d) for c, d in cats.items()}
    return {"rows": processed, "overall": metrics(tp, tn, fp, fn), "categories": cat_metrics}


def torch_loader(x):
    return DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=BATCH, shuffle=True, num_workers=0,
                      pin_memory=torch.cuda.is_available())


class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU(), nn.Linear(32,LATENT))
        self.decoder = nn.Sequential(nn.Linear(LATENT,32), nn.ReLU(), nn.Linear(32,64), nn.ReLU(), nn.Linear(64,d))
    def encode(self,x): return self.encoder(x)
    def decode(self,z): return self.decoder(z)
    def forward(self,x): return self.decode(self.encode(x))

class DenoisingAutoencoder(AE):
    """
    Denoising Autoencoder.

    The architecture is the same as the standard Autoencoder.
    The difference is during training: noisy input is provided
    to the network while the original clean input is used as
    the reconstruction target.
    """
    pass

class VAE(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.fc1=nn.Linear(d,64); self.fc2=nn.Linear(64,32); self.mu=nn.Linear(32,LATENT); self.logvar=nn.Linear(32,LATENT)
        self.fc3=nn.Linear(LATENT,32); self.fc4=nn.Linear(32,64); self.fc5=nn.Linear(64,d)
    def encode(self,x):
        h=torch.relu(self.fc1(x)); h=torch.relu(self.fc2(h)); return self.mu(h), self.logvar(h)
    def decode(self,z):
        h=torch.relu(self.fc3(z)); h=torch.relu(self.fc4(h)); return self.fc5(h)
    def forward(self,x):
        mu,lv=self.encode(x); std=torch.exp(0.5*lv); z=mu+torch.randn_like(std)*std; return self.decode(z),mu,lv


class CAE(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.e1=nn.Linear(d,64,bias=False); self.e2=nn.Linear(64,LATENT,bias=False); self.d1=nn.Linear(LATENT,64); self.d2=nn.Linear(64,d)
    def forward(self,x):
        h=torch.sigmoid(self.e1(x)); z=self.e2(h); r=self.d2(torch.relu(self.d1(z))); return r,h
    def penalty(self,h):
        deriv2=(h*(1-h))**2
        row_norm=(self.e1.weight**2).sum(dim=1)
        return (deriv2*row_norm).sum(dim=1).mean()


class DeepSVDD(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,64,bias=False),nn.ReLU(),nn.Linear(64,32,bias=False),nn.ReLU(),nn.Linear(32,LATENT,bias=False))
    def forward(self,x): return self.net(x)


def train_ae(model,x,denoise=False):
    model.to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=LR); loss_fn=nn.MSELoss(); last=0.0
    for ep in range(1,EPOCHS+1):
        total=n=0; model.train()
        for (b,) in torch_loader(x):
            b=b.to(DEVICE); inp=b + torch.randn_like(b)*NOISE_STD if denoise else b
            opt.zero_grad(); r=model(inp); loss=loss_fn(r,b); loss.backward(); opt.step(); total+=loss.item(); n+=1
        last=total/max(n,1); print(f"Epoch {ep:02d}/{EPOCHS} | loss={last:.6f}")
    return model,last


def train_vae(model,x):
    model.to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=LR); last=0.0
    for ep in range(1,EPOCHS+1):
        total=n=0; model.train()
        for (b,) in torch_loader(x):
            b=b.to(DEVICE); opt.zero_grad(); r,mu,lv=model(b)
            rec=nn.functional.mse_loss(r,b); kl=-0.5*torch.mean(1+lv-mu.pow(2)-lv.exp()); loss=rec+0.01*kl
            loss.backward(); opt.step(); total+=loss.item(); n+=1
        last=total/max(n,1); print(f"Epoch {ep:02d}/{EPOCHS} | loss={last:.6f}")
    return model,last


def train_cae(model,x):
    model.to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=LR); mse=nn.MSELoss(); last=0.0
    for ep in range(1,EPOCHS+1):
        total=n=0; model.train()
        for (b,) in torch_loader(x):
            b=b.to(DEVICE); opt.zero_grad(); r,h=model(b); loss=mse(r,b)+CAE_LAMBDA*model.penalty(h); loss.backward(); opt.step(); total+=loss.item(); n+=1
        last=total/max(n,1); print(f"Epoch {ep:02d}/{EPOCHS} | loss={last:.6f}")
    return model,last


@torch.no_grad()
def svdd_center(model,x):
    model.eval(); total=None; n=0
    loader=DataLoader(TensorDataset(torch.from_numpy(x)),batch_size=BATCH,shuffle=False,num_workers=0)
    for (b,) in loader:
        z=model(b.to(DEVICE)); s=z.sum(0); total=s if total is None else total+s; n+=len(b)
    c=total/max(n,1); eps=1e-3; c=torch.where(c.abs()<eps,torch.where(c>=0,torch.full_like(c,eps),torch.full_like(c,-eps)),c); return c.detach()


def train_svdd(model,x):
    model.to(DEVICE); c=svdd_center(model,x); opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-6); last=0.0
    for ep in range(1,EPOCHS+1):
        total=n=0; model.train()
        for (b,) in torch_loader(x):
            b=b.to(DEVICE); opt.zero_grad(); z=model(b); loss=((z-c)**2).sum(dim=1).mean(); loss.backward(); opt.step(); total+=loss.item(); n+=1
        last=total/max(n,1); print(f"Epoch {ep:02d}/{EPOCHS} | loss={last:.6f}")
    return model,c,last


@torch.no_grad()
def ae_scores(model,x):
    model.eval(); out=[]
    for s in range(0,len(x),BATCH):
        b=torch.from_numpy(x[s:s+BATCH]).to(DEVICE); r=model(b); out.append(((b-r)**2).mean(1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def vae_scores(model,x):
    model.eval(); out=[]
    for s in range(0,len(x),BATCH):
        b=torch.from_numpy(x[s:s+BATCH]).to(DEVICE); mu,_=model.encode(b); r=model.decode(mu); out.append(((b-r)**2).mean(1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def cae_scores(model,x):
    model.eval(); out=[]
    for s in range(0,len(x),BATCH):
        b=torch.from_numpy(x[s:s+BATCH]).to(DEVICE); r,_=model(b); out.append(((b-r)**2).mean(1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def svdd_scores(model,c,x):
    model.eval(); out=[]
    for s in range(0,len(x),BATCH):
        b=torch.from_numpy(x[s:s+BATCH]).to(DEVICE); z=model(b); out.append(((z-c)**2).sum(1).cpu().numpy())
    return np.concatenate(out)


def save_json(name,typ,threshold,loss,val,zero,model_file):
    obj={"model":name,"type":typ,"threshold":float(threshold),"threshold_percentile":THRESHOLD_PERCENTILE,"training_loss":loss,
         "validation":val,"zero_day":zero,"saved_model":str(model_file)}
    p=OUT/f"{name}_results.json"; p.write_text(json.dumps(obj,indent=4),encoding="utf-8"); return p


def update_master(name,typ,loss,threshold,val,zero):
    vm=val["overall"]; zm=zero["overall"]
    row={"Model":name,"Type":typ,"Threshold_%":THRESHOLD_PERCENTILE,"Threshold":threshold,"Training_Loss":loss,
         "Validation_Precision":vm["precision"],"Validation_Recall":vm["recall"],"Validation_F1":vm["f1"],
         "Validation_Accuracy":vm["accuracy"],"Validation_FPR":vm["fpr"],"Validation_Specificity":vm["specificity"],
         "ZeroDay_Detection_Rate":zm["recall"]}
    new=pd.DataFrame([row])
    if MASTER_FILE.exists():
        old=pd.read_csv(MASTER_FILE); old=old[old.Model!=name]; new=pd.concat([old,new],ignore_index=True)
    new.to_csv(MASTER_FILE,index=False)


def display(name,threshold,loss,val,zero):
    vm=val["overall"]; zm=zero["overall"]
    print("\n"+"="*62)
    print(f"{name.upper()} FINAL PRELIMINARY RESULT")
    print("="*62)
    print(f"Threshold percentile: {THRESHOLD_PERCENTILE}%")
    print(f"Threshold:           {threshold:.8f}")
    if loss is not None: print(f"Final train loss:     {loss:.6f}")
    print("\nREGULAR VALIDATION")
    print(f"Rows:       {val['rows']:,}")
    print(f"Accuracy:   {vm['accuracy']:.4f} ({vm['accuracy']*100:.2f}%)")
    print(f"Precision:  {vm['precision']:.4f} ({vm['precision']*100:.2f}%)")
    print(f"Recall:     {vm['recall']:.4f} ({vm['recall']*100:.2f}%)")
    print(f"F1-score:   {vm['f1']:.4f} ({vm['f1']*100:.2f}%)")
    print(f"FPR:        {vm['fpr']:.4f} ({vm['fpr']*100:.2f}%)")
    print(f"Specificity:{vm['specificity']:.4f} ({vm['specificity']*100:.2f}%)")
    print(f"Confusion: TP={vm['tp']:,} | TN={vm['tn']:,} | FP={vm['fp']:,} | FN={vm['fn']:,}")
    print("\nZERO-DAY")
    print(f"Total attacks: {zero['rows']:,}")
    print(f"Detected:      {zm['tp']:,}")
    print(f"Missed:        {zm['fn']:,}")
    print(f"Detection rate:{zm['recall']:.4f} ({zm['recall']*100:.2f}%)")
    print("FPR/Specificity: N/A (zero-day file contains attacks only)")
    print("\nZero-day category detection:")
    for c,m in sorted(zero["categories"].items()):
        print(f"  {c:24s}: {m['recall']*100:7.2f}% ({m['tp']:,}/{m['tp']+m['fn']:,})")


def run_torch_model(name,typ,model,train_fn,score_train_fn,score_eval_fn,save_fn,features,x):
    section(name)
    model,loss=train_fn(model,x)
    model_file=OUT/f"{name}.pt"; save_fn(model,model_file)
    tr=score_train_fn(model,x); threshold=float(np.percentile(tr,THRESHOLD_PERCENTILE))
    scorer=lambda xdf,xnp: score_eval_fn(xnp)
    print("\nEvaluating validation_set.csv...")
    val=eval_large(VALID,features,scorer,threshold,False)
    print("\nEvaluating zero_day_test_scaled_v2.csv...")
    zero=eval_large(ZERO,features,scorer,threshold,True)
    rf=save_json(name,typ,threshold,loss,val,zero,model_file); update_master(name,typ,loss,threshold,val,zero); display(name,threshold,loss,val,zero)
    print(f"\nModel saved to: {model_file}\nResults saved to: {rf}")
    del model; gc.collect();
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def main():
    check=[TRAIN,VALID,ZERO]
    for p in check:
        if not p.exists(): raise FileNotFoundError(p)
    section("ZEROSHIELD AI - SIX MODEL EXPERIMENT")
    print(f"Device: {DEVICE}")
    print(f"Threshold rule: {THRESHOLD_PERCENTILE}th percentile of benign training scores")
    features=get_features(); print(f"Features: {len(features)}")
    x=create_or_load=get_benign_sample(features)
    d=x.shape[1]

    section("AUTOENCODER")
    ae,loss=train_ae(AE(d),x,False)
    af=OUT/"Autoencoder.pt"
    torch.save({"state_dict":ae.state_dict(),"input_dim":d,"latent_dim":LATENT,"features":features},af)
    threshold=float(np.percentile(ae_scores(ae,x),THRESHOLD_PERCENTILE))
    scorer=lambda xdf,xnp: ae_scores(ae,xnp)
    val=eval_large(VALID,features,scorer,threshold,False)
    zero=eval_large(ZERO,features,scorer,threshold,True)
    rf=save_json("Autoencoder","DL",threshold,loss,val,zero,af)
    update_master("Autoencoder","DL",loss,threshold,val,zero)
    display("Autoencoder",threshold,loss,val,zero)
    print(f"\nModel saved to: {af}\nResults saved to: {rf}")
    del ae; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    section("VARIATIONAL AUTOENCODER")
    vae,loss=train_vae(VAE(d),x); vf=OUT/"Variational_Autoencoder.pt"; torch.save({"state_dict":vae.state_dict(),"input_dim":d,"latent_dim":LATENT,"features":features},vf)
    threshold=float(np.percentile(vae_scores(vae,x),THRESHOLD_PERCENTILE)); scorer=lambda xdf,xnp: vae_scores(vae,xnp)
    val=eval_large(VALID,features,scorer,threshold,False); zero=eval_large(ZERO,features,scorer,threshold,True); rf=save_json("Variational_Autoencoder","DL",threshold,loss,val,zero,vf); update_master("Variational_Autoencoder","DL",loss,threshold,val,zero); display("Variational_Autoencoder",threshold,loss,val,zero); print(f"\nModel saved to: {vf}\nResults saved to: {rf}"); del vae; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    section("DENOISING AUTOENCODER")
    dae,loss=train_ae(DenoisingAutoencoder(d),x,True); df=OUT/"Denoising_Autoencoder.pt"; torch.save({"state_dict":dae.state_dict(),"input_dim":d,"latent_dim":LATENT,"features":features,"noise_std":NOISE_STD},df)
    threshold=float(np.percentile(ae_scores(dae,x),THRESHOLD_PERCENTILE)); scorer=lambda xdf,xnp: ae_scores(dae,xnp); val=eval_large(VALID,features,scorer,threshold,False); zero=eval_large(ZERO,features,scorer,threshold,True); rf=save_json("Denoising_Autoencoder","DL",threshold,loss,val,zero,df); update_master("Denoising_Autoencoder","DL",loss,threshold,val,zero); display("Denoising_Autoencoder",threshold,loss,val,zero); print(f"\nModel saved to: {df}\nResults saved to: {rf}"); del dae; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    section("CONTRACTIVE AUTOENCODER")
    cae,loss=train_cae(CAE(d),x); cf=OUT/"Contractive_Autoencoder.pt"; torch.save({"state_dict":cae.state_dict(),"input_dim":d,"latent_dim":LATENT,"features":features,"contractive_lambda":CAE_LAMBDA},cf)
    threshold=float(np.percentile(cae_scores(cae,x),THRESHOLD_PERCENTILE)); scorer=lambda xdf,xnp: cae_scores(cae,xnp); val=eval_large(VALID,features,scorer,threshold,False); zero=eval_large(ZERO,features,scorer,threshold,True); rf=save_json("Contractive_Autoencoder","DL",threshold,loss,val,zero,cf); update_master("Contractive_Autoencoder","DL",loss,threshold,val,zero); display("Contractive_Autoencoder",threshold,loss,val,zero); print(f"\nModel saved to: {cf}\nResults saved to: {rf}"); del cae; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    section("DEEP SVDD")
    svdd,c,loss=train_svdd(DeepSVDD(d),x); sf=OUT/"Deep_SVDD.pt"; torch.save({"state_dict":svdd.state_dict(),"center":c.cpu(),"input_dim":d,"latent_dim":LATENT,"features":features},sf)
    threshold=float(np.percentile(svdd_scores(svdd,c,x),THRESHOLD_PERCENTILE)); scorer=lambda xdf,xnp: svdd_scores(svdd,c,xnp); val=eval_large(VALID,features,scorer,threshold,False); zero=eval_large(ZERO,features,scorer,threshold,True); rf=save_json("Deep_SVDD","DL",threshold,loss,val,zero,sf); update_master("Deep_SVDD","DL",loss,threshold,val,zero); display("Deep_SVDD",threshold,loss,val,zero); print(f"\nModel saved to: {sf}\nResults saved to: {rf}"); del svdd,c; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    section("ONE-CLASS SVM")
    rng=np.random.default_rng(SEED); idx=rng.choice(len(x),size=min(OCSVM_ROWS,len(x)),replace=False); xsvm=x[idx]
    print(f"OC-SVM training rows: {len(xsvm):,}")
    svm=OneClassSVM(kernel="rbf",nu=OCSVM_NU,gamma=OCSVM_GAMMA,cache_size=1024,shrinking=True,tol=1e-3); print("Training One-Class SVM..."); svm.fit(xsvm)
    sf=OUT/"One_Class_SVM.joblib"; tr=-svm.decision_function(pd.DataFrame(xsvm,columns=features)); threshold=float(np.percentile(tr,THRESHOLD_PERCENTILE)); joblib.dump({"model":svm,"features":features,"threshold":threshold},sf)
    scorer=lambda xdf,xnp: -svm.decision_function(xdf); val=eval_large(VALID,features,scorer,threshold,False); zero=eval_large(ZERO,features,scorer,threshold,True); rf=save_json("One_Class_SVM","ML",threshold,None,val,zero,sf); update_master("One_Class_SVM","ML",None,threshold,val,zero); display("One_Class_SVM",threshold,None,val,zero); print(f"\nModel saved to: {sf}\nResults saved to: {rf}")

    section("SIX MODEL RUN COMPLETE")
    if MASTER_FILE.exists(): print(pd.read_csv(MASTER_FILE).to_string(index=False))


if __name__ == "__main__":
    main()
