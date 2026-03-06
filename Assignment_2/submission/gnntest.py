#!/usr/bin/env python3

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

data = "archive/ml-100k"

tot_rat = 5
relative_ct = tot_rat*2

epochs = 150
bet_layer = 64
basss = 2
d_r = 0.7
e_r = 0.4
lr = 0.015
wgt_decay = 5e-4
cncl = 3
rukjao = 20
eval_after = 5


def data_load_krne(root,k):
    cols = ["user_id","item_id","rating","timestamp"]
    tr = pd.read_csv(os.path.join(root,f"u{k}.base"),sep="\t",names=cols)
    te = pd.read_csv(os.path.join(root,f"u{k}.test"),sep="\t",names=cols)
    # yeh test aur train de rha 
    for df in [tr,te]:
        df["user_id"] -= 1
        df["item_id"] -= 1
        df["rating"] -= 1
    # sb 0 to n-1 me shift kr diya ease rhega predictions me + 1 kr dunga 

    return tr,te

def make_graph(df,n_users,device):
    u = torch.tensor(df["user_id"].values,dtype=torch.long,device=device)
    i = torch.tensor(df["item_id"].values,dtype=torch.long,device=device)
    r = torch.tensor(df["rating"].values,dtype=torch.long,device=device)

    i_shift = i + n_users

    fwd = torch.stack([u,i_shift])
    bwd = torch.stack([i_shift,u])

    rel_fwd = r
    rel_bwd = r + tot_rat

    edge_idx = torch.cat([fwd,bwd],dim=1)
    edge_type = torch.cat([rel_fwd,rel_bwd],dim=0)
    # edges and type return kr rha 
    return edge_idx,edge_type


class SmallGCMC(nn.Module):
    def __init__(self,n_users,n_items):
        super().__init__()
        self.n_users = n_users
        self.u_embed = nn.Embedding(n_users,bet_layer)
        self.i_embed = nn.Embedding(n_items,bet_layer)
        self.g1 = RGCNConv(bet_layer,bet_layer,num_relations=relative_ct,num_bases=basss)
        self.g2 = RGCNConv(bet_layer,bet_layer,num_relations=relative_ct,num_bases=basss)
        self.norm1 = nn.LayerNorm(bet_layer)
        self.norm2 = nn.LayerNorm(bet_layer)
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(d_r)
        self.dec_bases = nn.Parameter(torch.randn(4,bet_layer,bet_layer))
        self.dec_coeff = nn.Parameter(torch.randn(tot_rat,4))

        self.init_all()

    def init_all(self):
        nn.init.xavier_normal_(self.u_embed.weight) # xavier init 
        nn.init.xavier_normal_(self.i_embed.weight)
        nn.init.xavier_normal_(self.dec_bases)
        nn.init.xavier_normal_(self.dec_coeff)

    def forward(self,e_idx,e_type,src,dst):
        x = torch.cat([self.u_embed.weight,self.i_embed.weight],dim=0)

        h1 = self.g1(x,e_idx,e_type)
        h1 = self.norm1(h1)
        h1 = self.act(h1)
        h1 = self.drop(h1)

        h2 = self.g2(h1,e_idx,e_type)
        h2 = self.norm2(h2)
        h2 = self.act(h2)
        h2 = self.drop(h2)

        z = h2 + x

        u_vec = z[src]
        i_vec = z[dst + self.n_users]

        Q = torch.einsum("rb,bio -> rio",self.dec_coeff,self.dec_bases)

        out = []
        for rr in range(tot_rat):
            q = Q[rr]
            u_q = torch.matmul(u_vec,q)
            score = (u_q * i_vec).sum(dim=1)
            out.append(score)

        logits = torch.stack(out,dim=1)
        return logits


def train_one(fold_id,model_id,train_df,e_idx,e_type,test_src,test_dst,test_lbl,n_users,n_items):
    device = e_idx.device

    model = SmallGCMC(n_users,n_items).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wgt_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=0.5,patience=5)

    train_src = torch.tensor(train_df["user_id"].values,dtype=torch.long,device=device)
    train_dst = torch.tensor(train_df["item_id"].values,dtype=torch.long,device=device)
    train_lbl = torch.tensor(train_df["rating"].values,dtype=torch.long,device=device)

    best = 0.0
    best_state = None
    bad = 0

    print(f"  training model {model_id+1}")

    for ep in range(1,epochs + 1):
        model.train()
        opt.zero_grad()

        mask = torch.rand(e_idx.size(1),device=device) > e_r
        sub_edges = e_idx[:,mask]
        sub_types = e_type[mask]

        logits = model(sub_edges,sub_types,train_src,train_dst)
        loss = F.cross_entropy(logits,train_lbl)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()

        if ep % eval_after == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(e_idx,e_type,test_src,test_dst)
                preds = val_logits.argmax(dim=1)
                acc = (preds == test_lbl).float().mean().item()

            print(f"    epoch {ep} | loss {loss.item():.4f} | acc {acc:.4f} | lr {opt.param_groups[0]['lr']:.6f}")

            sched.step(acc)

            if acc > best:
                best = acc
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
            else:
                bad += 1

            if bad >= rukjao:
                break

    model.load_state_dict(best_state)
    return model


def council_run(fold_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr,te = data_load_krne(data,fold_id)

    full = pd.read_csv(os.path.join(data,"u.data"),sep="\t",names=["u","i","r","t"])
    n_users = full["u"].max()
    n_items = full["i"].max()

    e_idx,e_type = make_graph(tr,n_users,device)

    test_src = torch.tensor(te["user_id"].values,dtype=torch.long,device=device)
    test_dst = torch.tensor(te["item_id"].values,dtype=torch.long,device=device)
    test_lbl = torch.tensor(te["rating"].values,dtype=torch.long,device=device)

    ensemble = 0

    print(f"\nfold {fold_id} starting...")

    for m in range(cncl):
        model = train_one(fold_id,m,tr,e_idx,e_type,test_src,test_dst,test_lbl,n_users,n_items)
        with torch.no_grad():
            logits = model(e_idx,e_type,test_src,test_dst)
            ensemble += logits

    ensemble /= cncl
    final_pred = ensemble.argmax(dim=1)
    final_acc = (final_pred == test_lbl).float().mean().item()
    f1_scr = torch.zeros(tot_rat,device=device)
    for r in range(tot_rat):
        tp = ((final_pred == r) & (test_lbl == r)).sum().item()
        fp = ((final_pred == r) & (test_lbl != r)).sum().item()
        fn = ((final_pred != r) & (test_lbl == r)).sum().item()

        prec = tp / (tp+fp+1e-8)
        rec = tp / (tp+fn+1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        f1_scr[r] = f1
    return final_acc,f1_scr.cpu().numpy().mean()


def main():
    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.exists(data):
        print("data missing")
        return

    results = []
    f1_scr_f = []
    for f in range(1,6):
        acc,f1_scr  = council_run(f)
        results.append(acc)
        f1_scr_f.append(f1_scr)
        print("fold",f,"acc:",acc,"f1:",f1_scr)
    avg = np.mean(results)
    std = np.std(results)
    f1_scr_avg = np.mean(f1_scr_f,axis=0)
    f1_scr_std = np.std(f1_scr_f,axis=0)
    print("\nfinal results")
    for i,v in enumerate(results):
        print("fold",i+1,v)
    for i in range(tot_rat):
        print(f"f1 score for rating {i+1}: mean {f1_scr_f[i]:.4f}")
    print("mean:",avg,"std:",std)
    print("mean_f1:",f1_scr_avg,"std_f1:",f1_scr_avg)


if __name__ == "__main__":
    main()
