from tqdm import tqdm

import pandas as pd
import numpy as np

import torch
import torch.utils.data as D

def test(experiment_id, ds_test, model, bs, num_workers, device, debug):
    test_loader = D.DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=num_workers)

    model.load_state_dict(torch.load('models/best_model_'+experiment_id+'.pth'))
    model.eval()

    with torch.no_grad():
        preds = np.empty(0)
        for x, _ in tqdm(test_loader):
            x = x.to(device)
            output = model(x)
            idx = output.max(dim=-1)[1].cpu().numpy()
            preds = np.append(preds, idx, axis=0)

    return preds