from tqdm import tqdm

import pandas as pd
import numpy as np

import torch
import torch.utils.data as D

def test(experiment_id, df_test, ds_test, plate_groups, experiment_type, model, bs, num_workers, device):
    test_loader = D.DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=num_workers)

    with torch.no_grad():
        for i, (x, _) in enumerate(tqdm(test_loader)):
            x = x.to(device)
            output = model(x).cpu().numpy()
            if i==0:
                preds = output
            else:
                preds = np.concatenate([preds, output], axis=0)
        print(preds.shape)

    assert len(preds) == len(df_test)
    mask = np.repeat(plate_groups[np.newaxis, :, experiment_type], len(preds), axis=0) != \
           np.repeat(df_test.plate.values[:, np.newaxis], 1108, axis=1)
    preds[mask] = 0

    preds = preds.argmax(1)

    return preds