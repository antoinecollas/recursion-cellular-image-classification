from tqdm import tqdm

import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F

def test(df_test, ds_test, plate_groups, experiment_type, model, bs, num_workers, device):
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=num_workers)

    with torch.no_grad():
        for i, (x, _) in enumerate(tqdm(test_loader)):
            x = x.to(device)
            output = model(x=x, test_mode=True)
            output = F.softmax(output, 1)
            output = output.cpu().numpy()
            if i==0:
                preds = output
            else:
                preds = np.concatenate([preds, output], axis=0)

    def rescale(preds):
        temp = np.sum(preds, axis=1)
        temp[temp==0] = 1
        temp = np.repeat(temp[:, np.newaxis], preds.shape[1], axis=1)
        preds = preds / temp
        return preds

    assert len(preds) == len(df_test)
    mask = np.repeat(plate_groups[np.newaxis, :, experiment_type], len(preds), axis=0) != \
           np.repeat(df_test.plate.values[:, np.newaxis], 1108, axis=1)
    preds[mask] = 0
    preds = rescale(preds)

    results = np.zeros(preds.shape[0])
    for i in range(preds.shape[0]):
        max_per_row_idx = np.argmax(preds, axis=1)
        max_row_idx = np.argmax(preds[np.arange(len(preds)), max_per_row_idx])
        max_column_idx = max_per_row_idx[max_row_idx]
        max_prob = preds[max_row_idx, max_column_idx]
        results[max_row_idx] = max_column_idx
        preds[:, max_column_idx] = 0
        preds[max_row_idx, :] = 0
        preds = rescale(preds)

    return results