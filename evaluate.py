import os
from typing import Dict, Optional, List
import cv2
import pandas as pd
import pyiqa
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import argparse
from pathlib import Path
from glob import glob


FROM_FOLDER_LISTING_RULE = lambda path: sorted(glob(os.path.join(path, '*.png')) + glob(os.path.join(path, '*.PNG')) +
                                               glob(os.path.join(path, '*.jpg')) + glob(os.path.join(path, '*.jpeg')) +
                                               glob(os.path.join(path, '*.JPG')) + glob(os.path.join(path, '*.JPEG')))


def read_image(path: str) -> torch.Tensor:
    img = torch.FloatTensor(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)/255
    return img.to(device="cuda")

class Valdataset(Dataset):
    def __init__(self, pred_paths, gt_paths):
        super().__init__()
        self.pred_paths = pred_paths
        self.gt_folder = Path(gt_paths[0]).parent
    def __len__(self):
        return len(self.pred_paths)
    def __getitem__(self, index):
        pred_path = self.pred_paths[index] 
        gt_path = self.gt_folder / Path(pred_path).name 

        pred = read_image(pred_path).unsqueeze(0)
        im_extra = read_image(gt_path).unsqueeze(0)
        return {'pred':pred, 'gt': im_extra, 'path': [Path(pred_path).stem]}


def estimate_metrics(paths_to_predictions: List[str], paths_to_targets: List[str],
                     bs: int,
                     pyiqa_metric_calcs, pyiqa_metrics) -> pd.DataFrame:

    if bs == 1:
        df = pd.DataFrame(columns=['Name'] + pyiqa_metrics)
        for i, (pred, targ) in tqdm(enumerate(zip(paths_to_predictions, paths_to_targets)),
                                        total=len(paths_to_targets)):
            img_name = os.path.basename(targ).split('.')[0]
            metrics = []
            pred = read_image(pred).unsqueeze(0)
            targ = read_image(targ).unsqueeze(0)
            for mc_name, mc in zip(pyiqa_metrics, pyiqa_metric_calcs):
                with torch.no_grad():
                    metric = mc(pred, targ).item()
                metrics.append(metric)
            df.loc[i] = [img_name] + metrics
    else:
        dataset = Valdataset(pred_paths=paths_to_predictions, 
                             gt_paths=paths_to_targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False)
        metrics = defaultdict(list)
        for data in tqdm(dataloader):
            pred, gt, names = data['pred'], data['gt'], data['path']
            metrics['Name'].extend(names[0])
            for mc_name, mc in zip(pyiqa_metrics, pyiqa_metric_calcs):
                with torch.no_grad():
                    metric = mc(pred.squeeze(1), gt.squeeze(1))
                torch.cuda.empty_cache()
                metrics[mc_name].extend(metric.squeeze().tolist())
        df = pd.DataFrame.from_dict(metrics)
                         
    df.loc[-1] = [None] + list(df.iloc[:, 1:].mean(axis=0))
    df.index = df.index + 1
    df.sort_index(inplace=True)
    df = df.rename({0: 'avg'})
    avg_dict = df.loc['avg'].drop('Name').to_dict()
    print("=== Average Metrics ===")
    print(f"Averages: {avg_dict}")
    print("=======================")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,  help="dataset to process")
    parser.add_argument("-bs", type=int, default=1)
    parser.add_argument("--inference_fold", type=str)
    parser.add_argument("--gt_fold", type=str, default="")
    parser.add_argument("--metrics_fold", type=str)
    args = parser.parse_args()

    pyiqa_metrics = ['clipiqa', 'musiq', 'niqe', 'maniqa-pipal']
    # if args.dataset != 'RealSet65':
    if args.gt_fold != "":
        pyiqa_metrics += ["psnr", "ssim", "lpips", 'dists']
        paths_to_targets = FROM_FOLDER_LISTING_RULE(args.gt_fold)
    else:
        paths_to_targets = FROM_FOLDER_LISTING_RULE(args.inference_fold)

    pyiqa_metric_calcs = []
    for m in pyiqa_metrics:
        if m == "psnr":
            pyiqa_metric_calcs.append(pyiqa.create_metric('psnr', device="cuda", as_loss=False, test_y_channel=True, color_space='ycbcr'))            
        elif m == "ssim":
            pyiqa_metric_calcs.append(pyiqa.create_metric('ssim', device="cuda", as_loss=False, test_y_channel=True, color_space='ycbcr'))            
        else:
            pyiqa_metric_calcs.append(pyiqa.create_metric(m, device="cuda", as_loss=False))

    results = estimate_metrics(paths_to_predictions=FROM_FOLDER_LISTING_RULE(args.inference_fold),
                               paths_to_targets=paths_to_targets,
                               bs=args.bs,
                               pyiqa_metric_calcs=pyiqa_metric_calcs, pyiqa_metrics=pyiqa_metrics)
    os.makedirs(args.metrics_fold, exist_ok=True)
    results.to_csv(os.path.join(args.metrics_fold, f"{args.dataset}.csv"))
