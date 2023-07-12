
import os
from dataset.rafdb import RAFDB
from torch.utils.data import DataLoader

import models
os.environ["USL_MODE"] = "USL"

import numpy as np
import torch

import usl_utils
from usl_utils import cfg, logger, print_b

usl_utils.init(default_config_file="usl/configs/cifar10_usl.yaml")

logger.info(cfg)


print_b("Loading model")

checkpoint = 'exp_mix/6_22_gt_200_1688/model.pth'

train_memory_dataset = RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"train"),hapi_data_dir='/home/jkl6486/HAPI',hapi_info='fer/rafdb/microsoft_fer/22-05-23',api=None,transform='Normal')
# test_dataset = RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"valid"),hapi_data_dir='/home/jkl6486/HAPI',hapi_info='fer/rafdb/microsoft_fer/22-05-23',api=None,transform='Normal')
task = 'emotion'
    
model = getattr(models,'resnet')(norm_par=train_memory_dataset.norm_par,model_name='resnet50',num_classes=7)
model.load_state_dict(checkpoint)
model.eval()


print_b("Loading dataset")

# train_memory_dataset, train_memory_loader = usl_utils.train_memory_cifar(
#     root_dir=cfg.DATASET.ROOT_DIR,
#     batch_size=cfg.DATALOADER.BATCH_SIZE,
#     workers=cfg.DATALOADER.WORKERS, transform_name=cfg.DATASET.TRANSFORM_NAME, cifar100=cifar100)

train_memory_loader = DataLoader(dataset=train_memory_dataset,
                    batch_size=64,
                    shuffle=False,sampler=None, pin_memory=True,
                    num_workers=8,drop_last=False)

targets = torch.tensor(train_memory_dataset.targets)
targets.shape

# %get feature list%
print_b("Loading feat list")
feats_list = usl_utils.get_feats_list(
    model, train_memory_loader, recompute=cfg.RECOMPUTE_ALL, force_no_extra_kwargs=True)

# %epresentativeness: Select Density Peaks%
print_b("Calculating first order kNN density estimation")
d_knns, ind_knns = usl_utils.partitioned_kNN(
    feats_list, K=cfg.USL.KNN_K, recompute=cfg.RECOMPUTE_ALL)
neighbors_dist = d_knns.mean(dim=1)
score_first_order = 1/neighbors_dist


num_centroids, final_sample_num = usl_utils.get_sample_info_cifar(
    chosen_sample_num=cfg.USL.NUM_SELECTED_SAMPLES)
logger.info("num_centroids: {}, final_sample_num: {}".format(
    num_centroids, final_sample_num))


recompute_num_dependent = cfg.RECOMPUTE_ALL or cfg.RECOMPUTE_NUM_DEP
for kMeans_seed in cfg.USL.SEEDS:
    print_b(f"Running k-Means with seed {kMeans_seed}")
    if final_sample_num <= 40:
        # This is for better reproducibility, but has low memory usage efficiency.
        force_no_lazy_tensor = True
    else:
        force_no_lazy_tensor = False
#Diversity: Pick One in Each Cluster return centroids
    # This has side-effect: it calls torch.manual_seed to ensure the seed in k-Means is set.
    # Note: NaN in centroids happens when there is no corresponding sample which belongs to the centroid
    cluster_labels, centroids = usl_utils.run_kMeans(feats_list, num_centroids, final_sample_num, Niter=cfg.USL.K_MEANS_NITERS,
                                                 recompute=recompute_num_dependent, seed=kMeans_seed, force_no_lazy_tensor=force_no_lazy_tensor)
  
    print_b("Getting selections with regularization")
    selected_inds = usl_utils.get_selection(usl_utils.get_selection_with_reg, feats_list, neighbors_dist, cluster_labels, num_centroids, final_sample_num=final_sample_num, iters=cfg.USL.REG.NITERS, w=cfg.USL.REG.W,
                                        momentum=cfg.USL.REG.MOMENTUM, horizon_dist=cfg.USL.REG.HORIZON_DIST, alpha=cfg.USL.REG.ALPHA, verbose=True, seed=kMeans_seed, recompute=recompute_num_dependent, save=True)

    counts = np.bincount(np.array(train_memory_dataset.targets)[selected_inds])

    print("Class counts:", sum(counts > 0))
    print(counts.tolist())

    print("max: {}, min: {}".format(counts.max(), counts.min()))

    print("Number of selected indices:", len(selected_inds))
    print("Selected IDs:")
    print(repr(selected_inds))
