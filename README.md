Basic
```shell
CUDA_VISIBLE_DEVICES=3 nohup python main.py --optimizer Lion  --epochs 200 --batch_size 64 --model resnet50_raw --lr 3e-4 --op_parameters full --num_classes 7 --dataset rafdb --lr_scheduler  --save --weight_decay 0.0 --log_dir  model/
```

Adversarial learning
```shell
CUDA_VISIBLE_DEVICES=0  nohup python cloudleak_main.py --optimizer Lion --epochs 264 --adv_train cw --log_dir random_rafdb_facepp_32sample --n_sample 64 --sample_times 32 --lr_warmup_epoch 64 --adaptive random --batch_size 64 --model resnet50 --lr 3e-4 --op_parameters full --validate_interval 1 --num_classes 7 --dataset rafdb --lr_scheduler --save --adv_valid pgd --api facepp &
```

Mixmatch
```shell
CUDA_VISIBLE_DEVICES=2 nohup python main.py --optimizer Lion  --epochs 1000 --batch_size 64 --model resnet50_raw --lr 3e-4 --op_parameters full --validate_interval 1 --num_classes 7 --dataset expw --lr_scheduler  --save --weight_decay 0.0 --log_dir mixmatch/10_22_ran_128_new_1000_329 --hapi_info fer/expw/microsoft_fer/22-05-23 --label_batch 128 --seed 329 --validate_interval 20 &
```

Adaptive
```shell
CUDA_VISIBLE_DEVICES=2 nohup python main.py --optimizer Lion  --epochs 264 --batch_size 64 --model resnet50_raw --lr 3e-4 --op_parameters full --validate_interval 1 --num_classes 7 --dataset rafdb --lr_scheduler  --save --weight_decay 0.0 --sample_times 64  --n_samples 64 --adaptive kcenter --lr_warmup_epoch 64 --log_dir adp/420/6_22_420_b64_lion_r50raw_adp_64_kcenter  --hapi_info fer/rafdb/microsoft_fer/22-05-23 --seed 420 --validate_interval 1 &
```


Still needs to be organized, if you have questions please email or submit an issue!


```txt
@inproceedings{liang2024model,
  title={Model extraction attacks revisited},
  author={Liang, Jiacheng and Pang, Ren and Li, Changjiang and Wang, Ting},
  booktitle={Proceedings of the 19th ACM Asia Conference on Computer and Communications Security},
  pages={1231--1245},
  year={2024}
}
```
