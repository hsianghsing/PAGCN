Created on Match, 1, 2022

Updated on June, 10, 2022

This is the Pytorch implementation for our paper:

@author: Guo


File：
```
│  dataloader.py
│  main.py
│  model.py
│  parse.py
│  Procedure.py
│  README.md
│  register.py
│  utils.py
│  world.py
│  
│          
└─data
    ├─ciao
    │      test_set.txt
    │      train_set.txt
    │      trust.txt
    │      
    ├─lastfm
    │      interaction_adj_mat.npz
    │      social_adj_mat.npz
    │      test_set.txt
    │      train_set.txt
    │      trust.txt
    │      
    └─Test
            test_set.txt
            train_set.txt
            trust.txt

```

# Environment Requirement
python==3.9.11

torch==1.11.0

cudatoolkit=11.3.1

torchaudio==0.11.0

torchvision==0.12.0

mkl==2021.4.0

mkl-service==2.4.0

numpy==1.21.2

pandas==1.4.1

scikit-learn==1.0.2

scipy==1.7.3

## Install pytorch
https://pytorch.org/get-started/locally/
# Dataset
Lastfm and ciao


# Run
Run with default parameters: bpr_batch=2048, latent_dim=64, layers=3, lr=0.001, decay=1e-3, personalities=2,
dataset=lastfm, epochs=2000, seed=2022, testbatch=100
```shell
python main.py
```

or 
```shell
python main.py --bpr_batch=2048 --latent_dim=64  --layers=3  --lr=0.001  --decay=1e-3   --personalities=2   --epochs=2000 
```
If you have multiple GPUs, please set the GPU number with:
```shell
--set_device=0
```

If you do not have GPU, please modify the `parse.py` with:
```python
    # parser.add_argument('--set_device', type=int, default=0)
```
# Modified code from

https://github.com/leo0481/SocialLGN

https://github.com/gusye1234/LightGCN-PyTorch

https://github.com/PTMZ/IMP_GCN

