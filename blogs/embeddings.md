---
layout: default
title:
permalink: /blogs/embed-fusion/
---


**Verified: all-4**

| Model     | NFCorpus | SciFact  | ArguAna  | Dim |# Params |
| :-------- | :------- | :------- | :------- | :-: |:-: 	 |
| arctic-m  | 0.36201  | 0.71586  | 0.5953   | 768 | 109M    |
| concat-4  | 0.37881  | 0.74585  | 0.62107  | 1920| 208M    |
| Leader    | 0.35762  | 0.73472  | 0.74956  | 768 | 150M    |



**Encoder-1024**

| Epoch     | NFCorpus | SciFact  | ArguAna  | 
| :-------- | :------- | :------- | :------- | 
| 05	    | 0.37122  | 0.72673  | 0.60192  | 
| 10        | 0.37345  | 0.72857  | 0.60424  | 
| 20        | 0.37242  | 0.7298   | 0.60437  | 
| **arctic-m**| 0.36201  | 0.71586  | 0.5953   | 
| **concat-4**| 0.37881  | 0.74585  | 0.62107  | 



**Encoder-1024, Truncated to 768 to match arctic-m:**

| Model     | NFCorpus | SciFact  | ArguAna  |
| :-------- | :------- | :------- | :------- |
| ep-10     | 0.36415  | 0.73006  | 0.60147  |
| ep-20     | 0.36432  | 0.73051  | 0.60236  |
| **arctic-m**| 0.36201  | 0.71586  | 0.5953 | 


**Models:**

| Model     | NFCorpus | SciFact  | ArguAna  | Dim |# Params |
| :-------- | :------- | :------- | :------- | :-: |:-: 	 |
| bge-small | 0.33708  | 0.72	  | 0.59499  | 384 | 33M     |
| gist      | 0.34691   | 0.70858 | 0.5912   | 384 | 33M     |
| e5-small  | 0.31806  | 0.67965  | 0.46865 | 384 | 33M     |







## Baseline encoder: 1152 > 512

```python
class EncoderConfig:
    DEFAULT = {
        'input_dim': 1152,
        'output_dim': 512,
        'dropout': 0.1
    }
class EncoderOnly(nn.Module):
    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or EncoderConfig.DEFAULT
        
        self.encoder = nn.Sequential(
            nn.LayerNorm(cfg['input_dim']),
            nn.Linear(cfg['input_dim'], cfg['output_dim']),
            nn.BatchNorm1d(cfg['output_dim']),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.LayerNorm(cfg['output_dim']),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, dim):
        out = self.encoder(x)
        return F.normalize(out[:,:dim], p=2, dim=1)
```

<details>


* Loss with MRL on `COMPRESSED_DIMENSIONS = [32, 64, 128, 256, 384, 512]`.  

* Some numbers:

| Epoch     | SciFact | NFCorpus   | Train / Val Loss   |  NFCor-384 |
| :-------- | :------ | :-------   | :----------------  |  :-------  |
| Epoch 001 | 0.x     | 0.34725	   | 0.000778  0.000317 |  0.        |
| Epoch 003 | 0.x     | 0.3592     | 0.000258  0.000274 |  0.        |
| Epoch 005 | 0.70568 | 0.36465    | 0.000225  0.000253 |  0.        |
| Epoch 007 | 0.x     | 0.3639     | 0.000222  0.000250 |  0.        |
| Epoch 009 | 0.x     | 0.36421    | 0.000219  0.000247 |  0.        |
| Epoch 012 | 0.x     | 0.36438    | 0.000218  0.000246 |  0.        |
| Epoch 015 | 0.70823 | **0.36441**| 0.000218  0.000247 |  0.36485   |
|[bge-small]| 0.72	  | 0.33708    | 
| [arctic-m]| 0.71586 | 0.36201    | 
| [Concat]  | 0.73095 | 0.37287    | 



### Normalize the concatenated training data: e5-small-arctic-m:

**Baseline -- naive concat:**

| Model     | NFCorpus | SciFact  | ArguAna | Dim |# Params |
| :-------- | :------- | :------- | :-------| :-: |:-: 	  |
| e5-small  | 0.31806  | 0.67965  | 0.46865 | 384 | 33M     |
| arctic-m  | 0.36201  | 0.71586  | 0.51724 | 768 | 109M    |
| concat    | 0.37044  | 0.51902  | 0.52922 | 1152| 142M    |

**Normed, with `[256, 512, 784]`:**

| Epoch     | NFCorpus | SciFact | ArguAna    | Train / Val Loss   | 
| :-------- | :------- | :------ | :------    | :----------------  | 
| Epoch 005 | 0.34412  | 0.52015 | 0.52427    | 0.000014  0.000015 | 
| Epoch 020 | 0.34194  | 0.52226 | 0.5248     | 12/15			   | 

**Normed, with `[64, 128, 256, 512, 784]`:**


| Epoch     | NFCorpus | SciFact | ArguAna    | Train / Val Loss   |
| :-------- | :------- | :------ | :------    | :----------------  |
| Epoch 010 | 0.3279   | 0.45692 | 0.51755    | 12/15			   |
| Epoch 015 | 0.33329  | 0.4868  | 0.52189    | 12/15			   |

### Raw concatenation (no norm.) of training data: e5-small-arctic-m:

**No-Norm, with `[256, 512, 784]`:**

| Epoch     | NFCorpus | SciFact | ArguAna    | Train / Val Loss   |  NFCor-384 |
| :-------- | :------- | :------ | :------    | :----------------  |  :-------  |
| Epoch 005 | 0.34206  | 0.52015 | 0.52427    | 0.000014  0.000015 |  0.        |
| Epoch 020 | 0.34194  | 0.52226 | 0.5248     | 12/15			   |  0.        |

**No-Norm, with `[64, 128, 256, 512, 784]`:**

| Epoch     | NFCorpus | SciFact | ArguAna    | Train / Val Loss   |  NFCor-384 |
| :-------- | :------- | :------ | :------    | :----------------  |  :-------  |
| Epoch 005 | 0.34412  | 0.52015 | 0.52427    | 0.000014  0.000015 |  0.        |
| Epoch 020 | 0.34194  | 0.52226 | 0.5248     | 12/15			   |  0.        |

</details>
