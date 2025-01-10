---
layout: default
title:
permalink: /blogs/embed-fusion/
---




**Models:**

| Model     | NFCorpus | SciFact  | Dim |# Params |
| :-------- | :------- | :------- | :-: |:-: 	  |
| bge-small | 0.33708  | 0.72	  | 384 | 33M     |
| arctic-m  | 0.36201  | 0.71586  | 768 | 109M    |
| concat    | 0.37287  | 0.73095  | 1152| 142M    |


## Simple encoder:  1152 > 768 > 512 (`simple_m1`)

<details>
```python

import torch.nn as nn

class EncoderConfig:
    DEFAULT = {
        'input_dim': 1152,
        'hidden_dim': 768,
        'output_dim': 512,
        'dropout': 0.1
    }

class EncoderOnly(nn.Module):
    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or EncoderConfig.DEFAULT
        
        self.encoder = nn.Sequential(
            nn.Linear(cfg['input_dim'], cfg['hidden_dim']),
            nn.BatchNorm1d(cfg['hidden_dim']),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(cfg['dropout']),
            
            nn.Linear(cfg['hidden_dim'], cfg['output_dim']),
            nn.BatchNorm1d(cfg['output_dim']),
            nn.LeakyReLU(0.2, inplace=True)
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
        return self.encoder(x)[:,:dim]
```


Output-dim = 512

Numbers: 
	* NFCorpus: 0.31671
	* SciFact:  0.63541

* Epoch 1/30:  Train Loss: 0.000102, Val Loss: 0.000053
* Epoch 10/30: Train Loss: 0.000036, Val Loss: 0.000039


### Simple model: 1152 > 512



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
            nn.Linear(cfg['input_dim'], cfg['output_dim']),
            nn.BatchNorm1d(cfg['output_dim']),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(cfg['dropout'])
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
        return self.encoder(x)[:,:dim]
```
</details>




## Simple model: 1152 > 512 with Normalized, without dropout.


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



<details>


## Auto-Encoder: 

### With reconstruction loss:




### With MRL style loss:




## Attention based:


## TODO:

* Understand transformer dimensions 
* Throw a batch in it and see how it evolves.
* Code the thing
* to google ai studio: I have this bert model, I want to do surgery upon. and recover the outputs after layer 5. how can i do so?


* Back to the model;
* Back to AE, simple stuff.


* Understand the data..



<details>

Try:

```
class ImprovedSimilarityLoss(nn.Module):
    def __init__(self, weight: float = 1.0, margin: float = 0.1, temperature: float = 0.05):
        super().__init__()
        self.weight = weight
        self.margin = margin
        self.temperature = temperature

    def forward(self, model_output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute similarities
        sim_outputs = F.cosine_similarity(model_output.unsqueeze(1), 
                                        model_output.unsqueeze(0), dim=-1) / self.temperature
        sim_targets = F.cosine_similarity(targets.unsqueeze(1), 
                                        targets.unsqueeze(0), dim=-1) / self.temperature
        
        # Create mask for positive and negative pairs
        batch_size = sim_outputs.size(0)
        mask = torch.eye(batch_size, device=sim_outputs.device)
        
        # Compute loss with margin
        pos_loss = ((sim_outputs - sim_targets) ** 2) * mask
        neg_loss = torch.relu(sim_outputs - self.margin) * (1 - mask)
        
        loss = (pos_loss.sum() + neg_loss.sum()) / (batch_size * (batch_size - 1))
        return self.weight * loss
```
