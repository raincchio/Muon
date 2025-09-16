# MuonG: An General optimizer for the hidden layers of neural networks

<img width="1242" height="840" alt="image" src="https://github.com/user-attachments/assets/46aea0e2-e740-45bc-97a5-992549371955" />

# update
<img width="668" height="91" alt="image" src="https://github.com/user-attachments/assets/a0b00d9b-b988-4b12-8ad4-8528119788bb" />


## Installation

```
pip install git+https://github.com/raincchio/Muon
```

## Usage

Muon is an optimizer for the hidden weights of a neural network.
Other parameters, such as embeddings, classifier heads, and hidden gains/biases should be optimized using standard AdamW.
Muon should be used as follows:

```python
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.90, 0.95), weight_decay=0.01)

# To replace the above, do the following:

from muon import MuonWithAuxAdam
hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
nonhidden_params = [*model.head.parameters(), *model.embed.parameters()]
param_groups = [
    dict(params=hidden_weights, use_muon=True,
         lr=0.02, weight_decay=0.01),
    dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
         lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
]
optimizer = MuonWithAuxAdam(param_groups)
```

