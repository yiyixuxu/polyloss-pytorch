# polyloss-pytorch

```python
class PolyLoss(softmax=False, ce_weight=None, reduction='mean', epsilon=1.0)
```

This class is used to compute the Poly-1 Loss between the `input` and `target` tensors.

Poly-1 Loss is defined as 
                
<img src="https://latex.codecogs.com/svg.image?L_\text{poly-1}&space;=-\log(P_t)&space;&plus;&space;\epsilon_1&space;\cdot&space;(1-P_t) ">

The predication `input` is compared with ground truth `target`. `Input` is expected to have shape `BNHW[D]` where `N` is number of classes. It can contains either logits or probabilities for each class, if passing logits as input, set `softmax=True`. `target` is expected to have shape `B1HW[D]` or `BNHW[D]` (one-hot format).

`epsilon` is the first polynomial coefficient in cross-entropy loss, in order to achieve best result, this value needs to be adjusted for different task and data. The optimal value for `epsilon` can be find through hyperparameter tunning 

The original paper: [Zhaoqi, L. et. al. (2022): PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions, 2022](https://arxiv.org/abs/2204.12511v1)

#### parameters
* `softmax (bool)` – if `True`, apply a softmax function to the prediction (i.e.`input`)
* `ce_weight(Tensor,optional)` – a manual rescaling weight given to each class. If given, has to be a Tensor of size `N`(it's same as `weight` argument for `nn.CrossEntropyLoss` class)
* `reduction(string, optional)` – Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the weighted mean of the output is taken, 'sum': the output will be summed.
* `epsilon`: the first polynomial coefficient. defaults to be `1.` 

#### Examples
```python
from PolyLoss import to_one_hot, PolyLoss

# Example of target in one-hot encoded format
loss = PolyLoss(softmax=True)
B, C, H, W = 2, 5, 3, 3
input = torch.rand(B, C, H, W, requires_grad=True)
target = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
target = to_one_hot(target[:, None, ...], num_classes=C)
output = loss(input, target)
output.backward()



# Example of target not in one-hot encoded format
loss = PolyLoss(softmax=True)
B, C, H, W = 2, 5, 3, 3
input = torch.rand(B, C, H, W, requires_grad=True)
target = torch.randint(low=0, high=C - 1, size=(B, 1, H, W)).long()
output = loss(input, target)
output.backward()
```

