# polyloss-pytorch

```python
class PolyLoss(softmax=False, ce_weight=None, reduction='mean', epsilon=1.0)
```

This class is used to compute the Poly-1 Loss between the `input` and `target` tensors.

Poly-1 Loss is defined as 
                
<img src="https://latex.codecogs.com/svg.image?L_\text{poly-1}&space;=-\log(P_t)&space;&plus;&space;\epsilon_1&space;\cdot&space;(1-P_t) ">

The predication `input` is compared with ground truth `target`. `Input` is expected to have shape `BNHW[D]` where `N` is number of classes. It can contains either logits or probabilities for each class, if passing logits as input, set `softmax=True`. `target` is expected to have shape `B1HW[D]`, `BHW[D]` or `BNHW[D]` (one-hot format).

`epsilon` is the first polynomial coefficient in cross-entropy loss, in order to achieve best result, this value needs to be adjusted for different task and data. The optimal value for `epsilon` can be found through hyperparameter tunning 

The original paper: [Zhaoqi, L. et. al. (2022): PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions, 2022](https://arxiv.org/abs/2204.12511v1)

### Parameters
* __softmax (bool)__ – if `True`, apply a softmax function to the prediction (i.e.`input`)
* __ce_weight(Tensor,optional)__ – a manual rescaling weight given to each class. If given, has to be a Tensor of size `N`(it's same as `weight` argument for `nn.CrossEntropyLoss` class)
* __reduction(string, optional)__ – Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the weighted mean of the output is taken, 'sum': the output will be summed.
* __epsilon__: the first polynomial coefficient. defaults to be `1.` 

### a colab tutorial: Use PolyLoss with Fast.ai and Weights & Biases

in [tutorial in colab](https://github.com/yiyixuxu/polyloss-pytorch/blob/master/tutorial_testing_polyloss_with_fastai_and_W%26B.ipynb), I provided an example of how to use PolyLoss in fastai (super easy!) and do a hyperparameter search with Weights & Biases. 


### How to Use 
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


# Example of PolyBCELoss
from PolyLoss import PolyBCELoss
loss = PolyBCELoss()
B, H, W = 2, 3, 3
input = torch.rand(B, H, W, requires_grad=True)
target = torch.empty(B,H,W).random_(2)
output = loss(input, target)
output.backward()

```
