# [Rewrite the Stars](https://arxiv.org/abs/2403.19967) - CVPR'24

In this folder, we show how to plot the decision boundary for 2D points.

Please feel free to tune the parameters to check different decision boundaries.


```bash
# test network decision boundary
python main.py --model {model-name}

# test SVM decision boundary
python svc.py --kernel {poly} --coef0 4 --degree 4 --gamma scale --dataset_noise 0.4
python svc.py --kernel {rbf} --coef0 4 --degree 4 --gamma 4 --dataset_noise 0.4

```
