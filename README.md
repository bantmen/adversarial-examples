# Adversarial examples

![alt text](https://image.ibb.co/e6FAV6/final.png "Sample visualization")

## Instructions
1) Train a simple CNN on mnist by running `python train.py`. You should see an output similar to `Test accuracy 0.9853`. The model is now saved under `./data/mnist_conv/`.
2) Generate adversarial samples for the model in first step. For example,
`python gen_adv.py --verbose=0 --victim_class=2 --wanted_class=6 --num_iterations=30 --num_viz_samples=10 --viz_path=adv_grid_viz.png`. If you are curious about misclassification rate and other useful information, then set `verbose=1`.
3) View the visual saved at `adv_grid_viz.png`. The first column corresponds to samples from the mnist test set that the classifier predicted correctly, while the second column are adversarially modified versions of the former that were misclassified as the target class and the last column is a grayscale image of the difference between the images in second and first column. Keep in mind that, since black is represented as 0 and white is represented as 1, images of the third column will be darker in positions with smaller differences.

## How
- Fast gradient sign method ([fgsm][1]) is normally applied in a single step to create an adversarial version of a sample when the attack is not targeted.
- However, since we want to create adversarial examples to be confused as a given class, fgsm is applied in small steps via gradient descent manner to minimize the cross entropy of the victim samples with respect to the target class.
- Values outside of [0, 1] are clipped to make the adversarial examples more interesting.

## TODO
- Reproduce [saliency map][2] based approach
- Look into retraining the targeted net with adversarial examples
- Look into Hessian (or other curvature) based targeted attacks
- Research/think of different attack methods

[1]:https://arxiv.org/abs/1412.6572
[2]:https://arxiv.org/abs/1511.07528