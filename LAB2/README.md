# CV Lab 2 - Report

## Mean-shift Implementation

### Mean-shift using for-loop

To implement the simple mean-shift method, I iterate all points by a for-loop. Then, for each point, I calculate its distance to all points, and calculate the Gaussian weights of all points when centering at the target point. Finally, we can update the target point by the weighted sum of all points. The algorithm is following the formula below. $K(x)$ denotes the Gaussian kernel function $K(x) = \frac{1}{b \sqrt{2\pi}} e^{\frac{1}{2}(\frac{x}{b})^2}$.  $(x_i - x)$ denotes the distance between the two points. $x_i \in R^3$, since we have RGB channels.

![ m(x) = \frac{ \sum_{x_i \in N(x)} K(x_i - x) x_i } {\sum_{x_i \in N(x)} K(x_i - x)} ](https://wikimedia.org/api/rest_v1/media/math/render/svg/fd7b010e1d7df763ffc64c2fe53d071e2c10d4f0)

### Mean-shift using batch acceleration

As I understand, the batch processing should take a batch of points as target points `x` in the function `distance(x, X)` rather than only one target point. Thus I adjust the for-loop in `meanshift_step()` to `for i in range(0, X.shape[0], BATCH_SIZE):`, such that in every iteration I update a batch size of points. The other batch-related functions are correspondingly adjusted from the for-loop version by adding additional dimensions during computation. Please see the code for details.

### Comparison

I ran the program over an i5-8300H CPU @ 2.30GHz for both for-loop and batch acceleration version. The result shows that for-loop version cost an average of $62.41$ seconds, while the batch acceleration version cost an average of $35.45$ seconds when I set the batch size as $400$. Interestingly, when I enlarged the batch size to $1000$, the running time did not decrease but increase to $38.21$ seconds. Smaller batch size of $20$ could speed up a bit to $34.33$ seconds. Besides, `torch` version was slower than `numpy` version on CPU.

