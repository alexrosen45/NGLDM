# Non-Gaussian Latent Diffusion Models

## Mini Diffusion

#### Train on mini MNIST dataset

Download the dataset from Assignment 3 [here](https://www.cs.toronto.edu/~rahulgk/courses/csc311_f23/index.html)

Install the required libraries and run ```python3 diffusion.py``` inside the ```mini-diffusion``` directory. For customization, see ```noise.py``` for valid distribution (noise type) arguments. For example,
```sh
python3 diffusion.py --noise_type="laplace" --epochs=1000
```
performs the diffusion forward process using noise sampled from a Laplace distribution with mean 0.

#### Evaluation

Generated images will also be saved to ```results/{noise_type}/test.npz```. Run 

```sh
python3 eval.py --fake_data_dir results/{noise_type}/test.npz
```

to evaluate the FID score of these images. It's important to note that FID score can only be computed when reasonable hyperparameters, such as a sufficient number of epochs, are used during training.
