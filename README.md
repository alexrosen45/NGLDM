# 311-project
CSC311 Diffusion Project

## Run on mini MNIST dataset

Download the dataset from Assignment 3 [here](https://www.cs.toronto.edu/~rahulgk/courses/csc311_f23/index.html)

Install the required libraries and run
```sh
python3 diffusion.py
```
inside the ```mini-diffusion``` directory. For customization, see the ```apply_markov_noise``` function in ```diffusion.py``` that lists valid distribution (noise type) arguments. For example,
```sh
python3 diffusion.py --noise_type="laplace" --epochs=1000
```
performs the diffusion forward process using noise sampled from a Laplace distribution with mean 0.

Generated images will be saved to ```results/mini-mnist/noise_name/test.npz```. In the same directory, run 

```sh
python3 eval.py --fake_data_dir results/mini_mnist/noise_name/test.npz  
```

to evaluate the FID score of the generated images.