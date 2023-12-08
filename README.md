# Non-Gaussian Latent Diffusion Models

## NGLDM

#### Evaluation and Generation

Download the weights for any model you want to test [here](https://huggingface.co/alexrosen45/NGLDM). For example, `models/cifar10/laplace/ngldm_cifar10_95.pth` would be the Laplace distribution model trained on CIFAR-10 after epoch 95. Navigate to the `diffusion` directory and run `eval.py` to evaluate FID scores with InceptionV3, or `generate.py` and `generate_large.py` to replicate the results found in the Appendix of the paper.

#### Training
Navigate to the `diffusion` directory and run `diffusion_cifar10.py` to train your model; see the bottom of this file to configure the script how you want. Each model on Hugging Face [(here)](https://huggingface.co/alexrosen45/NGLDM) took about 6-8hrs to train on a 3080 with a batch size of 10 over 200 epochs. To incorperate and autoencoder, you will need to modify `unet.py` and `ngldm.py`.

## Mini Diffusion

#### Train on mini MNIST dataset

Download the dataset from Assignment 3 [here](https://www.cs.toronto.edu/~rahulgk/courses/csc311_f23/index.html). Install the required libraries and run `diffusion.py` inside the `mini-diffusion` directory. For customization, see `noise.py` for valid distribution (noise type) arguments. For example,
```sh
python3 diffusion.py --noise_type="laplace" --epochs=1000
```
performs the diffusion forward process using noise sampled from a Laplace distribution with mean 0.

#### Evaluation

Generated images will be saved to ```results/{noise_type}/test.npz```. Run 

```sh
python3 eval.py --fake_data_dir results/{noise_type}/test.npz
```

to evaluate the FID score corresponding to the feature vectors of these images. It's important to note that FID score can only be computed when reasonable hyperparameters, like a sufficient number of epochs, are used during training.
