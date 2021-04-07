# Batch_FID_PyTorchPix2Pix
FID is one of the main metrics used to evaluate generative adversarial networks models, so in this repository we propose a calculation method to check the FID score over each epoch using the test data, the method is formated to the [PyTorch implementation of Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix#cyclegan-and-pix2pix-in-pytorch) and uses the packages [move-my-files](https://pypi.org/project/move-my-files/) and [pytorch-fid](https://pypi.org/project/pytorch-fid/). The method is composed of a few steps:

  1) test model with 5th_EpochCheckpoint
  2) Separate REAL images and FAKE images of target domain
  3) Calculate FID between REAL_B ←→ FAKE_B
  4) test model with 10th_EpochCheckpoint ..... AND LOOP AGAIN UNTIL 200 / 400 epoch checkpoint

*Note that the FID is calculated over the real data and generated data --> INPUT DATA IS SIMPLY IGNORED*

### Basic Usage

Run `test.py` of all checkpoints of the trained model using the [PyTorch implementation of Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix#cyclegan-and-pix2pix-in-pytorch)
```
!python test.py  --num_test 160 --epoch 5 --dataroot ./datasets/EXPERIMENT_NAME --name EXPERIMENT_NAME --netG --norm  --phase test --preprocess --model pix2pix
!python test.py  --num_test 160 --epoch 10 --dataroot ./datasets/EXPERIMENT_NAME --name EXPERIMENT_NAME --netG --norm --phase test --preprocess --model pix2pix
!python test.py  --num_test 160 --epoch 15 --dataroot ./datasets/EXPERIMENT_NAME --name EXPERIMENT_NAME --netG --norm --phase test --preprocess --model pix2pix
!python test.py  --num_test 160 --epoch 20 --dataroot ./datasets/EXPERIMENT_NAME --name EXPERIMENT_NAME --netG --norm --phase test --preprocess --model pix2pix
...
```


Then open `Batch_FID_PyTorchPix2Pix.ipynb` and set `EXPERIMENT_NAME` and `Epoch_range`

Run the cells to generate the lines of code for the calculation method

Copy the output and run on the next cell --> CALCULATE FID FOR EACH EPOCH

### TODO

Generate a script to delete generated files after calculation

Append fid_value from pytorch-fid to a list

Create csv file with FID per EPOCH

Plot a line graph to check increase or decrease on the quality of the generated images

Think about other implementation methods like: input domain ←→ target domain or the usability of it with CycleGAN...



