<h1> AutoGAF: Automatic General Anatomy Finder </h1>
<h2> A general framework for 3D location-finding in CT scans </h2>

The idea of this framework is that it's a fully generalised version of previous specific location-finding models (headHunter, neckNavigator, etc.). With this you can quickly and easily create new location-specific models in just a few lines of code.

The examples are currently set up to generate a Heart-locating model (heartHunter / tickerTracker).

<h2>What you'll need </h2>
<li> A training dataset of CT images (nifti format) ~100? <br>
<li> Masks of the structures you wish to find the centre-of-mass of for the training images (nifti format) <br>
<li> A linux machine with a GPU (ideally one with the 2080ti's or a 3090) <br>

<h2>Instructions </h2>
<li> Preprocessing <br>
<li> Training <br>
<li> Inference <br>

<h2>Training examples </h2>
  headHunter

![headHunter_sag](https://user-images.githubusercontent.com/35701423/152800442-d12a5904-040a-4f6b-a46f-6f47af9a59b9.gif)

  heartHunter
![heartHunter_cor](https://user-images.githubusercontent.com/35701423/152800422-7b194f56-e602-4e35-8837-0898dc63d26d.gif)
