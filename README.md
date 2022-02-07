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
![headHunter_sag](https://user-images.githubusercontent.com/35701423/152793790-7d8b4f0f-9377-4c70-bbd8-f40a69c1551f.gif)
  
  tickerTracker
![heartHunter_cor](https://user-images.githubusercontent.com/35701423/152793641-78e61f94-59a5-4177-897a-b1fd4ca66dc0.gif)
