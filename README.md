# Neuraldrugs
![](/NeuralDrugsFavicon.png?raw=true)

This is a small Friday project aimed at bringing some fun to those who deal with Neural Networks and got a bit tired of them. The idea is to emulate the impact of alcohol and other substances on a Neural Network. The project outcomes can be interesting. It can also help you measure the stability of your networks which can be quite useful.

Now let's take WaveNet as an example. WaveNet is a generative neural network architecture for image generation. ([https://github.com/Zeta36/tensorflow-image-wavenet](https://github.com/Zeta36/tensorflow-image-wavenet)) As a learning range, let's feed the network these photos of an object from different angles:

![](https://68.media.tumblr.com/f9a0cebe9bdff59af181da38728caad2/tumblr_inline_okgbhbd2Vf1stzac2_540.png)

The machine learning outcome for a clean network looks like some average case of this object. It can be guessed easily by its outline. It is black and white because the image has been restored from the network through the weight of the color contrast neurons. The resolution is only 64x64 because of the convolution but you can see it easily:

![](https://68.media.tumblr.com/351c98855736bc5024af5f62f1cfe0b9/tumblr_inline_okgbigYBWQ1stzac2_540.png)

This network consists of 16384 neurons (18 layers with a different number of neurons). Our parameter 0.01 means that 0.01% of the total neuron number will be damaged i.e. only one neuron after rounding. Take a look at the result of the random value weight change with only one random neuron out of 16384 in the network:

Now let's run our program and see the result:

`# ./neuraldrugs.py ./logdir/train/2017-01-24T06-34-00/model.ckpt-58352 --set_weights_random --dosage 0.01`

This network consists of 16384 neurons (18 layers with a different number of neurons). Our parameter 0.01 means that 0.01% of the total neuron number will be damaged i.e. only one neuron after rounding. Take a look at the result of the random value weight change with only one random neuron out of 16384 in the network:

![](https://68.media.tumblr.com/59877f35acb669134fa5c82afd53ec66/tumblr_inline_okgbm3qJ6B1stzac2_540.png)
