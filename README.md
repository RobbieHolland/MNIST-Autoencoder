**MNIST-Autoencoder**
---
Removing noise and irregularities from MNIST digits in Tensorflow.
Below are two digits called <b>A</b> and <b>B</b>.

Original (A): <img src="http://imgur.com/uqErPKC.png" height="100"> Corrupted (A): <img src="http://imgur.com/6lbm2hv.png" height="100">

Original (B): <img src="http://imgur.com/2roXuBv.png" height="100"> Corrupted (B): <img src="http://imgur.com/WqXmKDS.png" height="100">

The network (with model layers of size [768, 400, <b>n</b>, 400, 768] where <b>n</b> is size of the encoded layer) then attempts to reconstruct the original from the corrupted version:

| Code Layer Size (n)  | Asymptotic Error  |  Reconstructed A | Reconstructed B |
|---|---|---|---|
|  10 |  2.2e6 |   <img src="http://imgur.com/UVZHqpU.png" height="150">    | <img src="http://imgur.com/zwm7K3E.png" height="150">|
| 20  |  1.5e6 | <img src="http://imgur.com/AMvXzbH.png" height="150">    | <img src="http://imgur.com/67Tosuw.png" height="150">|
| 30  |  1.0e6 | <img src="http://imgur.com/S5qaKLx.png" height="150">    | <img src="http://imgur.com/DuwkQEf.png" height="150">|

It is evident that a coding layer of size <b>10</b> is insufficient to reconstruct the original image. The '4' also resembling a '9' and the '5' a '6'.

A coding layer of size <b>20</b> accurately reconstructs the digits and removes irregularities, such as the swish on the tail of the original '4'.

A coding layer of size <b>30</b> also accurately reconstructs the digits but starts to retain irrelevant information such as the tail of the original '4'.

### Conclusion
---
At least for this network model of [768, 400, <b>n</b>, 400, 768] the best found <b>n</b> was around 20, where 'best' is defined as a balance between accurately reconstructing the image without retaining irrelevant features of the original.

### Afterthoughts
---
I have since been informed that using Sigmoid for activation is a bit outdated and that ReLU provides sufficient non-linearities and trains faster.
