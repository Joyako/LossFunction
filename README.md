## Face Recognition in PyTorch 
### This project mainly contains various loss function and light neural network.

### Light Neural Network
<li> MobileNet-v3
<li> EfficientNet

### Loss function:  
These functions addresses deep face recognition (FR) problem under open-set protocol, 
where ideal face features are expected to have smaller maximal intra-class distance 
than minimal inter-class distance under a suitably chosen met- ric space.
<li> Softmax:  

<a href="https://www.codecogs.com/eqnedit.php?latex=L_{1}=-\frac{1}{N}&space;\sum_{i=1}^{N}&space;\log&space;\frac{e^{W_{y_{i}}^{T}&space;x_{i}&plus;b_{y_{i}}}}{\sum_{j=1}^{n}&space;e^{W_{j}^{T}&space;x_{i}&plus;b_{j}}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L_{1}=-\frac{1}{N}&space;\sum_{i=1}^{N}&space;\log&space;\frac{e^{W_{y_{i}}^{T}&space;x_{i}&plus;b_{y_{i}}}}{\sum_{j=1}^{n}&space;e^{W_{j}^{T}&space;x_{i}&plus;b_{j}}}" title="L_{1}=-\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{W_{y_{i}}^{T} x_{i}+b_{y_{i}}}}{\sum_{j=1}^{n} e^{W_{j}^{T} x_{i}+b_{j}}}" /></a>
<li> CenterLoss:   
<li> L-Softmax:
<li> Angular-Softmax:  
<li> Cosine-Softmax:  


### Reference
<li> https://github.com/MuggleWang/CosFace_pytorch/blob/master/layer.py  
<li>





