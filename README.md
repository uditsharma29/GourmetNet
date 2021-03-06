# GourmetNet
 
 We propose GourmetNet, a single-pass, end-to-end trainable network for food segmentation that achieves state-of-the-art performance. Food segmentation is an important problem as the first step for nutrition monitoring, food volume and calorie estimation.
Our novel architecture incorporates both channel attention and spatial attention information in an expanded multi-scale feature representation using our advanced Waterfall Atrous Spatial Pooling module.
GourmetNet refines the feature extraction process by merging features from multiple levels of the backbone through the two attention modules. The refined features are processed with the advanced multi-scale waterfall module that combines the benefits of cascade filtering and pyramid representations without requiring a separate decoder or postprocessing. Our experiments on two food datasets show that GourmetNet significantly outperforms existing current state-of-the-art methods

## Architecture

![model_HL2](https://user-images.githubusercontent.com/15017479/139361013-9dc4b319-cf8c-4be7-9d65-4d795c36844a.PNG)
