# Simple Baselines for Image Restoration

Liangyu Chen<sup>⋆</sup>, Xiaojie Chu<sup>⋆</sup>, Xiangyu Zhang, and Jian Sun 

MEGVII Technology, Beijing, CN 

{chenliangyu,chuxiaojie,zhangxiangyu,sunjian}@megvii.com 

Abstract. Although there have been significant advances in the field of image restoration recently, the system complexity of the state-of-the-art (SOTA) methods is increasing as well, which may hinder the convenient analysis and comparison of methods. In this paper, we propose a simple baseline that exceeds the SOTA methods and is computationally eficient. To further simplify the baseline, we reveal that the nonlinear activation functions, e.g. Sigmoid, ReLU, GELU, Softmax, etc. are not necessary: they could be replaced by multiplication or removed. Thus, we derive a Nonlinear Activation Free Network, namely NAFNet, from the baseline. SOTA results are achieved on various challenging benchmarks, e.g. 33.69 dB PSNR on GoPro (for image deblurring), exceeding the previous SOTA 0.38 dB with only 8.4% of its computational costs; 40.30 dB PSNR on SIDD (for image denoising), exceeding the previous SOTA 0.28 dB with less than half of its computational costs. The code and the pre-trained models are released at github.com/megvii-research/NAFNet. 

Keywords: Image Restoration, Image Denoise, Image Deblur 

## 1 Introduction

With the development of deep learning, the performance of image restoration methods improve significantly. Deep learning based methods[5,37,39,36,6,7,32,8,25] have achieved tremendous success. E.g. [39] and [8] achieve 40.02/33.31 dB of PSNR on SIDD[1]/GoPro[26] for image denoising/deblurring respectively. 

Despite their good performance, these methods sufer from high system complexity. For a clear discussion, we decompose the system complexity into two parts: inter-block complexity and intra-block complexity. First, the inter-block complexity, as shown in Figure 2. [7,25] introduce connections between varioussized feature maps; [5,37] are multi-stage networks and the latter stage refine the results of the previous stage. Second, the intra-block complexity, i.e. the various design choices inside the block. E.g. Multi-Dconv Head Transposed Attention Module and Gated Dconv Feed-Forward Network in [39] (as we shown in Figure 3a), Swin Transformer Block in [22], HINBlock in [5], and etc. It is not practical to evaluate the design choices one by one. 

Based on the above facts, a natural question arises: Is it possible that a network with low inter-block and low intra-block complexity can achieve SOTA performance? To accomplish the first condition (low inter-block complexity), this paper adopts the single-stage UNet as architecture (following some SOTA methods[39,36]) and focuses on the second condition. To this end, we start with a plain block with the most common components, i.e. convolution, ReLU, and shortcut[14]. From the plain block, we add/replace components of SOTA methods and verify how much performance gain do these components bring. By extensive ablation studies, we propose a simple baseline, as shown in Figure 3c, that exceeds the SOTA methods and is computationally eficient. It has the potential to inspire new ideas and make their verification easier. The baseline, which contains GELU[15] and Channel Attention Module[16] (CA), can be further simplified: we reveal that the GELU in the baseline can be regarded as a special case of the Gated Linear Unit[10] (GLU), and from this we empirically demonstrate that it can be replaced by a simple gate, i.e. element-wise product of feature maps. In addition, we reveal the similarity of the CA to GLU in form, and the nonlinear activation functions in CA could be removed either. In conclusion, the simple baseline could be further simplified to a nonlinear activation free network, noted as NAFNet. We mainly conduct experiments on SIDD[1] for image denoising, and GoPro[26] for image deblurring, following [5,39,37]. The main results are shown in Figure 1, our proposed baseline and NAFNet achieves SOTA results while being computationally eficient: 33.40/33.69 dB on GoPro, exceed previous SOTA[8] 0.09/0.38 dB, respectively, with 8.4% of its computational cost; 40.30 dB on SIDD, exceed [39] 0.28 dB with less than half of its computational costs. Extensive quantity and quality experiments are conducted to illustrate the efectiveness of our proposed baselines. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/1c38ec72677b4d09ac1238b5ba84f6c877179a1731d1d2c7f62e151820f7546f.jpg)



Fig. 1: PSNR vs. computational cost on Image Deblurring (left) and Image Denoising (right) tasks


The contributions of this paper are summarized as follows: 

1. By decomposing the SOTA methods and extracting their essential components, we form a baseline (in Figure 3c) with lower system complexity, which can exceed the previous SOTA methods and has a lower computational cost, as shown in Figure 1. It may facilitate the researchers to inspire new ideas and evaluate them conveniently. 

2. By revealing the connections between GELU, Channel Attention to Gated Linear Unit, we further simplify the baseline by removing or replacing the nonlinear activation functions (e.g. Sigmoid, ReLU, and GELU), and propose a nonlinear activation free network, namely NAFNet. It can match or surpass the baseline although being simplified. To the best of our knowledge, it is the first work demonstrates that the nonlinear activation functions may not be necessary for SOTA computer vision methods. This work may have the potential to expand the design space of SOTA computer vision methods. 

## 2 Related Works

## 2.1 Image Restoration

Image restoration tasks aim to restore a degraded image (e.g. noisy, blur) to a clean one. Recently, deep learning based methods[5,37,39,36,6,7,32,8,25] achieve SOTA results on these tasks, and most of the methods could be viewed as variants of a classical solution, UNet[29]. It stacks blocks to a U-shaped architecture with skip-connection. The variants bring performance gain, as well as the system complexity, and we broadly categorized the complexity as inter-block complexity and intra-block complexity. 

Inter-block Complexity [37,5] are multi-stage networks, i.e. the latter stage refine the results of the previous stage, and each stage is a U-shaped architecture. This design is based on the assumption that breaking down the dificult image restoration task into several subtasks contributes to performance. Diferently, [7,25] adopt the single-stage design and achieve competitive results, but they introduce complicated connections between various sized feature maps. Some methods adopt the above strategies both, e.g. [32]. Other SOTA methods, e.g. [39,36] maintain the simple structure of single-stage UNet, yet they introduce intra-block complexity, which we will discuss next. 

Intra-block Complexity There are numerous diferent intra-block design schemes, we pick a few examples here. [39] reduces the memory and time complexity of self-attention[34] by channelwise attention map rather than spatialwise. Besides, gated linear units[10] and depthwise convolution are adopted in the feed-forward network. [36] introduces window-based multi-head self-attention, which is similar to [22]. In addition, it introduces locally-enhanced feed-forward network in its block, which adds depthwise convolution to feed-forward network to enhance the local information capture ability. Diferently, we reveal that increasing system complexity is not the only way to improve performance: SOTA performance could be achieved by a simple baseline. 

## 2.2 Gated Linear Units

Gated Linear Units[10] (GLU) can be interpreted by the element-wise production of two linear transformation layers, one of which is activated with the nonlinearity. GLU or its variants has verified their efectiveness in NLP[30,10,9], and there is a prosperous trend of them in computer vision[32,39,17,20]. In this paper, we reveal the non-trivial improvement brought by GLU. Diferent from [30], we remove the nonlinear activation function in GLU without performance degradation. Furthermore, based on the fact that the nonlinear activation free GLU contains nonlinearity itself (as the product of two linear transformations raises nonlinearity), our baseline could be simplified by replacing the nonlinear activation functions with the multiplication of two feature maps. To the best of our knowledge, it is the first computer vision model achieves SOTA performance without nonlinear activation functions. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/7c5cc248f23f54441ebab6949abc361e15d2127467ecc0b4a35847bf8fef0f22.jpg)



(a) Multi-Stage Architecture


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/784582bc435859d2e61004b7a840d60bceea53c03ae45bccacc96edd4e67b0cb.jpg)



(b) Multi-Scale Fusion Architecture


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/9374f77f2692d3592cf7a7893ba288469679af3ef8362049ecb1d654afd9bdd5.jpg)



(c) UNet Architecture(ours)



Fig. 2: Comparison of architectures of image restoration models. Dashes to distinguish features of diferent sizes. (a) The multi-stage architecture[5,37] stacks UNet architecture serially. (b) The multi-scale fusion architecture[25,7] fusions the features in diferent scales. (c)UNet architecture, which is adopted by some SOTA methods[39,36]. We use it as our architecture. Some details have been deliberately omitted for simplicity, e.g. downsample/upsample layers, feature fusion modules, input/output shortcut, and etc.


## 3 Build A Simple Baseline

In this section, we build a simple baseline for image restoration tasks from scratch. To keep the structure simple, our principle is not to add entities if they are not necessary. The necessity is verified by empirical evaluation of restoration tasks. We mainly conduct experiments with the model size around 16 GMACs following HINet Simple[5], and the MACs are estimated by an input with the spatial size of $2 5 6 \times 2 5 6$ . The results of models with diferent capacities are in the experimental section. We mainly validate the results (PSNR) on two popular datasets for denoising (i.e. SIDD[1]) and deblurring (i.e. GoPro[26] dataset), based on the fact that those tasks are fundamental in low-level vision. The design choices are discussed in the following subsections. 

## 3.1 Architecture

To reduce the inter-block complexity, we adopt the classic single-stage U-shaped architecture with skip-connections, as shown in Figure 2c, following [39,36]. We believe the architecture will not be a barrier to performance. The experimental results confirmed our conjecture, in Table 6, 7 and Figure 1. 

## 3.2 A Plain Block

Neural Networks are stacked by blocks. We have determined how to stack blocks in the above (i.e. stacked in a UNet architecture), but how to design the internal structure of the block is still a problem. We start from a plain block with the most common components, i.e. convolution, ReLU, and shortcut[14], and the arrangement of these components follows [13,22], as shown in Figure 3b. We will note it as PlainNet for simplicity. Using a convolution network instead of a transformer is based on the following considerations. First, although transformers show good performance in computer vision, some works[13,23] claim that they may not be necessary for achieving SOTA results. Second, depthwise convolution is simpler than the self-attention[34] mechanism. Third, this paper is not intended to discuss the advantages and disadvantages of transformers and convolutional neural networks, but just to provide a simple baseline. The discussion of the attention mechanism is proposed in the subsequent subsection. 

## 3.3 Normalization

Normalization is widely adopted in high-level computer vision tasks, and there is also a popular trend in low-level vision. Although [26] abandoned Batch Normalization[18] as the small batch size may bring the unstable statistics[38], [5] re-introduce the Instance Normalization[33] and avoids the small batch size issue. However, [5] shows that adding instance normalization does not always bring performance gains and requires manual tuning. Diferently, under the prosperity of transformers, Layer Normalization[3] is used by more and more methods, including SOTA methods[32,39,36,23,22]. Based on these facts we conjecture Layer Normalization may be crucial to SOTA restorers, thus we add Layer Normalization to the plain block described above. This change can make training smooth, even with a 10× increase in learning rate. The larger learning rate brings significant performance gain: +0.44 dB (39.29 dB to 39.73 dB) on SIDD[1], +3.39 dB (28.51 dB to 31.90 dB) on GoPro[26] dataset. To sum up, we add Layer Normalization to the plain block as it can stabilize the training process. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/732f973f8d150477f4ffa3604abb9267322d0dc3f9aeec70324d9db3dc6e5d1b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/9a0a3453eac1e6fa863925409b6514820902a77007b3152b630c21c6b404991e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/088f9083c9bf8cf68e72e1cb0b5e21a1bf59bb0392fa597544ad2a5bb3e4ac90.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/4d493ecf1b2d8d11ecb7831b799b4f0435a4921086c87256ed6e9c3f8c6c53d3.jpg)



Fig. 3: Intra-block structure comparison. ⊗:matrix multiplication, ⊙/⊕:elementwise multiplication/addition. dconv: Depthwise convolution. Nonlinear activation functions are represented by yellow boxes. (a) Restormer’s block[39], some details are omitted for simplicity, e.g. reshaping the feature maps. (b) PlainNet’s block, which contains the most common components. (c) Our proposed baseline. Compares to (b), Channel Attention (CA) and LayerNorm are adopted. Besides, ReLU is replaced by GELU. (d) Our proposed Nonlinear Activation Free Network’s block. It replaces CA/GELU with Simplified Channel Attention(SCA) and SimpleGate respectively. The details of these components are shown in Fig 4


## 3.4 Activation

The activation function in the plain block, Rectified Linear Unit[28] (ReLU), is extensively used in computer vision. However, there is a tendency to replace ReLU with GELU[15] in SOTA methods[23,39,32,22,12]. This replacement is implemented in our model either. The performance stays comparable on SIDD (from 39.73 dB to 39.71 dB) which is consistent with the conclusion of [23], yet it brings 0.21 dB performance gain (31.90 dB to 32.11 dB) on GoPro. In short, we replace ReLU with GELU in the plain block, because it keeps the performance of image denoising while bringing non-trivial gain on image deblurring. 

## 3.5 Attention

Considering the recent popularity of the transformer in computer vision, its attention mechanism is an unavoidable topic in the design of the internal structure of the block. There are many variants of attention mechanisms, and we discuss only a few of them here. The vanilla self-attention mechanism[34], which is adopted by [12,4], generate the target feature by the linear combination of all features which are weighted by the similarity between them. Therefore, each feature contains global information, while it sufers from the quadratic computational complexity with the size of the feature map. Some image restoration tasks process data at high resolution which makes the vanilla self-attention not practical. Alternatively, [22,21,36] apply self-attention only in a fix-sized local window to alleviate the issue of increased computation. While it lacks global information. We do not take the window-based attention, as the local information could be well captured by the depthwise convolution [13,23] in the plain block. 

Diferently, [39] modifies the spatial-wise attention to channel-wise, avoids the computation issue while maintaining global information in each feature. It could be seen as a special variant of channel attention [16]. Inspired by [39], we realize the vanilla channel attention meets the requirements: computational eficiency and brings global information to the feature map. In addition, the efectiveness of channel attention has been verified in the image restoration task[37,8], thus we add the channel attention to the plain block. It brings 0.14 dB on SIDD[1] (39.71 dB to 39.85 dB), 0.24 dB on GoPro[26] dataset (32.11 dB to 32.35 dB). 

## 3.6 Summary

So far, we build a simple baseline from scratch, as we shown in Table 1. The architecture and the block are shown in Figure 2c and Figure 3c, respectively. Each component in the baseline is trivial, e.g. Layer Normalization, Convolution, GELU, and Channel Attention. But the combination of these trivial components leads to a strong baseline: it can surpass the previous SOTA results on SIDD and GoPro dataset with only a fraction of computation costs, as we shown in Figure 1 and Table 6,7. We believe the simple baseline could facilitate the researchers to evaluate their ideas. 

## 4 Nonlinear Activation Free Network

The baseline described above is simple and competitive, but is it possible to further improve performance while ensuring simplicity? Can it be simpler without performance loss? We try to answer these questions by looking for commonalities from some SOTA methods[32,39,20,17]. We find that in these methods, Gated Linear Units[10](GLU) are adopted. It implies that GLU might be promising. We will discuss it next. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/28d9ae378e613bc2118ec056c3ad6ffb32c676abd0c75117ba98a3742d3768df.jpg)



Fig. 4: Illustration of (a) Channel Attention[16] (CA), (b) Simplified Channel Attention (SCA), and (c) Simple Gate (SG). ⊙/∗: element-wise/channel-wise multiplication


Gated Linear Units The gated linear units could be formulated as: 

$$
G a t e (\mathbf {X}, f, g, \sigma) = f (\mathbf {X}) \odot \sigma (g (\mathbf {X})),\tag{1}
$$

where X represents the feature map, f and g are linear transformers, σ is a non-linear activation function, e.g. Sigmoid, and ⊙ indicates element-wise multiplication. As discussed above, adding GLU to our baseline may improve the performance yet the intra-block complexity is increasing as well. This is not what we expected. To address this, we revisit the activation function in the baseline, i.e. GELU[15]: 

$$
G E L U (x) = x \Phi (x),\tag{2}
$$

where Φ indicates the cumulative distribution function of the standard normal distribution. And based on [15], GELU could be approximated and implemented by: 

$$
0. 5 x (1 + t a n h [ \sqrt {2 / \pi} (x + 0. 0 4 4 7 1 5 x ^ {3}) ]).\tag{3}
$$

From Eqn. 1 and Eqn. 2, it can be noticed that GELU is a special case of GLU, i.e. f, g are identity functions and take σ as Φ. Through the similarity, we conjecture from another perspective that GLU may be regarded as a generalization of activation functions, and it might be able to replace the nonlinear activation functions. Further, we note that the GLU itself contains nonlinearity and does not depend on σ: even if the σ is removed, $G a t e ( \mathbf { X } ) = f ( \mathbf { X } ) \odot g ( \mathbf { X } )$ contains nonlinearity. Based on these, we propose a simple GLU variant: directly divide the feature map into two parts in the channel dimension and multiply them, as we shown in Figure 4c, noted as SimpleGate. Compared to the complicated implementation of GELU in Eqn.3, our SimpleGate could be implemented by an element-wise multiplication, that’s all: 

$$
S i m p l e G a t e (\mathbf {X}, \mathbf {Y}) = \mathbf {X} \odot \mathbf {Y},\tag{4}
$$

where X and Y are feature maps of the same size. 

By replacing GELU in the baseline to the proposed SimpleGate, the performance of image denoising (on SIDD[1]) and image deblurring (on GoPro[26] dataset) boost 0.08 dB (39.85 dB to 39.93 dB) and 0.41 dB (32.35 dB to 32.76 dB) respectively. The results demonstrate that GELU could be replaced by our proposed SimpleGate. At this point, only a few types of nonlinear activations left in the network: Sigmoid and ReLU in the channel attention module[16], and we will discuss the simplifications of it next. 

Simplified Channel Attention In Section 3, we adopt the channel attention[16] into our block as it captures the global information and it is computationally eficient. It is illustrated in Figure 4a: it squeezes the spatial information into channels first and then a multilayer perceptual applies to it to calculate the channel attention, which will be used to weight the feature map. It could be represented as: 

$$
C A (\mathbf {X}) = \mathbf {X} * \sigma (W _ {2} m a x (0, W _ {1} p o o l (\mathbf {X}))),\tag{5}
$$

where X represents the feature map, pool indicates the global average pooling operation which aggregates the spatial information into channels. σ is a nonlinear activation function, Sigmoid, $W _ { 1 } , W _ { 2 }$ are fully-connected layers and ReLU is adopted between two fully-connected layers. Last, ∗ is a channelwise product operation. If we regard the channel-attention calculation as a function, noted as Ψ with input X , Eqn. 5 could be re-writed as: 

$$
C A (\mathbf {X}) = \mathbf {X} * \Psi (\mathbf {X}).\tag{6}
$$

It can be noticed that Eqn. 6 is very similar to Eqn. 1. This inspires us to consider channel attention as a special case of GLU, which can be simplified like GLU in the previous subsection. By retaining the two most important roles of channel attention, that is, aggregating global information and channel information interaction, we propose the Simplified Channel Attention: 

$$
S C A (\mathbf {X}) = \mathbf {X} * W p o o l (\mathbf {X}).\tag{7}
$$

The notations follows Eqn. 5. Apparently, Simplified Channel Attention (Eqn. 7) is simpler than the original one (Eqn. 5), as shown in Figure 4a and Figure 4b. Although it is simpler, there is no loss of performance: +0.03 dB (39.93 dB to 39.96 dB) on SIDD and +0.09 dB (32.76 dB to 32.85 dB) on GoPro. 

Summary Starting from the baseline proposed in Section 3, we further simplify it by replacing the GELU with SimpleGate and Channel Attention to Simplified Channel Attention, without loss of performance. We emphasize that after the simplification, there are no nonlinear activation functions (e.g. ReLU, GELU, Sigmoid, etc.) in the network. So we call this baseline Nonlinear Activation Free Network, namely NAFNet. It can match or surpass the baseline although without nonlinear activation functions, as we shown in Figure 1 and Table 6,7. We can now answer the questions in the begining of this section by yes, because of the simplicity and efectiveness of NAFNet. 

## 5 Experiments

In this section, we analyze the efect of the design choices of NAFNet described in previous sections in detail. Next, we apply our proposed NAFNet to various image restoration applications, including RGB image denoising, image deblurring, raw image denoising, and image deblurring with JPEG artifacts. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/d24b7199bcdc405deb08e4dad34affe763c3da945e0cdb98adcdfaf8af78f846.jpg)



Fig. 5: Qualitative comparison of image denoising methods on SIDD[1]


## 5.1 Ablations

The ablation studys are conducted on image denoising (SIDD[1]) and deblurring (GoPro[26]) tasks. We follow experiments setting of [5] if not specified, e.g. 16 GMACs of computational budget, gradient clip, and PSNR loss. We train models with Adam[19] optimizer $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9$ , weight decay 0) for total 200K iterations with the initial learning rate $1 e ^ { - 3 }$ gradually reduced to $1 e ^ { - 6 }$ with the cosine annealing schedule[24]. The training patch size is $2 5 6 \times 2 5 6$ and batch size is 32. Training by patches and testing by the full image raises performance degradation[8], we solve it by adopting TLC[8] following MPRNet-local[8]. The efectiveness of TLC on GoPro<sup>1</sup> is shown in Tab 4. We mainly compare TLC with “test by patches” strategy, which is adopted by [5], [25], and etc. It brings performance gains and avoids the artifacts brought by patches. Moreover, we apply skip-init[11] to stabilize training following [23]. The default width and number of blocks are 32 and 36, respectively. We adjust the width to keep the computational budget hold if the number of blocks changed. We report Peak Signal to Noise Ratio (PSNR) and Structural SIMilarity (SSIM) in our experiments. The speed/memory/computational complexity evaluation is conducted with an input size of $2 5 6 \times 2 5 6$ , on an NVIDIA 2080Ti GPU. 

From PlainNet to the simple baseline: PlainNet is defined in Section 3, and its block is illustrated in Figure 3b. We find that the training of PlainNet is unstable under the default settings. As an alternative, we reduce the learning rate (lr) by a factor of 10 to make the model trainable. This issue is solved by introducing Layer Normalization (LN): the learning rate can be increased from $1 e ^ { - 4 }$ to $1 e ^ { - 3 }$ with a more stable training process. In PSNR, LN brings 0.46 dB and 3.39 dB on SIDD and GoPro respectively. Besides, GELU and Channel Attention (CA) also demonstrated their efectiveness in Table 1. 

From the simple baseline to NAFNet: As described in Section 3, NAFNet can be obtained by simplifying the baseline. In Tab 2, we show that there is no performance penalty for this simplification. Instead, the PSNR boosts 0.11 dB and 0.50 dB in SIDD and GoPro respectively. The computational complexity is consistent for a fair comparison, and details in the supplementary material. The speedup of modifications compared to Baseline is provided. In addition, no significant extra memory consumption compares to Baseline in inference. 


Table 1: Build a simple baseline from PlainNet. The efectiveness of Layer Normalization (LN), GELU, and Channel Attention (CA) have been verified. ∗ indicates that the training is unstable due to the large learning rate (lr)


<table><tr><td rowspan="2"></td><td rowspan="2">lr</td><td rowspan="2">LN</td><td rowspan="2">ReLU→GELU</td><td rowspan="2">CA</td><td colspan="2">SIDD</td><td colspan="2">GoPro</td></tr><tr><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td></tr><tr><td>PlainNet</td><td><eq>1e^{-4}</eq></td><td></td><td></td><td></td><td>39.29</td><td>0.956</td><td>28.51</td><td>0.907</td></tr><tr><td>PlainNet*</td><td><eq>1e^{-3}</eq></td><td></td><td></td><td></td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td></td><td><eq>1e^{-3}</eq></td><td>√</td><td></td><td></td><td>39.73</td><td>0.959</td><td>31.90</td><td>0.952</td></tr><tr><td></td><td><eq>1e^{-3}</eq></td><td>√</td><td>√</td><td></td><td>39.71</td><td>0.958</td><td>32.11</td><td>0.954</td></tr><tr><td>Baseline</td><td><eq>1e^{-3}</eq></td><td>√</td><td>√</td><td>√</td><td>39.85</td><td>0.959</td><td>32.35</td><td>0.956</td></tr></table>


Table 2: NAFNet is derived from the simplification of baseline, i.e. replacing GELU to SimpleGate (SG), and replacing Channel Attention (CA) to Simplified Channel Attention (SCA).


<table><tr><td></td><td>GELU→SG</td><td>CA→SCA</td><td colspan="2">SIDDPSNR SSIM</td><td colspan="2">GoProPSNR SSIM</td><td>speedup</td></tr><tr><td rowspan="3">Baseline</td><td rowspan="3">√</td><td></td><td>39.85</td><td>0.959</td><td>32.35</td><td>0.956</td><td>1.00×</td></tr><tr><td></td><td>39.93</td><td>0.960</td><td>32.76</td><td>0.960</td><td>0.98×</td></tr><tr><td>√</td><td>39.95</td><td>0.960</td><td>32.54</td><td>0.958</td><td>1.11×</td></tr><tr><td>NAFNet</td><td>√</td><td>√</td><td>39.96</td><td>0.960</td><td>32.85</td><td>0.960</td><td>1.09×</td></tr></table>

Number of blocks: We verify the efect of the number of blocks on NAFNet in Table 3. We mainly consider the latency at spatial size 720 × 1280, as this is the size of the entire GoPro image. In the process of increasing the number of blocks to 36, the performance of the model has been greatly improved, and the latency has not increased significantly (+14.5% compares to 9 blocks). When the number of blocks further increases to 72, the performance improvement of the model is not obvious, but the latency increases significantly (+30.0% compares to 36 blocks). Because 36 blocks can achieve a better performance/latency balance, we use it as the default option. 

Variants of σ in SimpleGate: Vanilla gated linear unit (GLU) contains a nonlinear activation function σ as formulated in Eqn. 1. Our proposed SimpleGate, as shown in Eqn. 4 and Figure 4c removes it. In other words, σ in SimpleGate is set as an identity function. We variants the σ from the identity function to diferent nonlinear activation functions in Table 5 to judge the importance of nonlinearity in σ. PSNR on SIDD is basically unafected (fluctuates from 39.96 dB to 39.99 dB), while PSNR on GoPro drops significantly (-0.11 dB to -0.35 dB), which indicates that in NAFNet, the σ in SimpleGate may not be necessary. 


Table 3: The efect of the number of blocks. The width is adjusted to keep the computational budget hold. Latency-256 and Latency-720 is based on the input size 256 × 256 and $7 2 0 \times 1 2 8 0$ respectively, in milliseconds


<table><tr><td rowspan="2"></td><td rowspan="2"># of blocks</td><td colspan="2">SIDD</td><td colspan="2">GoPro</td><td rowspan="2">Latency-256</td><td rowspan="2">Latency-720</td></tr><tr><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td></tr><tr><td rowspan="4">NAFNet</td><td>9</td><td>39.78</td><td>0.959</td><td>31.79</td><td>0.951</td><td>11.8</td><td>154.7</td></tr><tr><td>18</td><td>39.90</td><td>0.960</td><td>32.64</td><td>0.951</td><td>19.9</td><td>151.7</td></tr><tr><td>36</td><td>39.96</td><td>0.960</td><td>32.85</td><td>0.959</td><td>39.1</td><td>177.1</td></tr><tr><td>72</td><td>39.95</td><td>0.960</td><td>32.88</td><td>0.961</td><td>73.8</td><td>230.1</td></tr></table>


Table 4: Efectiveness of TLC[8] on GoPro[26]



Table 5: Variants of σ in SimpleGate(X, Y) = X ⊙ σ(Y)


<table><tr><td></td><td>patches?</td><td>TLC?</td><td>PSNR</td><td>SSIM</td></tr><tr><td rowspan="3">NAFNet</td><td rowspan="3">√</td><td></td><td>33.08</td><td>0.963</td></tr><tr><td></td><td>33.65</td><td>0.966</td></tr><tr><td>√</td><td>33.69</td><td>0.967</td></tr></table>

<table><tr><td rowspan="2">σ</td><td colspan="2">SIDD</td><td colspan="2">GoPro</td></tr><tr><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td></tr><tr><td>Identity(ours)</td><td>39.96</td><td>0.960</td><td>32.85</td><td>0.960</td></tr><tr><td>ReLU</td><td>39.98</td><td>0.960</td><td>32.59</td><td>0.958</td></tr><tr><td>GELU</td><td>39.97</td><td>0.960</td><td>32.72</td><td>0.959</td></tr><tr><td>Sigmoid</td><td>39.99</td><td>0.960</td><td>32.50</td><td>0.958</td></tr><tr><td>SiLU</td><td>39.96</td><td>0.960</td><td>32.74</td><td>0.960</td></tr></table>

## 5.2 Applications

We apply NAFNet to various image restoration tasks, follow the training settings of ablation study if not specified, except that it is enlarged by increasing the width from 32 to 64. Besides, batch size and total training iterations are 64 and 400K respectively, following [5]. Random crop augmentation is applied. We report the mean of three experimental results. The baseline is enlarged to achieve better results, details in the appendix. 

RGB Image Denoising We compare the RGB Image Denoising results with other SOTA methods on SIDD, show in Table 6. Baseline and its simplified version NAFNet, exceed the previous best result Restormer 0.28 dB with only a fraction of its computational cost, as shown in Figure 1. The qualitative results are shown in Figure 5. Our proposed baselines can restore more fine details compared to other methods. Moreover, we achieve SOTA result (40.15 dB) on the online benchmark, exceed previous top-ranked methods 0.23 dB. 

Image Deblurring We compare the deblurring results of SOTA methods on GoPro[26] dataset, flip and rotate augmentations are adopted. As we shown in Table 7 and Figure 1, our baseline and NAFNet surpass the previous best method MPRNet-local[8] 0.09 dB and 0.38 dB in PSNR, respectively, with only 8.4% of its computational costs. The visualization results are shown in Figure 6, our baselines can restore sharper results compares to other methods. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/7c9abb1a7f9d8aa033e1d8054fe628ba3d7f07658b8feec7e6f5622b28aedcbe.jpg)



Fig. 6: Qualitative comparison of image deblurring methods on GoPro[26]



Table 6: Image Denoising Results on SIDD[1]


<table><tr><td>Method</td><td>MPRNet [37]</td><td>MIRNet [40]</td><td>NBNet [6]</td><td>UFormer [36]</td><td>MAXIM [32]</td><td>HINet [5]</td><td>Restormer [39]</td><td>Baseline ours</td><td>NAFNet ours</td></tr><tr><td>PSNR</td><td>39.71</td><td>39.72</td><td>39.75</td><td>39.89</td><td>39.96</td><td>39.99</td><td>40.02</td><td>40.30</td><td>40.30</td></tr><tr><td>SSIM</td><td>0.958</td><td>0.959</td><td>0.959</td><td>0.960</td><td>0.960</td><td>0.958</td><td>0.960</td><td>0.962</td><td>0.962</td></tr><tr><td>MACs(G)</td><td>588</td><td>786</td><td>88.8</td><td>89.5</td><td>169.5</td><td>170.7</td><td>140</td><td>84</td><td>65</td></tr></table>

Raw Image Denoising We apply NAFNet to a raw image denoising task. The training and testing settings follow PMRID[35], and we noted the testing set as 4Scenes (as the dataset contains 39 raw images of 4 diferent scenes in various light conditions) for simplicity. In addition, we make fair comparison by changing the width and number of blocks of NAFNet from 32 to 16, 36 to 7, respectively, so that the computational cost is less than PMRID. The results shown in Table 8 and Figure 7 demonstrate NAFNet can surpass PMRID quantitatively and qualitatively. In addition, this experiment indicates our NAFNet can be scaled flexibly (from 1.1 GMACs to 65 GMACs). 

Image Deblurring with JPEG artifacts We conduct experiments on REDS[27] dataset, the training setting follows [5,32], and we evaluate the result on 300 images from the validation set (noted as REDS-val-300) following [5,32]. As shown in Table 9, our method outperforms other competing methods, including the 


Table 7: Image Deblurring Results on GoPro[26]


<table><tr><td>Method</td><td>MIMO-UNet [7]</td><td>HINet [5]</td><td>MAXIM [32]</td><td>Restormer [39]</td><td>UFormer [36]</td><td>DeepRFT [25]</td><td>MPRNet -local[8]</td><td>Baseline ours</td><td>NAFNet ours</td></tr><tr><td>PSNR</td><td>32.68</td><td>32.71</td><td>32.86</td><td>32.92</td><td>32.97</td><td>33.23</td><td>33.31</td><td>33.40</td><td>33.69</td></tr><tr><td>SSIM</td><td>0.959</td><td>0.959</td><td>0.961</td><td>0.961</td><td>0.967</td><td>0.963</td><td>0.964</td><td>0.965</td><td>0.967</td></tr><tr><td>MACs(G)</td><td>1235</td><td>170.7</td><td>169.5</td><td>140</td><td>89.5</td><td>187</td><td>778.2</td><td>84</td><td>65</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/0b75f2992d4ac7c0045f2f3b07fab9cc09610b39f54c05ccad1997d81537ab55.jpg)



Fig. 7: Qualitatively compare the noise reduction efects of PMRID[35] and our porposed NAFNet. Zoom in to see details



Table 8: Raw image denoising results on 4Scenes[35]



Table 9: Image deblurring results on REDS-val-300[27]


<table><tr><td>Method</td><td>PSNR</td><td>SSIM</td><td>MACs(G)</td></tr><tr><td>PMRID[35]</td><td>39.76</td><td>0.975</td><td>1.2</td></tr><tr><td>NAFNet(ours)</td><td>40.05</td><td>0.977</td><td>1.1</td></tr></table>

<table><tr><td>Method</td><td>PSNR</td><td>SSIM</td><td>MACs(G)</td></tr><tr><td>MPRNet[37]</td><td>28.79</td><td>0.811</td><td>776.7</td></tr><tr><td>HINet[5]</td><td>28.83</td><td>0.862</td><td>170.7</td></tr><tr><td>MAXIM[32]</td><td>28.93</td><td>0.865</td><td>169.5</td></tr><tr><td>NAFNet(ours)</td><td>29.09</td><td>0.867</td><td>65</td></tr></table>

previous winning solution (HINet) on the REDS dataset of NTIRE 2021 Image Deblurring Challenge Track2 JPEG artifacts[27]. 

## 6 Conclusions

By decomposing the SOTA methods, we extract the essential components and adopt them on a naive PlainNet. The obtained baseline reaches SOTA performance on image denoising and image deblurring tasks. By analyzing the baseline, we reveal that it can be further simplified: The nonlinear activation functions in it can be completely replaced or removed. From this, we propose a nonlinear activation free network, NAFNet. Although simplified, its performance is equal to or better than baseline. Our proposed baselines may facilitate the researchers to evaluate their ideas. In addition, this work has the potential to influence future computer vision model design, as we demonstrate that nonlinear activation functions are not necessary to achieve SOTA performance. 

Acknowledgements: This research was supported by National Key R&D Program of China (No. 2017YFA0700800) and Beijing Academy of Artificial Intelligence (BAAI). 

## Appendix

## A Other Details

## A.1 Inverted Bottleneck

Following [23] we adopt inverted bottleneck design in the baseline and NAFNet. We first discuss the setting of the ablation studies. In the baseline, the channel width within the first skip connection is always consistent with the input, its computational cost could be approximated by: 

$$
H \times W \times c \times c + H \times W \times c \times k \times k + H \times W \times c \times c,\tag{1}
$$

where H, W represent the spatial size of the feature map, c indicates the input dimension, and k is the kernel size of the depthwise convolution (3 in our experiments). In practice, $c \gg k \times k$ , thus Eqn. $( 1 ) \approx 2 \times H \times W \times c \times c$ . The hidden dimension within the second skip connection is twice the input dimension, its computational cost is: 

$$
H \times W \times c \times 2 c + H \times W \times 2 c \times c,\tag{2}
$$

notations following Eqn. (1). As a result, the overall computational cost of one baseline block $\approx 6 \times H \times W \times c \times c .$ 

As for NAFNet’s block, the SimpleGate module shrinks the channel width by half. We double the hidden dimension in the first skip connection, and its computational cost could be approximated by: 

$$
H \times W \times c \times 2 c + H \times W \times 2 c \times k \times k + H \times W \times c \times c,\tag{3}
$$

notations following Eqn. (1). And the hidden dimension in the second skip connection follows baseline. Its computational cost is: 

$$
H \times W \times c \times 2 c + H \times W \times c \times c.\tag{4}
$$

As a result, the overall computational cost of one NAFNet’s block $\approx 6 \times H \times$ $W \times c \times c ,$ , which is consistent with the baseline’s block. The advantage of this is that the baseline and NAFNet can share hyperparameters, such as the number of blocks, learning rate, etc. 

As for the applications, the hidden dimension of the baseline’s first skip connection is expanded to achieve better results. In addition, it should be noted that the above discussion omits the computation of some modules, e.g. layer normalization, GELU, channel attention, and etc., as their computational cost is negligible compared to convolution. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/90fd183aa3d6c0213fb601078b65137f4d788af3b50b9df567a9be32097ff3c4.jpg)



Fig. 1: Additional qualitatively comparison of raw image denoising results with PMRID[35]. Zoom in to see details


## A.2 Channel Attention and Simplified Channel Attention

For a feature map with width of $c ,$ the channel attention module shrinks it by a factor of r and then project it back into c (by fully-connect layer). The computational cost could be approximated by $c \times c / r + c / r \times c .$ As to the simplified channel attention module, its computational cost is $c \times c .$ . For a fair comparison, we choose $r = 2$ so that their computational costs are consistent in our experiments. 

## A.3 Feature Fusion

There are skip connections from the encoder block to the decoder block, and there are several ways to fuse the features of encoder/decoder. In [5], the encoder features are transformed by a convolution and then concatenate with the decoder features. In [39], features are concatenated first and then transformed by a convolution. Diferently, we simply element-wise add the encoder and decoder features as the feature fusion approach. 

## A.4 Downsample/Upsample Layer

For the downsample layer, we use the convolution with a kernel size of 2 and a stride of 2. This design choice is inspired by [2]. For the upsample layer, we double the channel width by a pointwise convolution first, and then follows a pixel shufle module[31]. 

## B More Visualization Results

We provide additional visualization results of raw image denoising, image deblurring, RGB image denoising tasks, as we shown in Figure 1, 2, and 3. Our baselines can restore more fine details compare to other methods. It is recommended to zoom in to compare the details in the red box. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/e978049fe8005665516120d741669ecbe51f86f97b22ec544c2146b9a3d8e48a.jpg)



PSNR Reference


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/5dfe7dd41c350a86bfb9fd07b2fc0d5443b7ed68541a20e77cf2cfe4f93afcfb.jpg)



19.45 dB Blurry


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/a0299f0928c6013ba02da3cecb5301851ac66ddf871433d48d7a9bc04f40c208.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/794b0cbd1e4e5438fd5a5b50d84016868fa49cd65246a1893b8bd96b5d5e9508.jpg)



25.81 dB HINet[5]


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/b722b42fc8d0942820d5928551d2f507350cf04a464d78f23b20df71e4f08a48.jpg)



27.03 dB MPRNet-local[8]


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/d200f8994741aa10f6cb5b5742d05c3a2142b715773b1bbf0d2466c877ffea1a.jpg)



26.96 dB Restormer[39]


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/8bc7be9f5348d6e8077a07e381654a8431a058f409c5db9ad05f10f057619955.jpg)



28.11 dB Baseline(ours)



25.67 dB MPRNet[37]


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/c833119e823bbd5d3126ea108136deba07b534ce94c44213a1a38a3441281779.jpg)



28.71 dB NAFNet(ours)



Fig. 2: Additional qualitative comparison of image deblurring methods


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/fb08819e-870d-4ea1-a780-10f4fb3e7c7a/f2961b0029dff79e0a1bf9e2ebf574badb445b4bfe1c65f92bbaae360e5bcdfa.jpg)



Fig. 3: Additional qualitative comparison of image denoising methods


## References



1. Abdelhamed, A., Lin, S., Brown, M.S.: A high-quality denoising dataset for smartphone cameras. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2018) 





2. Alsallakh, B., Kokhlikyan, N., Miglani, V., Yuan, J., Reblitz-Richardson, O.: Mind the pad–cnns can develop blind spots. arXiv preprint arXiv:2010.02178 (2020) 





3. Ba, J.L., Kiros, J.R., Hinton, G.E.: Layer normalization. arXiv preprint arXiv:1607.06450 (2016) 





4. Chen, H., Wang, Y., Guo, T., Xu, C., Deng, Y., Liu, Z., Ma, S., Xu, C., Xu, C., Gao, W.: Pre-trained image processing transformer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12299–12310 (2021) 





5. Chen, L., Lu, X., Zhang, J., Chu, X., Chen, C.: Hinet: Half instance normalization network for image restoration. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 182–192 (2021) 





6. Cheng, S., Wang, Y., Huang, H., Liu, D., Fan, H., Liu, S.: Nbnet: Noise basis learning for image denoising with subspace projection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4896– 4906 (2021) 





7. Cho, S.J., Ji, S.W., Hong, J.P., Jung, S.W., Ko, S.J.: Rethinking coarse-to-fine approach in single image deblurring. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4641–4650 (2021) 





8. Chu, X., Chen, L., , Chen, C., Lu, X.: Improving image restoration by revisiting global information aggregation. arXiv preprint arXiv:2112.04491 (2021) 





9. Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q.V., Salakhutdinov, R.: Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860 (2019) 





10. Dauphin, Y.N., Fan, A., Auli, M., Grangier, D.: Language modeling with gated convolutional networks. In: International conference on machine learning. pp. 933– 941. PMLR (2017) 





11. De, S., Smith, S.: Batch normalization biases residual blocks towards the identity function in deep networks. Advances in Neural Information Processing Systems 33, 19964–19975 (2020) 





12. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020) 





13. Han, Q., Fan, Z., Dai, Q., Sun, L., Cheng, M.M., Liu, J., Wang, J.: Demystifying local vision transformer: Sparse connectivity, weight sharing, and dynamic weight. arXiv preprint arXiv:2106.04263 (2021) 





14. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016) 





15. Hendrycks, D., Gimpel, K.: Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415 (2016) 





16. Hu, J., Shen, L., Sun, G.: Squeeze-and-excitation networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 7132–7141 (2018) 





17. Hua, W., Dai, Z., Liu, H., Le, Q.V.: Transformer quality in linear time. arXiv preprint arXiv:2202.10447 (2022) 





18. Iofe, S., Szegedy, C.: Batch normalization: Accelerating deep network training by reducing internal covariate shift. In: International conference on machine learning. pp. 448–456. PMLR (2015) 





19. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014) 





20. Liang, J., Cao, J., Fan, Y., Zhang, K., Ranjan, R., Li, Y., Timofte, R., Van Gool, L.: Vrt: A video restoration transformer. arXiv preprint arXiv:2201.12288 (2022) 





21. Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., Timofte, R.: Swinir: Image restoration using swin transformer. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 1833–1844 (2021) 





22. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 10012–10022 (2021) 





23. Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S.: A convnet for the 2020s. arXiv preprint arXiv:2201.03545 (2022) 





24. Loshchilov, I., Hutter, F.: Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983 (2016) 





25. Mao, X., Liu, Y., Shen, W., Li, Q., Wang, Y.: Deep residual fourier transformation for single image deblurring. arXiv preprint arXiv:2111.11745 (2021) 





26. Nah, S., Hyun Kim, T., Mu Lee, K.: Deep multi-scale convolutional neural network for dynamic scene deblurring. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 3883–3891 (2017) 





27. Nah, S., Son, S., Lee, S., Timofte, R., Lee, K.M.: Ntire 2021 challenge on image deblurring. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 149–165 (2021) 





28. Nair, V., Hinton, G.E.: Rectified linear units improve restricted boltzmann machines. In: Icml (2010) 





29. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical image computing and computer-assisted intervention. pp. 234–241. Springer (2015) 





30. Shazeer, N.: Glu variants improve transformer. arXiv preprint arXiv:2002.05202 (2020) 





31. Shi, W., Caballero, J., Husz´ar, F., Totz, J., Aitken, A.P., Bishop, R., Rueckert, D., Wang, Z.: Real-time single image and video super-resolution using an eficient sub-pixel convolutional neural network. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1874–1883 (2016) 





32. Tu, Z., Talebi, H., Zhang, H., Yang, F., Milanfar, P., Bovik, A., Li, Y.: Maxim: Multi-axis mlp for image processing. arXiv preprint arXiv:2201.02973 (2022) 





33. Ulyanov, D., Vedaldi, A., Lempitsky, V.: Instance normalization: The missing ingredient for fast stylization. arXiv preprint arXiv:1607.08022 (2016) 





34. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017) 





35. Wang, Y., Huang, H., Xu, Q., Liu, J., Liu, Y., Wang, J.: Practical deep raw image denoising on mobile devices. In: European Conference on Computer Vision. pp. 1–16. Springer (2020) 





36. Wang, Z., Cun, X., Bao, J., Liu, J.: Uformer: A general u-shaped transformer for image restoration. arXiv preprint arXiv:2106.03106 (2021) 





37. Waqas Zamir, S., Arora, A., Khan, S., Hayat, M., Shahbaz Khan, F., Yang, M.H., Shao, L.: Multi-stage progressive image restoration. arXiv e-prints pp. arXiv–2102 (2021) 





38. Yan, J., Wan, R., Zhang, X., Zhang, W., Wei, Y., Sun, J.: Towards stabilizing batch statistics in backward propagation of batch normalization. arXiv preprint arXiv:2001.06838 (2020) 





39. Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.H.: Restormer: Eficient transformer for high-resolution image restoration. arXiv preprint arXiv:2111.09881 (2021) 





40. Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.H., Shao, L.: Learning enriched features for real image restoration and enhancement. In: European Conference on Computer Vision. pp. 492–511. Springer (2020) 

