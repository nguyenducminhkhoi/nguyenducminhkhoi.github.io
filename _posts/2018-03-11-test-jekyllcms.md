---
title: Test Jekyllcms
---
## Introduction

CNNs can work so well in image recognition task but prone to have error in adversarial tests because they use Pooling-layers. That is, to detect a object, its object parts are important but the relative position as well as the pose of each part is not important. To overcome this limitation, recently papers on Capsule Networks propose the idea of routing by agreement instead of using max-pooling. The whole Capsule Network represents the part-whole relationships. If multiple parts all agree for the existence of object category, that object category will have high activation. Capsules in lower layer encapsulate the low features like shapes or patterns, while capsules in higher layer encapsulate the high features like object parts (e.g eyes, mount, nose). Capsules represent that information in a pose matrix $4 \times 4$ and activation score (how confident of object it represent exist). 

---

## Drawbacks of current Capsule Networks

Recent work on Capsule Networks suppose that we have same fixed level of abstraction for multiple object categories. However it is not correct in the real world that some objects need higher order of abstract than others. For example the levels of abstraction of car and human are different while car just needs 2 or 3 levels of abstraction: shapes (e.g. circle, rectangle) to car parts (e.g. wheel, mirror, head-light), human needs higher level of abstraction to recognize: shapes to small body parts (e.g. toes, hair, mount, eyes) to large body parts (face, left hand, right hand). Therefore, with some simple datasets without noisy background and objects are supposed to have the same level of abstraction like MNIST, Capsule Network can work very well and get the state-of-the-art results. However when moving to more real datasets such as SmallNORB, the performance decreases significantly because the difference between level of abstraction become large between object categories (e.g. human, car, animal, airplane). It almost cannot apply to real image dataset such as ImageNet where the levels of abstraction of different object categories vary significantly. If we use the same level of abstraction for different categories, some object categories will cumulatively absorb the surrounding background to their own poses. If the surrounding background is perfect without any noise (like SmallNORB or MNIST) it does not matter much however of background is noisy they will greatly affect the recognization performance. Therefore, we need to know when to stop grouping parts together to form an object. 

Another drawback of Capsule Networks is that the templates for each object-part relationship are fixed. That is suitable for some kind of objects that the relative positions between object parts never change like car, bicycle, chair. However with object like human the relative positions between object parts vary differently depending the current pose. Consequently, if we apply Capsule Networks to detect human in the image is will give low activation for nonstandard human poses. Therefore we need to apply more flexible mechanism when routing by agreement.

Recent work on Capsule Networks assumes that each capsule will learn a discriminating feature representing an object part. However, there is no constraint on this feature should be a sematically meaningful object parts. Thus, one capsule can learn object parts like the left example. Therefore, we need some constraints on the capsule to learn semantically meaningful object parts and each capsule will learn separate object parts like the right example
![IMAGE](quiver-image-url/600F5BF32E4C224296F411E43D281A23.jpg =183x168) ![IMAGE](quiver-image-url/7F69309C65E1411040ED8DCA46B9CB24.jpg =187x161)

The design of Capsule Networks promise to give more interpretable power than CNNs. That is we can understand what is object parts each capsule representing by using its pose matrix and know exactly the contribution of each capsule to the prediction of object by using routing coefficient (which is different depending on each instance of the same object category). In other word, Capsule Networks give us a parse tree of each object parts and their components with contributions. We cannot that kind of interpretability with traditional CNNs. To visuallize an object part from its pose matrix, we need to use a decoder. However, the pose matrix of each object part is in different space (i.e. different basis) we should use different decoders to decode, that make the learning more difficult due to the increasing of parameter to learn. Therefore, we need more constraints on the pose matrix of each object part to have the same basis, thus we can visualize each object part by just using one decoder learned from reconstructing the object pose matrix layer with input image.

---

## Suggested Improvements
Here we suggest 3 major improvements:
* In Capsule Architecture for different levels of abstraction
* In Routing algorithms for deformable parts
* Adding more constraints to capsule pose matrices for learning meaningful pose matrix

### Improvement in architecture:
The architecture of Capsule Network as follow:

![IMAGE](quiver-image-url/A721B5569D8A66C3AE484E105A2740F3.jpg =915x101)

We can add ClassCaps right after each PrimaryCaps and ConvCaps to get prediction result right after each Capsule layer. If one object category has enough object parts, it can output prediction directly, it does not need to wait until other object parts merging, this will prevent capsule in higher layer cumulatively absorbing surrounding background. There are two options. 

The first option is to get final ClassCaps based on multiple ClassCaps of each layers. We make assumption here is the object when completed merging its object parts will give highest activation. If it absorbs surrounding background, the activation for this object will decrease. We can implement it by this architecture where ClassCaps is max over each capsule types. ClassCaps1, ClassCaps2 and ClassCaps3 share their weights

![IMAGE](quiver-image-url/7BD59944D10737E02A0DB2477C8105E8.jpg =946x387)

The second option is to have multiple losses for each ClassCaps, the total loss is sum of the losses. With this option, we assume that each capsule will receive feedback directly from the loss so the object capsule at lower layer will stop sending information to higher layer capsule. Each ClassCaps will have different weights.

![IMAGE](quiver-image-url/474CD3748E55A33EF7F83E64817694FE.jpg =911x376)

---

### Improvement in routing algorithm
If we use angle routing (proposed in Dynamic Capsule Paper), we have the 
following algorithm, capsule $i$ in layer $L$ representing parts and capsule $j$ 
in layer $L+1$ representing objects 
**Procedure** angle_routing($a, p$):
> **input**: $a, p$: activations and poses of capsules in layer $L$; 
> **learnable**: $W$: tranformation matrix: part-whole relationship  
                 $\beta_a, \beta_v$: learnable discriminative value
> **output**: $a' \text{ and } \mu$: activations and poses of capsule in layer $L+1$
> 
>  $V = p * W$ # votes or predictions of capsule $i$ in layer $L$
>  $R_{ij} = 0$ # intialize routing coefficient for all capsules $i$ in layer $L$ 
and capsules $j$ in layer $L+1$
>
>  **for** t iterations **do**:
> 1. $\forall i \in \Omega_{L}: R_{ij}' = \frac{e^{R_{ij}}a_i}{\sum_k{e^{R_{ik}}}}  $ # weighted capsule activations of layer $L$ by routing coeff
> 2. $\forall j \in \Omega_{L+1}: \mu_j = \frac{\sum_i{R_{ij}V_{ij}}}{\sum_i{R_{ij}}}$ # weighted mean
> 3. $\forall j \in \Omega_{L+1}: \sigma_j^2 = \frac{\sum_i{R_{ij} (V_{ij} - \mu_j)^2}}{\sum_i{R_{ij}}}$ # weighted variance
> 4. $\forall j \in \Omega_{L+1}, const_j = (\beta_v + \log{\sigma_{j}^2})\sum_i{R_{ij}}$ 
> 5. $\forall j \in \Omega_{L+1}: a_j' = sigmoid(\lambda(\beta_a - \sum_h{const_j^h}))$ # if $\sigma^2$ is large -> activation is small,  $h$ is channel
> 6. $\forall i \in \Omega_{L}, \forall j \in \Omega_{L+1}:u_{ij} = \frac{\mu_{j} \cdot V_{ij}}{||\mu_{j}||_2*||V_{ij}||_2}$ # angle or similarity
> 7. $\forall i \in \Omega_{L}, \forall j \in \Omega_{L+1}: R_{ij} = R_{ij} + u_{ij}$ # increase coefficient of pairs that have large similarity
>
> **return**  $\mu, a'$
   
We can see that Capsule Network and **Deformable Part Model** (DPM) can supplement each other. DPM helps Capsule Network to handle deformable part-whole relationship while Capsule Network helps DPM handle occlusion senario. DPM is not robust to occlusion of parts but Capsule Network can work well with the lacking of some parts as long as most of other parts can agree with each other. The idea for routing can handle deformable parts is to apply  (DPM) and use its score as activation score of capsule $j$. The original model of DPM as follows:
$$score(z) = \sum_{i=0}^{n}{F_i \cdot \phi_i(I, p_i)} - \sum_{i=1}^{n}{d_i \cdot 
\psi(p_i, p_0)+ b}$$
where $$\psi(p_i, p_0) = (dx_i, dy_i, dx_i^2, dy_i^2)$$ with $$dx_i = x_i–x_0 \text{ and } dy_i = y_i–y_0$$ 

$F_i$ is filter for each part and $d_i$ is deformable coefficient and $b$ is bias term, that can be learned by SGD. The first term in $score(z)$ is data term representing the score of each filter at their respective locations and the second term is deformation cost that depends on the relative position of each part with respect to the root.

To apply for capsule, we replace the first term in $score(z)$ by weithed sum of activation of each capsule $i$ in layer $L$ (parts). The $\psi(p_i, p_0)$ in second term can be calculated as

$$\psi(p_i, p_j) = (\sigma_{ij}, \sigma_{ij}^2)$$
 
So, we have new DPM angle routing as follows:

**Procedure** DPM_angle_routing($a, p$):
> **input**: $a, p$: activations and poses of capsules in layer $L$; 
> **learnable**: $W$: tranformation matrix: part-whole relationship,
                 $d$: learnable deformable coefficient, size 16 * 2
> **output**: $a' \text{ and } \mu$: activations and poses of capsule in layer $L+1$
> 
>  $V = p * W$ # votes or predictions of capsule $i$ in layer $L$
>  $R_{ij} = 0$ # intialize routing coefficient for all capsules $i$ in layer $L$ and capsules $j$ in layer $L+1$
>
>  **for** t iterations **do**:
> 1. $\forall i \in \Omega_{L}: R_{ij}' = \frac{e^{R_{ij}}a_i}{\sum_k{e^{R_{ik}}}}  $ # weighted capsule activations of layer $L$ by routing coeff
> 2. $\forall j \in \Omega_{L+1}: \mu_j = \frac{\sum_i{R_{ij}V_{ij}}}{\sum_i{R_{ij}}}$ # weighted mean
> 3. $\forall i \in \Omega_{L}, \forall j \in \Omega_{L+1}: dist_{ij} = V_{ij} - \mu_j$
> 4. $\forall i \in \Omega_{L}, \forall j \in \Omega_{L+1}: \psi_{ij} = (dist_{ij}, dist_{ij}^2)$
> 5. $\forall j \in \Omega_{L+1}: a'_j = \sum_i{R_{ij}} - \sum_i{d_{ij}} \cdot \psi_{ij} + b$
> 6. $\forall i \in \Omega_{L}, \forall j \in \Omega_{L+1}:u_{ij} = \frac{\mu_{j} \cdot V_{ij}}{||\mu_{j}||_2*||V_{ij}||_2}$ # angle or similarity
> 7. $\forall i \in \Omega_{L}, \forall j \in \Omega_{L+1}: R_{ij} = R_{ij} + u_{ij}$ # increase coefficient of pairs that have large similarity
>
> **return**  $\mu, a'$

---
### Additional constraints to make Capsule Networks more interpretable
What makes Capsule Networks more powerful than traditional CNNs is instead of using scores (scalar value) for representing the activation of a neuron, Capsule Networks use both activations and pose matrices for reprenting a capsule. Therefore, pose matrices contain much richer information about object parts they represent for. We can easily visualize the object parts by decoding these pose matrices. Therefore, Capsule networks can be seen as more powerful and more interpretable than traditional CNN. To reach that goal, the recent version of Capsule Networks are is not enough, we need to add more constraints such as:
* Each Capsule represents a semantically separate meaningful object part
* The pose matrices should have the same basis for easier decoding

First, to make capsules represent a semantically separate meaningful object parts. We first convert capsule output to feature map using L1/L2/L_infinity or:
![IMAGE](quiver-image-url/C219725435D05DB4545BCC9C50F1B055.jpg =785x114)
Then, We can use the idea from [Interpretable Convolutional Neural Networks](https://arxiv.org/abs/1710.00935)

Second, to make capsules' pose matrices $P$ have the same basis, we use $QR$ decomposition of pose matric (i.e. $P= QR$) where each column $q$ of $Q$ is the basis of $P$. We can use the standard deviation of pose matrices' basis as the loss. 