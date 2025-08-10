## Dual-Label Guided Unrestricted Target Attack with Diffusion Model

Please cite our work if you find it useful:  
‌\*\*"Enhanced Unrestricted Adversarial Attacks via Dual-Label Guidance in Diffusion Models"\*\*‌  
*(Under Review at Neurocomputing)*

## Acknowledge

Our work is based on the following theoretical works:

* [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
* [Diffusion Models Beat GANS on Image Synthesis.](https://arxiv.org/pdf/2105.05233)

### Download Diffusion Model Weights

Download the diffusion model weights from the following links:

* [256x256\_classifier.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt)
* [256x256\_diffusion.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt)

### Generate Adversarial Examples

Place the clean images and the target model in the corresponding folders. Then, run the DLUA.py to generate adversarial examples.

### Evaluate the attack performance

Place the target model in the corresponding folder, modify the path of your adversarial examples, and run the "test.py" program to evaluate the attack performance of the adversarial examples.

