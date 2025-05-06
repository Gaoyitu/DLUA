import torch as th
from PIL import Image
from advertorch.attacks import PGDAttack
from torchvision.models import resnet50
from torchvision import transforms
from torch import nn
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    create_classifier,
    args_to_dict,
    model_and_diffusion_defaults,
    classifier_defaults,
)

# ------------------------------
# 1) PGD attack helper
# ------------------------------
def perform_advertorch_pgd(image_tensor, target_label, target_model,
                           eps, alpha, iters, targeted=True):
    """
    Apply PGD attack to `image_tensor` targeting `target_label` on `target_model`.
    """
    device = image_tensor.device
    target_label_tensor = th.tensor([target_label], device=device)

    def model_with_timesteps(x):
        return target_model(x)

    adversary = PGDAttack(
        predict=model_with_timesteps,
        loss_fn=nn.CrossEntropyLoss(),
        eps=eps,
        nb_iter=iters,
        eps_iter=alpha,
        rand_init=True,
        clip_min=-1.0,
        clip_max=1.0,
        targeted=targeted,
    )
    x_adv = adversary.perturb(image_tensor, target_label_tensor)
    return x_adv.clamp(-1, 1)

# ------------------------------
# 2) Image loading / preprocessing
# ------------------------------
def load_and_preprocess_image(image_path, image_size=256):
    """
    Load an image from `image_path`, resize, normalize to [-1,1], and add batch dim.
    """
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])
    return transform(img).unsqueeze(0)

# ------------------------------
# 3) Main generation with dual-label guidance
# ------------------------------
def generate_image(image_path,
                   attack_target_label,
                   add_noise_steps=10,
                   denoise_steps=50,
                   alpha=2.0,
                   beta=1.5):
    """
    Generate an adversarial example via dual-label classifier guidance
    combined with PGD at each diffusion denoising step.
    """
    # -- load diffusion model --
    model_args = {
        'image_size': 256, 'class_cond': True, 'learn_sigma': True,
        'num_channels': 256, 'num_res_blocks': 2,
        'num_heads': 4, 'num_head_channels': 64,
        'num_heads_upsample': 4,
        'attention_resolutions': '32,16,8', 'dropout': 0.0,
        'diffusion_steps': 1000, 'noise_schedule': 'linear',
        'resblock_updown': True, 'use_fp16': False,
        'use_scale_shift_norm': True,
    }
    model, diffusion = create_model_and_diffusion(
        **{**model_and_diffusion_defaults(), **model_args})
    model.load_state_dict(
        th.load("256x256_diffusion.pt", map_location="cuda:0"),
        strict=False)
    model.to("cuda:0").float().eval()

    # -- load classifier for guidance --
    classifier_args = {
        'image_size': 256,
        'classifier_use_fp16': False,
        'classifier_width': 128,
        'classifier_depth': 2,
        'classifier_attention_resolutions': '32,16,8',
        'classifier_use_scale_shift_norm': True,
        'classifier_resblock_updown': True,
        'classifier_pool': 'adaptive',
    }
    classifier = create_classifier(
        **{**classifier_defaults(), **classifier_args})
    classifier.load_state_dict(
        th.load("256x256_classifier.pt", map_location="cuda:0"),
        strict=False)
    classifier.to("cuda:0").float().eval()

    # -- load target model for PGD attacks --
    target_model = resnet50()
    target_model.load_state_dict(
        th.load("resnet50-0676ba61.pth", map_location="cuda:0"),
        strict=False)
    target_model.to("cuda:0").eval()

    # -- dual-label cond_fn: single backward pass --
    def cond_fn_single(x, t, y, y_star):
        """
        Dual-label classifier guidance:
          compute G = (1+α)·log p(y) + β·log p(y*) – (α+β)·log p_uncond
          then return ∇_x G scaled.
        """
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True).float()
            logits = classifier(x_in, t)
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            log_p_y      = log_probs[range(len(logits)), y.view(-1)]
            log_p_y_star = log_probs[range(len(logits)), y_star.view(-1)]
            log_p_uncond = log_probs.logsumexp(dim=-1)

            # weighted sum of log-probs
            combined = ((1 + alpha) * log_p_y +
                        beta * log_p_y_star -
                        (alpha + beta) * log_p_uncond)
            grad = th.autograd.grad(combined.sum(), x_in)[0]

        # scale the guidance
        return grad * 4.18

    # -- load & preprocess input image --
    original_img = load_and_preprocess_image(image_path).to("cuda:0")

    # -- infer original label from classifier --
    with th.no_grad():
        logit0 = classifier(original_img, th.tensor([0], device="cuda:0"))
        original_label = logit0.argmax(dim=-1)

    # -- forward noise (q_sample) --
    noisy = original_img
    for t in range(add_noise_steps):
        tt = th.tensor([t], dtype=th.long, device="cuda:0")
        with th.no_grad():
            noisy = diffusion.q_sample(noisy, tt)

    # save intermediate noisy image
    noisy_vis = ((noisy + 1) * 127.5).clamp(0,255).to(th.uint8)
    noisy_vis = noisy_vis.permute(0,2,3,1).cpu().numpy()[0]
    Image.fromarray(noisy_vis).save("noisy_image.png")

    # -- reverse denoising with dual-label guidance + PGD --
    for i in reversed(range(add_noise_steps,
                             add_noise_steps + denoise_steps)):
        tt = th.tensor([i], dtype=th.long, device="cuda:0")
        y = original_label
        y_star = th.tensor([attack_target_label],
                           device="cuda:0")

        with th.no_grad():
            out = diffusion.p_sample(
                model, noisy, tt,
                model_kwargs={"y": y_star},
                cond_fn=lambda x, t, **kw: cond_fn_single(x, t, y, y_star)
            )
            noisy = out["sample"]

        # apply PGD attack on the current sample
        noisy = perform_advertorch_pgd(
            noisy, attack_target_label, target_model,
            eps=0.015, alpha=0.005, iters=3, targeted=True)

        # save every 10 steps
        if (add_noise_steps + denoise_steps - i) % 10 == 0 or i == add_noise_steps:
            img_vis = ((noisy + 1) * 127.5).clamp(0,255).to(th.uint8)
            img_vis = img_vis.permute(0,2,3,1).cpu().numpy()[0]
            Image.fromarray(img_vis).save(f"step_{i}.png")

    # -- save final adversarial image --
    final_vis = ((noisy + 1) * 127.5).clamp(0,255).to(th.uint8)
    final_vis = final_vis.permute(0,2,3,1).cpu().numpy()[0]
    Image.fromarray(final_vis).save("final_image.png")
    print("Image generation complete!")

# Example invocation
generate_image(
    image_path=r"D:\work\guided-diffusion-main\test_images\610.png",
    attack_target_label=30,
    add_noise_steps=10,
    denoise_steps=50
)
