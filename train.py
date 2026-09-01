# =====================================================================================
# MedCLIPSeg (Kvasir-trained) -- Cross-Domain Generalization + Ablation pipeline
#
#   TRAIN  : Kvasir-SEG only
#   EVAL   : Kvasir (in-domain, ID) + ClinicDB, ColonDB, CVC300, BKAI (zero-shot, OOD)
#            -> Table B: cross-domain generalization, WITH and WITHOUT TTA
#   ABLATE : Baseline -> +EndoViT -> +LoRA-Attn PVL -> +Boundary Feedback -> +TTA
#            -> Table A: architecture contribution (each row trained from scratch)
#   OPTIONAL: LoRA-scope study (vision-only / text-only / both branches of the PVL
#             cross-attention) -> Table C
#   FIGURE : WACV-style qualitative grid (Images / Segmentation / Uncertainty) across
#            Kvasir + all zero-shot OOD datasets, rendered at the end of the run.
#
#   Vision Encoder     : EndoViT (MAE-pretrained ViT-B/16 on Endo700k) replaces
#                         UniMedCLIP's CLIP visual tower (toggle: USE_ENDOVIT_VISUAL)
#   PVL Adapter        : bidirectional cross-attention adapter, Q/K/V/out_proj
#                         wrapped with LoRA (toggle: USE_LORA_ATTN_ADAPTER, LORA_SCOPE)
#   Seg head           : SE (channel recalibration) + lightweight boundary head whose
#                         logit is fed back into every PVL adapter on the NEXT forward
#                         pass via a learnable per-adapter gate (toggle:
#                         USE_BOUNDARY_FEEDBACK). Boundary head is NOT wired back into
#                         the segmentation logits directly.
#   Losses             : Dice + BCE + Tversky (main) + BCE boundary-supervision term.
#   Inference           : Inverse-variance TTA (flip + multiscale + MC-dropout) is
#                         treated purely as an INFERENCE-TIME enhancement, never mixed
#                         into the architecture ablation -- it is applied/compared as
#                         its own axis (see Table B and the last two rows of Table A).
# =====================================================================================
import os, sys, subprocess, shutil, glob, math, random, time, json
from statistics import mean
from functools import partial
from collections import OrderedDict

# -------------------------------------------------------------------------------------
# 0a. NUMPY 2.0 COMPATIBILITY SHIM
# -------------------------------------------------------------------------------------
import numpy as _np_compat

_NUMPY_2_REMOVED_ALIASES = {
    "Inf": _np_compat.inf, "Infinity": _np_compat.inf,
    "NINF": -_np_compat.inf, "PINF": _np_compat.inf,
    "NAN": _np_compat.nan, "NaN": _np_compat.nan,
    "NZERO": -0.0, "PZERO": 0.0,
    "infty": _np_compat.inf,
    "bool": bool, "int": int, "float": float, "complex": complex,
    "object": object, "str": str, "long": int,
}
for _name, _val in _NUMPY_2_REMOVED_ALIASES.items():
    if not hasattr(_np_compat, _name):
        setattr(_np_compat, _name, _val)
del _name, _val
print(f"NumPy {_np_compat.__version__}: patched {len(_NUMPY_2_REMOVED_ALIASES)} legacy "
      f"aliases (np.Inf, np.NaN, ...) for compatibility with older pinned deps.")

# -------------------------------------------------------------------------------------
# 0. CONFIG
# -------------------------------------------------------------------------------------
NUM_EPOCHS      = 100
ABLATION_EPOCHS = 30
BATCH_SIZE      = 16
TEST_MC         = 20
USE_TTA         = True
SEED            = 1

# ---- which extra studies to run ----
RUN_CROSS_DOMAIN_EVAL   = True
RUN_ABLATION_STUDY      = False
RUN_LORA_SCOPE_STUDY    = False

# ---- seg-head switches ----
USE_SE      = True
BOUNDARY_W  = 0.3
TVERSKY_W   = 0.3
TVERSKY_A   = 0.3
TVERSKY_B   = 0.7

# ---- LoRA-Attention PVL Adapter switches ----
USE_LORA_ATTN_ADAPTER = True
LORA_R          = 8
LORA_ALPHA      = 16
LORA_TARGETS    = ("q_proj", "k_proj", "v_proj", "out_proj")
LORA_SCOPE      = "both"  # "both" | "vision" | "text"

# ---- Boundary-Feedback adapter gating ----
USE_BOUNDARY_FEEDBACK  = True
BOUNDARY_GATE_INIT     = 0.0

# ---- EndoViT visual backbone switches ----
USE_ENDOVIT_VISUAL      = True
ENDOVIT_REPO_ID         = "egeozsoy/EndoViT"
ENDOVIT_CKPT_FILE_CANDIDATES = ("pytorch_model.bin", "endovit.pth", "checkpoint.pth")
ENDOVIT_IMG_SIZE        = 224
ENDOVIT_PATCH_SIZE      = 16
ENDOVIT_EMBED_DIM       = 768
ENDOVIT_DEPTH           = 12
ENDOVIT_NUM_HEADS       = 12
ENDOVIT_FREEZE_BACKBONE = True

ENDOVIT_MEAN = (0.3464, 0.2280, 0.2228)
ENDOVIT_STD  = (0.2520, 0.2128, 0.2093)
INCOMING_MEAN = (0.48145466, 0.4578275, 0.40821073)
INCOMING_STD  = (0.26862954, 0.26130258, 0.27577711)

CANDIDATE_VISUAL_ATTRS = [
    "visual", "clip_model.visual", "image_encoder", "vision_model",
    "clip.visual", "model.visual", "backbone.visual",
]

# -------------------------------------------------------------------------------------
# 0b. DATASET REGISTRY
# -------------------------------------------------------------------------------------
KAGGLE_KVASIR   = "/kaggle/input/datasets/krishnaryali3/kvasir/Kvasir"
KAGGLE_CLINICDB = "/kaggle/input/datasets/krishnaryali3/clinicdb/ClinicDB"
KAGGLE_COLONDB  = "/kaggle/input/datasets/krishnaryali3/colondb-ployp/ColonDB"
KAGGLE_CVC300   = "/kaggle/input/datasets/krishnaryali3/cvc300/CVC300"
KAGGLE_BKAI     = "/kaggle/input/datasets/krishnaryali3/bkai-ployp/BKAI"

DATASET_PATHS = OrderedDict([
    ("Kvasir",   KAGGLE_KVASIR),
    ("ClinicDB", KAGGLE_CLINICDB),
    ("ColonDB",  KAGGLE_COLONDB),
    ("CVC300",   KAGGLE_CVC300),
    ("BKAI",     KAGGLE_BKAI),
])

TRAIN_SOURCE    = "Kvasir"
TARGET_DATASETS = ["ClinicDB", "ColonDB", "CVC300", "BKAI"]
ALL_DATASETS    = [TRAIN_SOURCE] + TARGET_DATASETS
SOURCE          = TRAIN_SOURCE

REPO_DIR = "/kaggle/working/MedCLIPSeg"
OUT_DIR  = "/kaggle/working/runs"

# -------------------------------------------------------------------------------------
# 1. CLONE REPO + INSTALL DEPS
# -------------------------------------------------------------------------------------
def sh(cmd):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True)

if not os.path.isdir(REPO_DIR):
    sh(f"git clone --depth 1 https://github.com/HealthX-Lab/MedCLIPSeg {REPO_DIR}")

sh("pip install -q monai fvcore timm einops open_clip_torch "
   "'transformers==4.56.0' ftfy regex sentencepiece openpyxl easydict "
   "huggingface_hub connected-components-3d matplotlib || true")

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "utils"))

# -------------------------------------------------------------------------------------
# 2. STAGE ALL FIVE DATASETS
# -------------------------------------------------------------------------------------
def stage_dataset(src, name):
    dst = os.path.join(REPO_DIR, "data", name)
    if os.path.islink(dst):
        os.unlink(dst)
    elif os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.isdir(os.path.join(src, "Train_Folder")):
        cand = glob.glob(os.path.join(src, "*", "Train_Folder"))
        if cand:
            src = os.path.dirname(cand[0])
        else:
            cand2 = glob.glob(f"/kaggle/input/**/{name}/Train_Folder", recursive=True)
            if cand2:
                src = os.path.dirname(cand2[0])
    assert os.path.isdir(os.path.join(src, "Train_Folder")), \
        f"Could not find Train_Folder under {src}. Fix KAGGLE_{name.upper()}."
    os.symlink(src, dst)
    print(f"  staged {name:10s}  <-  {src}")

print("\n==================  STAGING DATASETS  ==================")
for name, path in DATASET_PATHS.items():
    stage_dataset(path, name)

# -------------------------------------------------------------------------------------
# 3. IMPORTS FROM REPO
# -------------------------------------------------------------------------------------
import torch, monai, numpy as np, cv2, pandas as pd
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from timm.models.vision_transformer import VisionTransformer

from datasets.dataloader import DatasetSegmentation, RandomGenerator, ValGenerator
from utils.main_utils import load_cfg_from_cfg_file, read_text
import trainers.medclipseg_unimedclip as M
from trainers import build_medclipseg_unimedclip

# -------------------------------------------------------------------------------------
# 3b. Robust file resolution (extension-agnostic).
# -------------------------------------------------------------------------------------
import datasets.dataloader as DL

def _resolve(folder, fname):
    p = os.path.join(folder, fname)
    if os.path.exists(p):
        return p
    stem = os.path.splitext(fname)[0]
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
                ".PNG", ".JPG", ".JPEG", ".BMP", ".TIF", ".TIFF"):
        cand = os.path.join(folder, stem + ext)
        if os.path.exists(cand):
            return cand
    hits = glob.glob(os.path.join(folder, stem + ".*"))
    return hits[0] if hits else None

def _getitem_fixed(self, idx):
    image_filename, mask_filename, text = self.data_pairs[idx]
    img_path = _resolve(self.input_path,  image_filename)
    msk_path = _resolve(self.output_path, mask_filename)
    if img_path is None:
        raise FileNotFoundError(f"No image for '{image_filename}' in {self.input_path}")
    if msk_path is None:
        raise FileNotFoundError(f"No mask for '{mask_filename}' in {self.output_path}")
    image = cv2.imread(img_path)
    if image is None:
        raise IOError(f"cv2 failed to read image: {img_path}")
    image = cv2.resize(image, (self.image_size, self.image_size))
    mask = cv2.imread(msk_path, 0)
    if mask is None:
        raise IOError(f"cv2 failed to read mask: {msk_path}")
    mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
    mask[mask < 127] = 0
    mask[mask >= 127] = 1
    image, mask = DL.correct_dims(image, mask)
    if self.one_hot_mask:
        mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
    inputs = {
        "image": image, "ground_truth_mask": mask,
        "image_name": image_filename, "mask_name": mask_filename,
        "text_prompt": text, "dataset_name": self.task_name,
    }
    if self.joint_transform:
        inputs = self.joint_transform(inputs)
    return inputs

DL.DatasetSegmentation.__getitem__ = _getitem_fixed
print("Patched DatasetSegmentation.__getitem__ (extension-robust file resolution).")

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(SEED)

# =====================================================================================
# 4. ENDOVIT VISUAL BACKBONE  (replaces UniMedCLIP's CLIP visual tower)
# =====================================================================================
class TIMMBlockWrapper(nn.Module):
    """Wraps a timm Block to accept/return Sequence-First (L, B, D) format expected
    by OpenAI CLIP's unrolled forward pass."""
    def __init__(self, timm_block):
        super().__init__()
        self.block = timm_block
    def forward(self, x, *args, **kwargs):
        return self.block(x.permute(1, 0, 2)).permute(1, 0, 2)

class MockTransformer(nn.Module):
    def __init__(self, timm_blocks):
        super().__init__()
        self.resblocks = nn.ModuleList([TIMMBlockWrapper(b) for b in timm_blocks])
    def forward(self, x, *args, **kwargs):
        for blk in self.resblocks:
            x = blk(x)
        return x

class RenormalizingConv1(nn.Module):
    """Intercepts the image at .conv1 and applies domain renormalization before the
    Conv2d projection. Keeps its OWN copies of the renorm buffers."""
    def __init__(self, conv1, endovit_mean, endovit_std, incoming_mean, incoming_std):
        super().__init__()
        self.conv1 = conv1
        self.weight = conv1.weight
        self.bias = conv1.bias
        self.register_buffer("endovit_mean", endovit_mean.clone())
        self.register_buffer("endovit_std", endovit_std.clone())
        self.register_buffer("incoming_mean", incoming_mean.clone())
        self.register_buffer("incoming_std", incoming_std.clone())

    def _renormalize(self, x):
        x01 = x * self.incoming_std + self.incoming_mean
        return (x01 - self.endovit_mean) / self.endovit_std

    def forward(self, x):
        x = self._renormalize(x)
        return self.conv1(x)


class EndoViTVisualEncoder(nn.Module):
    """Wraps timm's MAE-style ViT-B/16 to mimic the OpenAI CLIP ViT interface."""
    def __init__(self, img_size=ENDOVIT_IMG_SIZE, patch_size=ENDOVIT_PATCH_SIZE,
                 embed_dim=ENDOVIT_EMBED_DIM, depth=ENDOVIT_DEPTH,
                 num_heads=ENDOVIT_NUM_HEADS, repo_id=ENDOVIT_REPO_ID,
                 ckpt_file_candidates=ENDOVIT_CKPT_FILE_CANDIDATES,
                 freeze=ENDOVIT_FREEZE_BACKBONE, proj_dim=None):
        super().__init__()
        self.patch_size = patch_size
        self.output_dim = proj_dim if proj_dim is not None else embed_dim

        self.vit = VisionTransformer(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0, class_token=True,
        )

        ckpt_path, last_err = None, None
        for fname in ckpt_file_candidates:
            try:
                ckpt_path = hf_hub_download(repo_id=repo_id, filename=fname)
                break
            except Exception as e:
                last_err = e
        if ckpt_path is None:
            raise FileNotFoundError(
                f"None of {ckpt_file_candidates} exist in HF repo '{repo_id}'."
            ) from last_err

        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        model_keys = set(self.vit.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in model_keys}
        self.vit.load_state_dict(filtered, strict=False)
        print(f"EndoViT: loaded {len(filtered)}/{len(model_keys)} encoder tensors.")

        self.register_buffer("endovit_mean",  torch.tensor(ENDOVIT_MEAN).view(1, 3, 1, 1))
        self.register_buffer("endovit_std",   torch.tensor(ENDOVIT_STD).view(1, 3, 1, 1))
        self.register_buffer("incoming_mean", torch.tensor(INCOMING_MEAN).view(1, 3, 1, 1))
        self.register_buffer("incoming_std",  torch.tensor(INCOMING_STD).view(1, 3, 1, 1))

        self.conv1 = RenormalizingConv1(
            self.vit.patch_embed.proj,
            self.endovit_mean, self.endovit_std,
            self.incoming_mean, self.incoming_std,
        )
        self.register_parameter("class_embedding",
            nn.Parameter(self.vit.cls_token.squeeze(0).squeeze(0).clone()))
        self.register_parameter("positional_embedding",
            nn.Parameter(self.vit.pos_embed.squeeze(0).clone()))
        self.ln_pre = nn.Identity()
        self.transformer = MockTransformer(self.vit.blocks)
        self.ln_post = self.vit.norm

        if proj_dim is not None and proj_dim != embed_dim:
            scale = embed_dim ** -0.5
            self.proj = nn.Parameter(scale * torch.randn(embed_dim, proj_dim))
            print(f"EndoViT: added trainable projection {embed_dim} -> {proj_dim}.")
        else:
            self.proj = None

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad_(False)
            self.class_embedding.requires_grad_(False)
            self.positional_embedding.requires_grad_(False)
            print("EndoViT backbone frozen.")

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def _renormalize(self, x):
        x01 = x * self.incoming_std + self.incoming_mean
        return (x01 - self.endovit_mean) / self.endovit_std

    def forward(self, x, *args, **kwargs):
        x = self._renormalize(x)
        feats = self.vit.forward_features(x)
        if self.proj is not None:
            feats = feats @ self.proj
        return feats


def _infer_visual_proj_dim(model):
    dim = getattr(model, "text_proj_dim", None)
    if isinstance(dim, int) and dim > 0:
        return dim
    upscale = getattr(model, "upscale", None)
    if upscale is not None:
        for m in upscale.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                return m.in_channels
    return None


def swap_in_endovit(model, endovit_module=None):
    if endovit_module is None:
        target_device = next(model.parameters()).device \
            if any(True for _ in model.parameters()) else device
        proj_dim = _infer_visual_proj_dim(model)
        if proj_dim is None:
            print("WARNING: could not infer target visual feature dim for EndoViT.")
        endovit_module = EndoViTVisualEncoder(proj_dim=proj_dim).to(target_device)

    for path in CANDIDATE_VISUAL_ATTRS:
        parts = path.split(".")
        obj, ok = model, True
        for p in parts[:-1]:
            if not hasattr(obj, p):
                ok = False; break
            obj = getattr(obj, p)
        if ok and hasattr(obj, parts[-1]) and isinstance(getattr(obj, parts[-1]), nn.Module):
            old = getattr(obj, parts[-1])
            setattr(obj, parts[-1], endovit_module)
            print(f"Swapped visual encoder at '.{path}' "
                  f"({old.__class__.__name__} -> EndoViTVisualEncoder).")
            return endovit_module

    print(f"WARNING: could not find a visual tower at any of {CANDIDATE_VISUAL_ATTRS}")
    return None

# =====================================================================================
# 5. LoRA-ATTENTION PVL ADAPTER
# =====================================================================================
class LoRALinear(nn.Module):
    """Wraps an existing pretrained nn.Linear (frozen) and adds a trainable low-rank
    delta: y = base(x) + (alpha/r) * B(A(x))."""
    def __init__(self, base_linear: nn.Linear, r=8, alpha=16):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base_out + self.scale * lora_out


_LORA_SCOPE_TO_ATTN_NAMES = {
    "both":   ("cross_attn_img_to_txt", "cross_attn_txt_to_img"),
    "vision": ("cross_attn_img_to_txt",),
    "text":   ("cross_attn_txt_to_img",),
}

def inject_lora_into_pvl_adapters(model, r=LORA_R, alpha=LORA_ALPHA,
                                   targets=("q_proj", "k_proj", "v_proj", "out_proj"),
                                   scope="both"):
    """Walk model.pvl_adapters and replace named projection Linears inside the
    requested cross-attn direction(s) with LoRALinear, in place."""
    if not hasattr(model, "pvl_adapters"):
        print("No `pvl_adapters` ModuleList found on this model -- nothing to patch.")
        return 0
    attn_names = _LORA_SCOPE_TO_ATTN_NAMES.get(scope, _LORA_SCOPE_TO_ATTN_NAMES["both"])
    n_patched = 0
    n_adapters = len(model.pvl_adapters)
    for i, adapter in enumerate(model.pvl_adapters):
        two_way = getattr(adapter, "two_way", None)
        if two_way is None:
            continue
        for attn_name in attn_names:
            attn = getattr(two_way, attn_name, None)
            if attn is None:
                continue
            for proj_name in targets:
                lin = getattr(attn, proj_name, None)
                if isinstance(lin, nn.Linear) and not isinstance(lin, LoRALinear):
                    setattr(attn, proj_name, LoRALinear(lin, r=r, alpha=alpha))
                    n_patched += 1
    expected = n_adapters * len(attn_names) * len(targets)
    print(f"LoRA-injected {n_patched}/{expected} projection(s) across {n_adapters} "
          f"PVL adapters (scope='{scope}').")
    return n_patched

# =====================================================================================
# 6. BOUNDARY-FEEDBACK ADAPTER GATING
# =====================================================================================
class BoundaryFeedbackGate(nn.Module):
    def __init__(self, init=BOUNDARY_GATE_INIT):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(float(init)))


def register_boundary_feedback_hooks(model, h_patch, w_patch):
    if not hasattr(model, "pvl_adapters"):
        print("No `pvl_adapters` found -- boundary-feedback gating not attached.")
        return [], None

    gates = nn.ModuleList([BoundaryFeedbackGate() for _ in model.pvl_adapters]).to(device)
    model._boundary_gates = gates
    handles = []
    n_tokens = h_patch * w_patch

    def make_hook(idx):
        def _hook(module, args, kwargs):
            boundary = getattr(model, "_last_boundary", None)
            if boundary is None:
                return None
            try:
                b = boundary.detach()
                if b.dim() == 3:
                    b = b.unsqueeze(1)
                b_small = F.adaptive_avg_pool2d(torch.sigmoid(b), (h_patch, w_patch))
                b_flat = b_small.flatten(2).transpose(1, 2)  # (B, N, 1)
                g = torch.tanh(gates[idx].gate)

                new_args = list(args)
                changed = False
                for i, a in enumerate(new_args):
                    if torch.is_tensor(a) and a.dim() == 3 and a.shape[1] == n_tokens \
                            and a.shape[0] == b_flat.shape[0]:
                        new_args[i] = a + g * b_flat.to(a.dtype)
                        changed = True
                new_kwargs = dict(kwargs) if kwargs else {}
                for k, a in list(new_kwargs.items()):
                    if torch.is_tensor(a) and a.dim() == 3 and a.shape[1] == n_tokens \
                            and a.shape[0] == b_flat.shape[0]:
                        new_kwargs[k] = a + g * b_flat.to(a.dtype)
                        changed = True

                if changed:
                    return (tuple(new_args), new_kwargs)
                return None
            except Exception as e:
                print(f"  [boundary-gate hook {idx}] skipped this call ({e})")
                return None
        return _hook

    for i, adapter in enumerate(model.pvl_adapters):
        h = adapter.register_forward_pre_hook(make_hook(i), with_kwargs=True)
        handles.append(h)
    print(f"Registered boundary-feedback hooks on {len(handles)} pvl_adapters "
          f"(token grid {h_patch}x{w_patch} = {n_tokens} tokens).")
    return handles, gates

# =====================================================================================
# 7. SEG HEAD  (SE only + lightweight boundary head)
# =====================================================================================
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        hid = max(8, ch // r)
        self.fc = nn.Sequential(nn.Linear(ch, hid), nn.GELU(), nn.Linear(hid, ch), nn.Sigmoid())
    def forward(self, x):
        s = x.mean(dim=(2, 3))
        s = self.fc(s).unsqueeze(-1).unsqueeze(-1)
        return x * s

class SegHead(nn.Module):
    """SE recalibration of the feature map, plus a small, separate boundary branch."""
    def __init__(self, in_ch, use_se=True):
        super().__init__()
        self.use_se = use_se
        hid = max(64, in_ch // 4)
        if use_se:
            self.se = SEBlock(in_ch)
        self.boundary_body = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3, padding=1, bias=False), nn.GroupNorm(8, hid), nn.GELU(),
            nn.Conv2d(hid, hid, 3, padding=1, bias=False), nn.GroupNorm(8, hid), nn.GELU())
        self.boundary_out = nn.Conv2d(hid, 1, 1)

    def refine_features(self, feat_map):
        if self.use_se:
            feat_map = self.se(feat_map)
        return feat_map

    def predict_boundary(self, feat_map):
        h = self.boundary_body(feat_map)
        return self.boundary_out(h)

def _attach_seghead(self):
    self.seg_head = SegHead(self.text_proj_dim, use_se=USE_SE).to(self.device)

def compute_seg_logits_v2(self, image_features, text_features, B, H, W):
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    cls_token = image_features[:, 0, :]
    cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
    seg_feats = image_features[:, 1:, :]
    seg_feats = seg_feats / seg_feats.norm(dim=-1, keepdim=True)
    h_patch = H // self.patch_size
    w_patch = W // self.patch_size
    seg_feats = seg_feats.reshape(B, h_patch, w_patch, -1).permute(0, 3, 1, 2)
    feat_map  = self.upscale(seg_feats)
    feat_map  = self.seg_head.refine_features(feat_map)

    base = torch.einsum("bqc, bchw -> bqhw",
                        self.mask_head(text_features).unsqueeze(1), feat_map)
    seg_logits = F.interpolate(base, self.im_size, mode="bilinear",
                               align_corners=False).squeeze(1)

    b_logit = self.seg_head.predict_boundary(feat_map)
    self._last_boundary = F.interpolate(b_logit, self.im_size, mode="bilinear",
                                        align_corners=False).squeeze(1)

    return seg_logits, cls_token

M.CustomCLIP.compute_seg_logits = compute_seg_logits_v2

# ---------------------------------------------------------------------------------
# Idempotent patching
# ---------------------------------------------------------------------------------
if not getattr(M.CustomCLIP, "_seghead_patched", False):
    _orig_init = M.CustomCLIP.__init__
    def _patched_init(self, cfg, clip_model, output_hidden_states=False,
                      _orig_init=_orig_init):
        _orig_init(self, cfg, clip_model, output_hidden_states)
        _attach_seghead(self)
        if USE_ENDOVIT_VISUAL:
            swap_in_endovit(self)
        if USE_LORA_ATTN_ADAPTER:
            n = inject_lora_into_pvl_adapters(self, r=LORA_R, alpha=LORA_ALPHA,
                                               targets=LORA_TARGETS,
                                               scope=globals().get("LORA_SCOPE", "both"))
            if n == 0:
                print("WARNING: 0 LoRA projections injected -- inspect model.pvl_adapters.")
        self._last_boundary = None
    M.CustomCLIP.__init__ = _patched_init
    M.CustomCLIP._seghead_patched = True
    print("Patched CustomCLIP.__init__.")
else:
    print("CustomCLIP.__init__ already patched this session -- skipping re-patch.")

if not getattr(M, "_build_patched", False):
    _orig_build = M.build_medclipseg_unimedclip
    def build_patched(cfg, _orig_build=_orig_build):
        model = _orig_build(cfg)
        for name, p in model.named_parameters():
            if any(k in name for k in
                   ("seg_head", "lora_", "text_proj", "text_gate", "_boundary_gates")):
                p.requires_grad_(True)
        return model
    M.build_medclipseg_unimedclip = build_patched
    build_medclipseg_unimedclip = build_patched
    M._build_patched = True
    print("Patched build_medclipseg_unimedclip.")
else:
    build_medclipseg_unimedclip = M.build_medclipseg_unimedclip
    print("build_medclipseg_unimedclip already patched -- reusing it.")

def mask_to_boundary(mask):
    m = mask.unsqueeze(1).float()
    k = 3
    dil = F.max_pool2d(m, k, 1, k // 2)
    ero = -F.max_pool2d(-m, k, 1, k // 2)
    return (dil - ero).squeeze(1).clamp(0, 1)

# =====================================================================================
# 7b. GLOBAL-FLAG HELPER
# =====================================================================================
def configure_globals(endovit=True, lora=True, boundary=True, se=True, lora_scope="both"):
    global USE_ENDOVIT_VISUAL, USE_LORA_ATTN_ADAPTER, USE_BOUNDARY_FEEDBACK, USE_SE, LORA_SCOPE
    USE_ENDOVIT_VISUAL     = endovit
    USE_LORA_ATTN_ADAPTER  = lora
    USE_BOUNDARY_FEEDBACK  = boundary
    USE_SE                 = se
    LORA_SCOPE             = lora_scope
    print(f"[configure_globals] EndoViT={endovit}  LoRA={lora} (scope={lora_scope})  "
          f"Boundary-Feedback={boundary}  SE={se}")

# =====================================================================================
# 8. CONFIG BUILDER
# =====================================================================================
BASE_CFG_NAME = "Kvasir"

def get_cfg(name, base=BASE_CFG_NAME):
    cfg = load_cfg_from_cfg_file(os.path.join(REPO_DIR, "configs", f"{base}.yaml"))
    data_root = os.path.join(REPO_DIR, "data", name)
    cfg.DATASET.NAME             = name
    cfg.DATASET.TRAIN_PATH       = os.path.join(data_root, "Train_Folder") + "/"
    cfg.DATASET.VAL_PATH         = os.path.join(data_root, "Val_Folder") + "/"
    cfg.DATASET.TEST_PATH        = os.path.join(data_root, "Test_Folder") + "/"
    cfg.DATASET.TEXT_PROMPT_PATH = os.path.join(data_root, "Prompts_Folder") + "/"
    cfg.MODEL.DEVICE = device
    cfg.update({"data_percentage": 100, "seed": SEED, "output_dir": OUT_DIR,
                "prompt_design": "original", "source_dataset": SOURCE, "resume": False})
    return cfg

os.makedirs(OUT_DIR, exist_ok=True)
ckpt_dir = os.path.join(OUT_DIR, TRAIN_SOURCE, "trained_models", f"seed{SEED}")
os.makedirs(ckpt_dir, exist_ok=True)

# =====================================================================================
# 9. DATA / LOSSES / TRAIN-LOOP / EVAL HELPERS
# =====================================================================================
ce_loss   = nn.BCEWithLogitsLoss()
dice_loss = monai.losses.DiceLoss(include_background=False, sigmoid=True, reduction="mean")

def tversky_loss(logits, label, alpha=TVERSKY_A, beta=TVERSKY_B, eps=1e-6):
    p = torch.sigmoid(logits)
    g = label.float()
    if p.ndim == 3: p = p.unsqueeze(1)
    if g.ndim == 3: g = g.unsqueeze(1)
    dims = (1, 2, 3)
    tp = (p * g).sum(dims); fp = (p * (1 - g)).sum(dims); fn = ((1 - p) * g).sum(dims)
    t = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1 - t).mean()

def calc_loss(cfg, logits, label):
    loss = (cfg.TRAIN.DICE_WEIGHT * dice_loss(logits, label)
            + cfg.TRAIN.CE_WEIGHT * ce_loss(logits, label.float()))
    if TVERSKY_W > 0:
        loss = loss + TVERSKY_W * tversky_loss(logits, label)
    return loss

def build_dataloaders(source, batch_size=BATCH_SIZE):
    cfg = get_cfg(source)
    train_tf = transforms.Compose(
        [RandomGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])])
    val_tf   = ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])
    train_text = read_text(cfg.DATASET.TEXT_PROMPT_PATH + "Train_text.xlsx")
    val_text   = read_text(cfg.DATASET.TEXT_PROMPT_PATH + "Val_text.xlsx")

    train_ds = DatasetSegmentation(cfg.DATASET.TRAIN_PATH, cfg.DATASET.NAME,
                                   train_text, train_tf, image_size=cfg.DATASET.SIZE)
    val_ds   = DatasetSegmentation(cfg.DATASET.VAL_PATH,   cfg.DATASET.NAME,
                                   val_text,   val_tf,   image_size=cfg.DATASET.SIZE)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)
    print(f"[{source}] Train {len(train_ds)} | Val {len(val_ds)}")
    return cfg, train_dl, val_dl

@torch.no_grad()
def quick_val(model, val_dl):
    model.eval(); ds = []
    for b in val_dl:
        img = b["image"].to(device); msk = b["ground_truth_mask"].to(device)
        logit = model(img, text=b["text_prompt"], num_samples=1)[0]
        p = (torch.sigmoid(logit) > 0.5).float()
        if p.ndim == 3: p = p.unsqueeze(1)
        if msk.ndim == 3: msk = msk.unsqueeze(1)
        inter = (p * msk).sum((1, 2, 3))
        uni   = p.sum((1, 2, 3)) + msk.sum((1, 2, 3))
        ds.extend(((2 * inter + 1e-7) / (uni + 1e-7)).cpu().numpy())
    model.train(); return float(mean(ds))

def train_model(model, cfg, train_dl, val_dl, num_epochs, ckpt_path, tag=""):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.TRAIN.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-4)

    print(f"\n================== TRAINING [{tag}] on {cfg.DATASET.NAME} "
          f"(EndoViT={USE_ENDOVIT_VISUAL}, LoRA={USE_LORA_ATTN_ADAPTER}[{LORA_SCOPE}], "
          f"Boundary-Feedback={USE_BOUNDARY_FEEDBACK}) ==================")
    model.train(); best = 0.0
    for epoch in range(num_epochs):
        losses = []
        for b in tqdm(train_dl, desc=f"[{tag}] epoch {epoch+1}/{num_epochs}", leave=False):
            img = b["image"].to(device); msk = b["ground_truth_mask"].to(device)
            seg_logits, clip_loss = model(image=img, text=b["text_prompt"])
            loss = calc_loss(cfg, seg_logits, msk) + cfg.TRAIN.CLIP_WEIGHT * clip_loss
            if BOUNDARY_W > 0 and getattr(model, "_last_boundary", None) is not None:
                b_tgt = mask_to_boundary(msk.float())
                loss = loss + BOUNDARY_W * F.binary_cross_entropy_with_logits(
                    model._last_boundary, b_tgt)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        vd = quick_val(model, val_dl)
        print(f"[{tag}] epoch {epoch+1:3d} | train_loss {mean(losses):.4f} | "
              f"val_dice {vd:.4f}" + ("  *new best*" if vd > best else ""))
        if vd > best:
            best = vd
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_dice": best},
                       ckpt_path)
    print(f"[{tag}] Best val Dice: {best:.4f}  ->  {ckpt_path}")
    return best

# =====================================================================================
# 11. INVERSE-VARIANCE TTA INFERENCE
# =====================================================================================
@torch.no_grad()
def predict_iv_tta(model, img, text, mc=TEST_MC, scales=(1.0, 0.85, 1.15)):
    H = img.shape[-1]
    views = []
    base_views = [("id", img), ("hf", torch.flip(img, dims=[3]))]
    for tag, v in base_views:
        for s in scales:
            if s != 1.0:
                vv = F.interpolate(v, scale_factor=s, mode="bilinear", align_corners=False)
                vv = F.interpolate(vv, size=(H, H), mode="bilinear", align_corners=False)
            else:
                vv = v
            samp = torch.sigmoid(model(image=vv, text=text, num_samples=mc))
            m = samp.mean(0); var = samp.var(0) + 1e-4
            if tag == "hf":
                m = torch.flip(m, dims=[2]); var = torch.flip(var, dims=[2])
            views.append((m, var))
    num = sum(m / var for m, var in views)
    den = sum(1.0 / var for _, var in views)
    return num / den

@torch.no_grad()
def predict_iv_tta_with_uncertainty(model, img, text, mc=TEST_MC, scales=(1.0, 0.85, 1.15)):
    """Returns fused mean prediction AND uncertainty map."""
    H = img.shape[-1]
    views = []
    base_views = [("id", img), ("hf", torch.flip(img, dims=[3]))]
    for tag, v in base_views:
        for s in scales:
            if s != 1.0:
                vv = F.interpolate(v, scale_factor=s, mode="bilinear", align_corners=False)
                vv = F.interpolate(vv, size=(H, H), mode="bilinear", align_corners=False)
            else:
                vv = v
            samp = torch.sigmoid(model(image=vv, text=text, num_samples=mc))
            m = samp.mean(0); var = samp.var(0) + 1e-4
            if tag == "hf":
                m = torch.flip(m, dims=[2]); var = torch.flip(var, dims=[2])
            views.append((m, var))
    num = sum(m / var for m, var in views)
    den = sum(1.0 / var for _, var in views)
    fused_mean = num / den
    fused_var  = 1.0 / den
    return fused_mean, fused_var

@torch.no_grad()
def predict_plain(model, img, text, mc=TEST_MC):
    samp = torch.sigmoid(model(image=img, text=text, num_samples=mc))
    return samp.mean(0)

# =====================================================================================
# 12. EVALUATE  (DSC + NSD + HD95)
# =====================================================================================
from SurfaceDice import (compute_surface_distances,
                         compute_surface_dice_at_tolerance,
                         compute_dice_coefficient)

def compute_hd95(surface_distances):
    try:
        d_gt   = np.asarray(surface_distances["distances_gt_to_pred"])
        d_pred = np.asarray(surface_distances["distances_pred_to_gt"])
        all_d = np.concatenate([d_gt, d_pred])
        if all_d.size == 0:
            return 0.0
        return float(np.percentile(all_d, 95))
    except Exception:
        return float("nan")

def dsc_nsd_hd95(gt, pr):
    gt = (gt > 127).astype(np.uint8); pr = (pr > 127).astype(np.uint8)
    if gt.max() == 0 and pr.max() == 0:
        return 1.0, 1.0, 0.0
    if gt.max() == 0 and pr.max() > 0:
        return 0.0, 0.0, float("nan")
    d = compute_dice_coefficient(gt.astype(bool), pr.astype(bool))
    sd = compute_surface_distances(
        gt.astype(bool)[..., None], pr.astype(bool)[..., None], [1, 1, 1])
    n = compute_surface_dice_at_tolerance(sd, 2)
    h = compute_hd95(sd)
    return float(d), float(n), float(h)

def evaluate_dataset(model, name, use_tta=True, prompt_variant="original", batch_size=16):
    """Evaluate on the TEST split of any staged dataset."""
    tcfg = get_cfg(name)
    ttf  = ValGenerator(output_size=[tcfg.DATASET.SIZE, tcfg.DATASET.SIZE])
    prompt_file = ("Test_text_original.xlsx" if prompt_variant == "original"
                   else f"Test_text_{prompt_variant}.xlsx")
    ttext = read_text(tcfg.DATASET.TEXT_PROMPT_PATH + prompt_file)
    tds   = DatasetSegmentation(tcfg.DATASET.TEST_PATH, name, ttext, ttf,
                                image_size=tcfg.DATASET.SIZE)
    tdl   = DataLoader(tds, batch_size=batch_size, shuffle=False, num_workers=2)
    gt_dir = os.path.join(tcfg.DATASET.TEST_PATH, "label")

    model.eval()
    dscs, nsds, hds = [], [], []
    for b in tqdm(tdl, desc=f"Testing {name} [{prompt_variant}] TTA={use_tta}", leave=False):
        img = b["image"].to(device)
        prob = (predict_iv_tta(model, img, b["text_prompt"]) if use_tta
                else predict_plain(model, img, b["text_prompt"]))
        pred = (prob > 0.5).cpu().numpy().astype(np.uint8) * 255
        for i, mname in enumerate(b["mask_name"]):
            gpath = os.path.join(gt_dir, mname)
            if not os.path.exists(gpath):
                cand = glob.glob(os.path.join(gt_dir, os.path.splitext(mname)[0] + ".*"))
                if not cand:
                    continue
                gpath = cand[0]
            gt = cv2.imread(gpath, cv2.IMREAD_GRAYSCALE)
            pr = cv2.resize(pred[i], (gt.shape[1], gt.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
            d, n, h = dsc_nsd_hd95(gt, pr)
            dscs.append(d); nsds.append(n)
            if not math.isnan(h):
                hds.append(h)

    md = 100 * float(mean(dscs))
    mn = 100 * float(mean(nsds))
    mh = float(mean(hds)) if hds else float("nan")
    print(f"{name:10s}  [{prompt_variant:>12s}]  TTA={str(use_tta):5s}  N={len(dscs):4d}  "
          f"DSC {md:5.2f}%   NSD {mn:5.2f}%   HD95 {mh:6.2f}px")
    return md, mn, mh, len(dscs)

def harmonic_mean(a, b):
    if a + b == 0:
        return 0.0
    return 2 * a * b / (a + b)

# =====================================================================================
# 10. TRAIN THE FINAL MODEL
# =====================================================================================
configure_globals(endovit=True, lora=True, boundary=True, se=True, lora_scope="both")
set_seed(SEED)

cfg, train_dl, val_dl = build_dataloaders(TRAIN_SOURCE)
model = build_medclipseg_unimedclip(cfg)
model = model.to(device)

H_patch = cfg.DATASET.SIZE // ENDOVIT_PATCH_SIZE
if USE_BOUNDARY_FEEDBACK:
    _, _ = register_boundary_feedback_hooks(model, H_patch, H_patch)

ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
best_val_dice = train_model(model, cfg, train_dl, val_dl, NUM_EPOCHS, ckpt_path,
                             tag="ProbVLP-Final")

# =====================================================================================
# 13. CROSS-DOMAIN EVALUATION  (Table B)
# =====================================================================================
if RUN_CROSS_DOMAIN_EVAL:
    print("\n================== CROSS-DOMAIN GENERALIZATION (Table B) ==================")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    rows = []
    for ds_name in ALL_DATASETS:
        for use_tta in [False, True]:
            dsc, nsd, hd, n = evaluate_dataset(model, ds_name, use_tta=use_tta)
            rows.append({"Dataset": ds_name, "TTA": use_tta,
                         "DSC": dsc, "NSD": nsd, "HD95": hd, "N": n})

    df_b = pd.DataFrame(rows)
    print("\n=== Table B: Cross-Domain Generalization ===")
    print(df_b.to_string(index=False))
    df_b.to_csv(os.path.join(OUT_DIR, "table_B_cross_domain.csv"), index=False)
