import torch
import torch.nn.functional as F
import numpy as np
import cv2

# Simplified FBA network head (enough for inference)
class FBAMatting(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models.resnet import resnet34
        self.backbone = resnet34(weights=None)   # no pretrained weights
        self.backbone.fc = torch.nn.Identity()

        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(512 + 4, 256, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 128, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 1, 1)
        )

    def forward(self, image, trimap):
        # image: Bx3xHxW (0-1), trimap: Bx1xHxW (0/0.5/1)
        feats = self.backbone(image)                 # Bx512
        b, _, h, w = image.shape
        feats = feats.view(b, 512, 1, 1).expand(-1, -1, h, w)
        x = torch.cat([feats, image, trimap], dim=1) # Bx(512+4)xHxW
        alpha = torch.sigmoid(self.head(x))
        return alpha

def load_fba(weights_path):
    model = FBAMatting()
    # IMPORTANT: weights_only=False
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def make_trimap(alpha, erode_size=10, dilate_size=10):
    fg = (alpha >= 0.98).astype(np.uint8) * 255
    bg = (alpha <= 0.02).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    fg = cv2.erode(fg, kernel)
    bg = cv2.erode(bg, kernel)
    unknown = 255 - cv2.max(fg, bg)

    trimap = np.zeros_like(alpha, dtype=np.float32)
    trimap[bg == 255] = 0.0
    trimap[unknown == 255] = 0.5
    trimap[fg == 255] = 1.0
    return trimap

@torch.no_grad()
def refine_alpha_fba(image_bgr, alpha_init, model):
    h, w = alpha_init.shape
    trimap = make_trimap(alpha_init, erode_size=5, dilate_size=5)

    img = torch.from_numpy(image_bgr[:, :, ::-1].astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0)
    tri = torch.from_numpy(trimap).unsqueeze(0).unsqueeze(0).float()

    def pad32(x):
        H, W = x.shape[-2:]
        ph = (32 - H % 32) % 32
        pw = (32 - W % 32) % 32
        return F.pad(x, (0, pw, 0, ph), mode='reflect'), ph, pw

    img_p, ph, pw = pad32(img)
    tri_p, _, _ = pad32(tri)

    model = model.cpu()
    alpha_pred = model(img_p, tri_p)
    if ph or pw:
        alpha_pred = alpha_pred[:, :, :-ph or None, :-pw or None]

    alpha_np = alpha_pred.squeeze().cpu().numpy()
    tri_np = tri.squeeze().numpy()
    alpha_np[tri_np == 0] = 0
    alpha_np[tri_np == 1] = 1
    alpha_np = np.clip(alpha_np, 0, 1)
    return alpha_np.astype(np.float32)