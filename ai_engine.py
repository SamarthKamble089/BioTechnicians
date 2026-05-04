import torch
import numpy as np

def run_inference(volume_data, organ_type="Brain"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_data = torch.tensor(volume_data, dtype=torch.float32).to(device)

    # 1. ISOLATE SOFT TISSUE
    background = tensor_data.mean()
    
    if organ_type != "Bone":
        # Find the density where bone starts (95th percentile)
        active_data = tensor_data[tensor_data > background]
        if active_data.numel() == 0:
            return np.zeros(volume_data.shape, dtype=np.float32)
            
        bone_limit = torch.quantile(active_data, 0.95)
        
        # Strictly grab voxels that are heavier than background but lighter than bone
        soft_tissue = tensor_data[(tensor_data > background) & (tensor_data < bone_limit)]
    else:
        soft_tissue = tensor_data[tensor_data > background]
        bone_limit = float('inf')

    # 2. DYNAMIC THRESHOLDING
    threshold_map = {"Brain": 0.985, "Lungs": 0.995, "Heart": 0.950, "Spleen": 0.975, "Prostate": 0.990}
    q = threshold_map.get(organ_type, 0.985)

    # 3. STOCHASTIC SAMPLING (RTX 5070 Optimization)
    max_samples = 1000000
    if soft_tissue.numel() > max_samples:
        step = soft_tissue.numel() // max_samples
        sample = soft_tissue[::step]
        tumor_threshold = torch.quantile(sample, q)
    elif soft_tissue.numel() > 0:
        tumor_threshold = torch.quantile(soft_tissue, q)
    else:
        return np.zeros(volume_data.shape, dtype=np.float32)

    # 4. GENERATE BINARY MASK
    # Voxels must be denser than the tumor threshold, but LESS dense than bone
    mask = ((tensor_data > tumor_threshold) & (tensor_data < bone_limit)).float()

    # Fallback safety: If standard threshold yields nothing, grab the absolute densest soft-tissue
    if torch.sum(mask) < 50 and soft_tissue.numel() > 50:
        fallback_threshold = torch.quantile(soft_tissue, 0.99)
        mask = ((tensor_data > fallback_threshold) & (tensor_data < bone_limit)).float()

    return mask.cpu().numpy()

def calculate_dice(ai_mask, ground_truth):
    # Professional accuracy metric
    intersection = np.logical_and(ai_mask, ground_truth)
    if (ai_mask.sum() + ground_truth.sum()) == 0: return 1.0
    return (2.0 * intersection.sum()) / (ai_mask.sum() + ground_truth.sum())