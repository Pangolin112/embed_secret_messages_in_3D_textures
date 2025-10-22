import torch
import torch.nn.functional as F

def extract_unique_colors(semantic_map, tolerance=1e-5):
    """
    Extract unique colors from semantic map to identify different segments.
    
    Args:
        semantic_map: [1, 3, H, W] - RGB semantic map
        tolerance: tolerance for color matching (for floating point comparison)
    
    Returns:
        unique_colors: tensor of shape [N, 3] containing unique RGB values
        color_masks: list of binary masks, one for each unique color
    """
    batch_size, channels, height, width = semantic_map.shape # [1, 3, 512, 512]

    # Reshape to [H*W, 3] for easier processing
    pixels = semantic_map[0].permute(1, 2, 0).reshape(-1, 3)  # [H*W, 3]
    
    # Find unique colors
    unique_colors = []
    color_masks = []
    
    # Get unique colors using torch.unique
    # # Round to avoid floating point issues
    # if semantic_map.dtype == torch.float32:
    #     # For float tensors, round to avoid numerical issues
    #     pixels_rounded = torch.round(pixels * 255) / 255
    #     unique_pixels = torch.unique(pixels_rounded, dim=0)
    # else:
    #     unique_pixels = torch.unique(pixels, dim=0)

    # without rounding
    unique_pixels = torch.unique(pixels, dim=0)
    print(f"Found {len(unique_pixels)} unique colors in semantic map")
    
    # Create masks for each unique color
    semantic_map_flat = semantic_map[0].permute(1, 2, 0)  # [H, W, 3]
    
    for color in unique_pixels:
        # Create mask for this color
        if semantic_map.dtype == torch.float32:
            # For float comparison
            diff = torch.abs(semantic_map_flat - color.unsqueeze(0).unsqueeze(0))
            mask = (diff.sum(dim=2) < tolerance).float()
        else:
            # For integer comparison
            mask = ((semantic_map_flat == color.unsqueeze(0).unsqueeze(0)).all(dim=2)).float()
        
        # Only keep colors that have sufficient pixels (filter out noise)
        if mask.sum() > 10:  # Minimum 10 pixels per segment
            unique_colors.append(color)
            color_masks.append(mask)
    
    if len(unique_colors) > 0:
        unique_colors = torch.stack(unique_colors)
    else:
        unique_colors = torch.empty(0, 3, device=semantic_map.device)
    
    return unique_colors, color_masks


def compute_histogram(values, num_bins=256, normalize=True):
    """
    Compute histogram of values using differentiable operations.
    
    Args:
        values: 1D tensor of pixel values in [0, 1]
        num_bins: number of histogram bins
        normalize: whether to normalize the histogram
    
    Returns:
        hist: histogram tensor of shape [num_bins]
    """
    device = values.device
    
    # Handle empty input
    if len(values) == 0:
        return torch.zeros(num_bins, device=device)
    
    # Create bin edges
    bin_edges = torch.linspace(0, 1, num_bins + 1, device=device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = 1.0 / num_bins
    
    # Soft binning using Gaussian kernels (differentiable)
    sigma = bin_width / 2  # Standard deviation for Gaussian kernel
    
    # Expand dimensions for broadcasting
    values_expanded = values.unsqueeze(1)  # [N, 1]
    centers_expanded = bin_centers.unsqueeze(0)  # [1, num_bins]
    
    # Compute soft assignments using Gaussian kernel
    distances = (values_expanded - centers_expanded) ** 2
    weights = torch.exp(-distances / (2 * sigma ** 2))
    
    # Normalize weights so they sum to 1 for each pixel
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)
    
    # Compute histogram by summing weights
    hist = weights.sum(dim=0)
    
    if normalize:
        hist = hist / (hist.sum() + 1e-10)
    
    return hist


def compute_wasserstein_distance(hist1, hist2):
    """
    Compute 1D Wasserstein (Earth Mover's) distance between two histograms.
    
    Args:
        hist1, hist2: normalized histograms of shape [num_bins]
    
    Returns:
        distance: Wasserstein distance
    """
    # Compute cumulative distributions
    cdf1 = torch.cumsum(hist1, dim=0)
    cdf2 = torch.cumsum(hist2, dim=0)
    
    # Wasserstein-1 distance is the L1 distance between CDFs
    distance = torch.abs(cdf1 - cdf2).sum() / len(hist1)
    
    return distance


def compute_chi_square_distance(hist1, hist2, eps=1e-10):
    """
    Compute chi-square distance between two histograms.
    
    Args:
        hist1, hist2: histograms of shape [num_bins]
        eps: small value to avoid division by zero
    
    Returns:
        distance: chi-square distance
    """
    diff = (hist1 - hist2) ** 2
    sum_hist = hist1 + hist2 + eps
    distance = (diff / sum_hist).sum() * 0.5
    
    return distance


def compute_histogram_loss_color_segments(semantic_map, rendered_image, original_image,
                                          num_bins=256, normalize=True, 
                                          distance_type='wasserstein'):
    """
    Compute histogram loss within each color-coded segmentation area.
    
    Args:
        semantic_map: [1, 3, H, W] - RGB semantic map where each unique color is a segment
        rendered_image: [1, 3, H, W] - rendered RGB image
        original_image: [1, 3, H, W] - original RGB image
        num_bins: number of histogram bins
        normalize: whether to normalize histograms
        distance_type: 'wasserstein', 'chi_square', 'l1', or 'l2'
    
    Returns:
        total_loss: scalar tensor representing the total histogram loss
        segment_info: dictionary with color keys and loss values
    """
    
    device = semantic_map.device
    _, _, height, width = semantic_map.shape
    
    # Ensure images are in [0, 1] range
    rendered_image = torch.clamp(rendered_image, 0, 1)
    original_image = torch.clamp(original_image, 0, 1)
    
    # Extract unique colors and their masks
    unique_colors, color_masks = extract_unique_colors(semantic_map)
    unique_colors = unique_colors.to(device)
    color_masks = [mask.to(device) for mask in color_masks]
    
    if len(unique_colors) == 0:
        return torch.tensor(0.0, device=device), {}
    
    # Convert images to grayscale for histogram computation
    # Using standard RGB to grayscale conversion
    rendered_gray = 0.299 * rendered_image[0, 0] + 0.587 * rendered_image[0, 1] + 0.114 * rendered_image[0, 2]
    original_gray = 0.299 * original_image[0, 0] + 0.587 * original_image[0, 1] + 0.114 * original_image[0, 2]
    
    total_loss = torch.tensor(0.0, device=device)
    segment_info = {}
    
    # Process each color segment
    for idx, (color, mask) in enumerate(zip(unique_colors, color_masks)):
        # Skip if segment is too small
        num_pixels = mask.sum()
        if num_pixels < 1000:
            continue
        
        # Extract pixels within this segment
        mask_bool = mask > 0.5
        rendered_pixels = rendered_gray[mask_bool]
        original_pixels = original_gray[mask_bool]
        
        # Compute histograms
        hist_rendered = compute_histogram(rendered_pixels, num_bins, normalize)
        hist_original = compute_histogram(original_pixels, num_bins, normalize)
        
        # Compute distance based on selected metric
        if distance_type == 'wasserstein':
            segment_loss = compute_wasserstein_distance(hist_rendered, hist_original)
        elif distance_type == 'chi_square':
            segment_loss = compute_chi_square_distance(hist_rendered, hist_original)
        elif distance_type == 'l1':
            segment_loss = torch.abs(hist_rendered - hist_original).sum()
        elif distance_type == 'l2':
            segment_loss = torch.sqrt(((hist_rendered - hist_original) ** 2).sum() + 1e-10)
        else:
            raise ValueError(f"Unknown distance type: {distance_type}")
        
        # Weight by segment size (proportion of image)
        segment_weight = num_pixels / (height * width)
        weighted_loss = segment_loss * segment_weight
        
        total_loss = total_loss + weighted_loss
        
        # Store segment information
        color_key = tuple(color.cpu().numpy().tolist())
        segment_info[color_key] = {
            'loss': segment_loss.item(),
            'weighted_loss': weighted_loss.item(),
            'num_pixels': int(num_pixels.item()),
            'weight': segment_weight.item()
        }
    
    return total_loss, segment_info


def compute_histogram_loss_multichannel_segments(semantic_map, rendered_image, original_image,
                                                 num_bins=256, normalize=True,
                                                 distance_type='wasserstein'):
    """
    Compute histogram loss for each RGB channel separately within each segment.
    
    Args:
        semantic_map: [1, 3, H, W] - RGB semantic map
        rendered_image: [1, 3, H, W] - rendered RGB image
        original_image: [1, 3, H, W] - original RGB image
        num_bins: number of histogram bins
        normalize: whether to normalize histograms
        distance_type: distance metric to use
    
    Returns:
        total_loss: combined loss across all channels and segments
        detailed_info: detailed loss information per segment and channel
    """
    
    device = semantic_map.device
    _, _, height, width = semantic_map.shape
    
    # Ensure images are in [0, 1] range
    rendered_image = torch.clamp(rendered_image, 0, 1)
    original_image = torch.clamp(original_image, 0, 1)
    
    # Extract unique colors and their masks
    unique_colors, color_masks = extract_unique_colors(semantic_map)
    
    if len(unique_colors) == 0:
        return torch.tensor(0.0, device=device), {}
    
    total_loss = torch.tensor(0.0, device=device)
    detailed_info = {}
    
    # Process each color segment
    for idx, (color, mask) in enumerate(zip(unique_colors, color_masks)):
        num_pixels = mask.sum()
        if num_pixels < 10:
            continue
        
        mask_bool = mask > 0.5
        color_key = tuple(color.cpu().numpy().tolist())
        channel_losses = []
        
        # Process each RGB channel
        for channel in range(3):
            rendered_channel = rendered_image[0, channel]
            original_channel = original_image[0, channel]
            
            # Extract pixels
            rendered_pixels = rendered_channel[mask_bool]
            original_pixels = original_channel[mask_bool]
            
            # Compute histograms
            hist_rendered = compute_histogram(rendered_pixels, num_bins, normalize)
            hist_original = compute_histogram(original_pixels, num_bins, normalize)
            
            # Compute distance
            if distance_type == 'wasserstein':
                channel_loss = compute_wasserstein_distance(hist_rendered, hist_original)
            elif distance_type == 'chi_square':
                channel_loss = compute_chi_square_distance(hist_rendered, hist_original)
            elif distance_type == 'l1':
                channel_loss = torch.abs(hist_rendered - hist_original).sum()
            else:  # l2
                channel_loss = torch.sqrt(((hist_rendered - hist_original) ** 2).sum() + 1e-10)
            
            channel_losses.append(channel_loss)
        
        # Average loss across channels
        segment_loss = sum(channel_losses) / 3
        
        # Weight by segment size
        segment_weight = num_pixels / (height * width)
        weighted_loss = segment_loss * segment_weight
        
        total_loss = total_loss + weighted_loss
        
        # Store detailed information
        detailed_info[color_key] = {
            'total_loss': segment_loss.item(),
            'weighted_loss': weighted_loss.item(),
            'channel_losses': [cl.item() for cl in channel_losses],
            'num_pixels': int(num_pixels.item()),
            'weight': segment_weight.item()
        }
    
    return total_loss, detailed_info


# Main usage function
def compute_loss(data_secret, model_outputs_secret, original_image_secret, 
                 use_multichannel=False, distance_type='wasserstein'):
    """
    Main function to compute histogram loss with your data.
    
    Args:
        data_secret: dictionary containing the semantic map
        model_outputs_secret: dictionary containing the rendered RGB image
        original_image_secret: original image tensor
        use_multichannel: if True, compute loss for each RGB channel separately
        distance_type: 'wasserstein', 'chi_square', 'l1', or 'l2'
    
    Returns:
        loss: total histogram loss
    """
    # Prepare inputs
    semantic_map_secret = data_secret["semantic"].unsqueeze(dim=0).permute(0, 3, 1, 2)  # [1, 3, h, w]
    rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(0).permute(0, 3, 1, 2)
    original_image = original_image_secret.unsqueeze(dim=0).permute(0, 3, 1, 2)
    
    if use_multichannel:
        # Compute multi-channel histogram loss
        loss, detailed_info = compute_histogram_loss_multichannel_segments(
            semantic_map_secret,
            rendered_image_secret,
            original_image,
            num_bins=256,
            normalize=True,
            distance_type=distance_type
        )
    else:
        # Compute grayscale histogram loss
        loss, segment_info = compute_histogram_loss_color_segments(
            semantic_map_secret,
            rendered_image_secret,
            original_image,
            num_bins=256,
            normalize=True,
            distance_type=distance_type
        )
    
    return loss


# Optional: Visualization helper
def visualize_segments(semantic_map):
    """
    Helper function to visualize the extracted segments.
    
    Args:
        semantic_map: [1, 3, H, W] RGB semantic map
    
    Returns:
        segment_visualization: dictionary with segment information
    """
    unique_colors, color_masks = extract_unique_colors(semantic_map)
    
    print(semantic_map[0])

    print(f"Found {len(unique_colors)} unique segments")
    
    visualization = {}
    for idx, (color, mask) in enumerate(zip(unique_colors, color_masks)):
        color_rgb = color.cpu().numpy()
        num_pixels = mask.sum().item()
        percentage = (num_pixels / mask.numel()) * 100
        
        visualization[idx] = {
            'color': color_rgb,
            'num_pixels': int(num_pixels),
            'percentage': percentage
        }
        
        print(f"Segment {idx}: RGB{tuple(color_rgb)}, "
              f"{int(num_pixels)} pixels ({percentage:.2f}%)")
    
    return visualization