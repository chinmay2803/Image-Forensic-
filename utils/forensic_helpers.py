import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

# ============================================================
#  Forensic Helper Functions (shared by both pipelines)
# ============================================================

def error_level_analysis(img_path, quality=90):
    """
    Perform Error Level Analysis (ELA) on a JPEG image.
    Returns:
        ela_image (PIL Image)
        ela_mean (float): Mean intensity of ELA map
    """
    orig = Image.open(img_path).convert("RGB")
    tmp_path = "__tmp_ela.jpg"

    # Save and reopen at lower quality to find differences
    orig.save(tmp_path, "JPEG", quality=quality)
    resaved = Image.open(tmp_path).convert("RGB")

    diff = ImageChops.difference(orig, resaved)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if extrema else 0
    scale = 255.0 / max_diff if max_diff != 0 else 1.0
    diff = ImageEnhance.Brightness(diff).enhance(scale)

    # Compute average brightness as ELA strength metric
    diff_np = np.array(diff)
    diff_gray = cv2.cvtColor(diff_np, cv2.COLOR_RGB2GRAY)
    ela_mean = float(diff_gray.mean())

    try:
        os.remove(tmp_path)
    except:
        pass

    return diff, ela_mean


def noise_residual(img_path):
    """
    Compute simple noise residual using Laplacian filter.
    Returns:
        noise_image (numpy array)
        noise_variance (float)
    """
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, 0.0

    noise = cv2.Laplacian(gray, cv2.CV_64F)
    noise_img = cv2.convertScaleAbs(noise)
    noise_var = float(np.var(noise_img))

    return noise_img, noise_var


def boundary_artifacts(image_path, mask_resized=None):
    """
    Detect boundary artifacts using Laplacian + Sobel filters.
    Optionally restrict to the predicted mask region.
    Returns:
        combined_map, overlay_image, boundary_score
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute Laplacian and Sobel edges
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)

    if np.max(sobel_mag) > 0:
        sobel = np.uint8(255 * sobel_mag / np.max(sobel_mag))
    else:
        sobel = np.zeros_like(gray, dtype=np.uint8)

    # Combine edge maps
    combined = cv2.addWeighted(lap, 0.5, sobel, 0.5, 0)

    # Apply mask (if available)
    if mask_resized is not None:
        try:
            combined = cv2.bitwise_and(combined, combined, mask=mask_resized.astype(np.uint8))
        except Exception:
            pass

    # Calculate boundary score (normalized intensity sum)
    score = float(combined.sum()) / (combined.size * 255.0)

    # Create heatmap overlay
    heat = cv2.applyColorMap(combined, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_rgb, 0.75, heat, 0.35, 0)

    return combined, overlay, score


def copy_move_detector(img_path, max_features=1000, good_ratio=0.6):
    """
    Detect potential copy-move forgeries using ORB feature matching.
    Returns:
        visualization_image (RGB numpy array)
        good_match_count (int)
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, 0

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp, des = orb.detectAndCompute(gray, None)
    if des is None or len(kp) < 10:
        return img_rgb, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des, des)
    if len(matches) == 0:
        return img_rgb, 0

    matches = sorted(matches, key=lambda x: x.distance)
    max_d = matches[-1].distance if matches else 1.0
    good_matches = [m for m in matches if m.distance < good_ratio * max_d and m.queryIdx != m.trainIdx]

    out = img_rgb.copy()
    for m in good_matches:
        pt1 = tuple(np.round(kp[m.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(kp[m.trainIdx].pt).astype(int))
        if pt1 != pt2:
            cv2.line(out, pt1, pt2, (255, 0, 0), 2)
            cv2.circle(out, pt1, 3, (0, 255, 0), -1)
            cv2.circle(out, pt2, 3, (0, 255, 0), -1)

    return out, len(good_matches)

def extract_image_forensics_info(img_path):
    """
    Extract comprehensive image forensic information including:
    - Basic image properties
    - File metadata
    - Compression analysis
    - Quantization tables (for JPEG)
    """
    from PIL import Image
    from PIL.ExifTags import TAGS
    import os
    from datetime import datetime
    
    forensics_info = {}
    qtables = {}
    
    try:
        img = Image.open(img_path)
        file_stats = os.stat(img_path)
        
        # Basic Image Properties
        forensics_info['Format'] = img.format or 'Unknown'
        forensics_info['Mode'] = img.mode
        forensics_info['Size'] = f"{img.width} x {img.height} pixels"
        forensics_info['File Size'] = f"{file_stats.st_size / 1024:.2f} KB"
        forensics_info['Modified Date'] = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        # Try to get EXIF data
        exif_data = img.getexif()
        if exif_data:
            exif_count = 0
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if isinstance(value, bytes):
                    try:
                        value = value.decode('utf-16').strip('\x00')
                    except:
                        value = str(value[:50])
                forensics_info[tag] = str(value)[:100]
                exif_count += 1
            forensics_info['EXIF_Tags_Found'] = exif_count
        else:
            forensics_info['EXIF Data'] = 'No EXIF metadata present (may indicate editing or compression)'
        
        # Color Information
        if hasattr(img, 'palette'):
            forensics_info['Color Palette'] = 'Yes' if img.palette else 'No'
        
        # Quantization Tables (JPEG)
        if img.format == 'JPEG':
            qtable_data = img.quantization
            if qtable_data:
                for i, table in enumerate(qtable_data.values()):
                    qtables[f'Q-Table {i}'] = str(list(table[:8]))  # First 8 values
                qtables['Quality Estimate'] = estimate_jpeg_quality(qtable_data)
            else:
                qtables['info'] = 'No quantization tables found'
        else:
            qtables['info'] = f'Not a JPEG (Format: {img.format})'
            
    except Exception as e:
        forensics_info['error'] = str(e)
        qtables['error'] = str(e)
    
    return forensics_info, qtables

def estimate_jpeg_quality(qtables):
    """Rough estimate of JPEG quality from quantization table"""
    try:
        first_table = list(qtables.values())[0]
        avg_val = sum(first_table[:8]) / 8
        if avg_val < 10:
            return "High (90-100%)"
        elif avg_val < 20:
            return "Medium-High (75-90%)"
        elif avg_val < 40:
            return "Medium (50-75%)"
        else:
            return "Low (<50%)"
    except:
        return "Unknown"