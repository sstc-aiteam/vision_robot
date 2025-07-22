import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 配置路徑
checkpoint = "/home/itrib30156/llm_vision/sam/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg  = "/home/itrib30156/llm_vision/sam/sam2.1_hiera_l.yaml"
image_path = "/home/itrib30156/llm_vision/sam/638830629509100000.jpeg"
result_image_path = "simple_result_4.jpg"
result_image_mask_path = "simple_result_4_mask.png"

def get_boundary_points_and_center(mask):
    """獲取物體的上下左右四個邊界點和重心"""
    coords = np.column_stack(np.where(mask > 0))
    
    if len(coords) == 0:
        return None, None
    
    # 獲取邊界點
    top_point = coords[np.argmin(coords[:, 0])]      # 最上方的點
    bottom_point = coords[np.argmax(coords[:, 0])]   # 最下方的點
    left_point = coords[np.argmin(coords[:, 1])]     # 最左邊的點
    right_point = coords[np.argmax(coords[:, 1])]    # 最右邊的點
    
    # 轉換為 (x, y) 格式
    boundary_points = {
        'top': (top_point[1], top_point[0]),
        'bottom': (bottom_point[1], bottom_point[0]),
        'left': (left_point[1], left_point[0]),
        'right': (right_point[1], right_point[0])
    }
    
    # 計算重心 (使用OpenCV moments方法)
    M = cv2.moments((mask * 255).astype(np.uint8))
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center = (cx, cy)
    else:
        # 備用方法：像素平均
        mean_y = np.mean(coords[:, 0])
        mean_x = np.mean(coords[:, 1])
        center = (int(mean_x), int(mean_y))
    
    return boundary_points, center

def draw_points_on_image(image, boundary_points, center, mask=None):
    """在圖像上繪制邊界點、重心和遮罩輪廓"""
    result_image = image.copy()
    
    # 繪制遮罩輪廓
    if mask is not None:
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
    
    # 繪制邊界點
    colors = {
        'top': (255, 0, 0),     # 藍色
        'bottom': (0, 255, 0),  # 綠色  
        'left': (0, 0, 255),    # 紅色
        'right': (255, 0, 255)  # 紫色
    }
    
    labels = {
        'top': 'T',
        'bottom': 'B',
        'left': 'L', 
        'right': 'R'
    }
    
    for point_name, (x, y) in boundary_points.items():
        color = colors[point_name]
        cv2.circle(result_image, (int(x), int(y)), 8, color, -1)
        cv2.circle(result_image, (int(x), int(y)), 10, (255, 255, 255), 2)
        
        cv2.putText(result_image, labels[point_name], 
                   (int(x) - 10, int(y) - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, color, 2)
    
    # 繪制重心
    cx, cy = center
    cv2.circle(result_image, (cx, cy), 12, (255, 255, 0), -1)  # 青色
    cv2.circle(result_image, (cx, cy), 15, (0, 0, 0), 2)
    
    # 繪制十字標記
    cv2.line(result_image, (cx-10, cy), (cx+10, cy), (0, 0, 0), 2)
    cv2.line(result_image, (cx, cy-10), (cx, cy+10), (0, 0, 0), 2)
    
    cv2.putText(result_image, 'CENTER', 
               (cx - 30, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, (0, 0, 0), 2)
    
    return result_image

# 主程序
if not torch.cuda.is_available():
    print("⚠️ 未檢測到 CUDA，將使用 CPU（速度較慢）")

try:
    # 載入模型
    print("載入 SAM2...")
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    print("✓ 模型載入成功")
    
    # 載入圖像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"無法載入圖像: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    print(f"✓ 圖像載入成功: {w}x{h}")
    
    # 預測
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image_rgb)
        
        input_point = np.array([[w//2, h//2]])
        input_label = np.array([1])
        
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True # 會自動選擇分數最高的
        )
    
    # 選擇最佳遮罩
    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx]
    best_score = scores[best_mask_idx]
    
    print(f"✓ 預測完成! 最佳分數: {best_score:.4f}")
    
    # 獲取邊界點和重心
    boundary_points, center = get_boundary_points_and_center(best_mask)
    
    if boundary_points and center:
        print("\n✓ 邊界點坐標:")
        for point_name, coord in boundary_points.items():
            print(f"{point_name.upper()}: {coord}")
        
        print(f"\n✓ 重心坐標: {center}")
        
        # 繪制結果
        result_image = draw_points_on_image(image, boundary_points, center, best_mask)
        
        # 保存結果
        cv2.imwrite(result_image_path, result_image)
        cv2.imwrite(result_image_mask_path, (best_mask * 255).astype(np.uint8))
        
        print("\n✓ 文件已保存:")
        print(f"\n🎯 最終結果:")
        print(f"上點: {boundary_points['top']}")
        print(f"下點: {boundary_points['bottom']}")
        print(f"左點: {boundary_points['left']}")
        print(f"右點: {boundary_points['right']}")
        print(f"重心: {center}")
        
    else:
        print("✗ 無法找到物體的邊界點或重心")

except Exception as e:
    print(f"✗ 錯誤: {e}")
    print("\n可能的解決方案:")
    print("1. 檢查路徑是否正確")
    print("2. 確認 SAM2 是否正確安裝") 
    print("3. 檢查檢查點文件是否存在")
    print("4. 確認圖像文件存在且可讀取")