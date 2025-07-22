import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from pyzbar.pyzbar import decode as pyzbar_decode
import warnings
from datetime import datetime
import os

# 設置
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=UserWarning)

class SimpleSAM2QRDetector:
    """簡化的 SAM2 QR碼檢測系統 - 直接在原圖上工作"""
    
    def __init__(self, checkpoint_path, model_cfg_path, standard_qr_size_mm=53.5):
        """初始化"""
        print("🤖 Loading SAM2 model...")
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg_path, checkpoint_path))
        self.standard_qr_size_mm = standard_qr_size_mm
        print("✅ SAM2 model loaded successfully")
    
    def segment_paper(self, image):
        """使用SAM2分割紙張"""
        print("📄 Segmenting paper with SAM2...")
        
        # 轉換為RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        h, w = image_rgb.shape[:2]
        print(f"🖼️ Image size: {w}×{h}")
        
        # 使用簡單的中心點策略
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image_rgb)
            
            # 單一中心點
            input_point = np.array([[w//2, h//2]])
            input_label = np.array([1])
            
            print(f"🎯 Using center point: ({w//2}, {h//2})")
            
            # 預測
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
        
        # 選擇最佳遮罩
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        best_score = scores[best_mask_idx]
        
        print(f"✅ Paper segmentation completed! Score: {best_score:.4f}")
        
        return best_mask, best_score, input_point
    
    def get_paper_corners(self, mask):
        """從遮罩獲取紙張角點"""
        print("📐 Getting paper corners...")
        
        # 從遮罩提取輪廓
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("❌ No contours found")
            return None, None
        
        # 找最大輪廓
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        
        print(f"📊 Paper contour area: {area} pixels")
        
        # 多邊形近似獲取角點
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) >= 4:
            corners = approx.reshape(-1, 2)
            print(f"✅ Found {len(corners)} corner points")
            return contour, corners
        else:
            # 使用邊界矩形作為備用
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            print("✅ Using bounding rectangle as corners")
            return contour, box
    
    def calculate_paper_dpi(self, corners):
        """根據紙張角點計算DPI"""
        print("📏 Calculating paper DPI...")
        
        if corners is None or len(corners) < 4:
            print("❌ Invalid corners for DPI calculation")
            return None
        
        # 排序角點：TL, TR, BR, BL
        center = np.mean(corners, axis=0)
        top_points = sorted(corners, key=lambda p: p[1])[:2]
        bottom_points = sorted(corners, key=lambda p: p[1])[2:]
        top_points = sorted(top_points, key=lambda p: p[0])
        bottom_points = sorted(bottom_points, key=lambda p: p[0])
        ordered = np.array([top_points[0], top_points[1], bottom_points[1], bottom_points[0]])
        
        tl, tr, br, bl = ordered
        
        # 計算紙張像素尺寸
        width_px = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        height_px = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        
        print(f"📐 Paper dimensions: {width_px:.1f} × {height_px:.1f} px")
        
        # A4標準尺寸
        a4_width_mm, a4_height_mm = 210, 297
        
        # 判斷方向並計算DPI
        if width_px > height_px:
            # 橫向
            dpi = width_px / (a4_height_mm / 25.4)
            orientation = "Landscape"
            paper_width_mm = a4_height_mm
            paper_height_mm = a4_width_mm
        else:
            # 縱向
            dpi = width_px / (a4_width_mm / 25.4)
            orientation = "Portrait"
            paper_width_mm = a4_width_mm
            paper_height_mm = a4_height_mm
        
        px_to_mm = 25.4 / dpi
        
        print(f"📋 Orientation: {orientation}")
        print(f"📏 DPI: {dpi:.1f}")
        print(f"🔄 Conversion: {px_to_mm:.4f} mm/px")
        
        return {
            'dpi': dpi,
            'orientation': orientation,
            'px_to_mm': px_to_mm,
            'size_px': {'width': width_px, 'height': height_px},
            'size_mm': {'width': paper_width_mm, 'height': paper_height_mm},
            'corners': ordered
        }
    
    def detect_qr_codes_on_original(self, image):
        """直接在原始圖像上檢測QR碼"""
        print("🔍 Detecting QR codes on original image...")
        
        # 轉換為灰階
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        qr_results = []
        found_centers = []
        
        # 嘗試多種預處理方法
        methods = [
            ("direct", lambda img: img),
            ("clahe", lambda img: cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(img)),
            ("blur", lambda img: cv2.GaussianBlur(img, (3, 3), 0)),
            ("adaptive", lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
        ]
        
        for method_name, preprocess in methods:
            print(f"   📋 Trying {method_name} method...")
            processed = preprocess(gray)
            qr_codes = pyzbar_decode(processed)
            
            method_found = 0
            for qr in qr_codes:
                if len(qr.polygon) >= 4:
                    corners = [(point.x, point.y) for point in qr.polygon[:4]]
                    center = (int(np.mean([p[0] for p in corners])), 
                             int(np.mean([p[1] for p in corners])))
                    
                    # 檢查重複
                    is_duplicate = any(abs(center[0] - c[0]) < 50 and abs(center[1] - c[1]) < 50 
                                     for c in found_centers)
                    
                    if not is_duplicate:
                        # 計算QR碼像素尺寸
                        width_px = (np.linalg.norm(np.array(corners[1]) - np.array(corners[0])) + 
                                   np.linalg.norm(np.array(corners[2]) - np.array(corners[3]))) / 2
                        height_px = (np.linalg.norm(np.array(corners[3]) - np.array(corners[0])) + 
                                    np.linalg.norm(np.array(corners[2]) - np.array(corners[1]))) / 2
                        
                        qr_results.append({
                            'id': len(qr_results) + 1,
                            'corners': corners,
                            'center': center,
                            'size_px': {'width': width_px, 'height': height_px},
                            'data': qr.data.decode('utf-8') if qr.data else "",
                            'method': method_name
                        })
                        found_centers.append(center)
                        method_found += 1
            
            print(f"   Found {method_found} new QR codes with {method_name}")
            
            if len(qr_results) >= 4:
                break
        
        print(f"✅ Total QR codes detected: {len(qr_results)}")
        return qr_results
    
    def calculate_qr_sizes_with_perspective_correction(self, qr_results, paper_info):
        """計算QR碼實際尺寸 - 加入透視校正"""
        print("📊 Calculating QR code sizes with perspective correction...")
        
        if not paper_info:
            print("❌ No paper info available for size calculation")
            return
        
        corners = paper_info['corners']
        tl, tr, br, bl = corners
        
        # 計算紙張的透視變形參數
        paper_width_top = np.linalg.norm(tr - tl)
        paper_width_bottom = np.linalg.norm(br - bl)
        paper_height_left = np.linalg.norm(bl - tl)
        paper_height_right = np.linalg.norm(br - tr)
        
        print(f"📐 Paper edge lengths:")
        print(f"   Top: {paper_width_top:.1f}px, Bottom: {paper_width_bottom:.1f}px")
        print(f"   Left: {paper_height_left:.1f}px, Right: {paper_height_right:.1f}px")
        
        # A4 標準尺寸
        if paper_info['orientation'] == 'Landscape':
            paper_width_mm, paper_height_mm = 297, 210
        else:
            paper_width_mm, paper_height_mm = 210, 297
        
        for qr in qr_results:
            qr_x, qr_y = qr['center']
            
            # 計算 QR 碼在紙張中的相對位置 (0-1)
            # 使用雙線性插值找到 QR 碼的紙張座標
            paper_x_ratio, paper_y_ratio = self.get_paper_relative_position(
                qr_x, qr_y, corners
            )
            
            print(f"   QR{qr['id']} position in paper: ({paper_x_ratio:.2f}, {paper_y_ratio:.2f})")
            
            # 根據位置計算局部 DPI
            # 在該位置插值計算紙張的局部像素密度
            
            # 水平方向的像素密度變化
            top_dpi = paper_width_top / (paper_width_mm / 25.4)
            bottom_dpi = paper_width_bottom / (paper_width_mm / 25.4)
            local_dpi_horizontal = top_dpi + (bottom_dpi - top_dpi) * paper_y_ratio
            
            # 垂直方向的像素密度變化  
            left_dpi = paper_height_left / (paper_height_mm / 25.4)
            right_dpi = paper_height_right / (paper_height_mm / 25.4)
            local_dpi_vertical = left_dpi + (right_dpi - left_dpi) * paper_x_ratio
            
            # 使用平均 DPI
            local_dpi = (local_dpi_horizontal + local_dpi_vertical) / 2
            local_px_to_mm = 25.4 / local_dpi
            
            print(f"   QR{qr['id']} local DPI: {local_dpi:.1f} (vs paper avg: {paper_info['dpi']:.1f})")
            
            # 使用局部 DPI 計算實際尺寸
            actual_width_mm = qr['size_px']['width'] * local_px_to_mm
            actual_height_mm = qr['size_px']['height'] * local_px_to_mm
            actual_avg_mm = (actual_width_mm + actual_height_mm) / 2
            
            # 理論像素尺寸
            theoretical_px = self.standard_qr_size_mm / local_px_to_mm
            
            # 計算誤差
            error_mm = actual_avg_mm - self.standard_qr_size_mm
            error_percent = (error_mm / self.standard_qr_size_mm) * 100
            
            qr['analysis'] = {
                'actual_size_mm': {
                    'width': actual_width_mm,
                    'height': actual_height_mm,
                    'avg': actual_avg_mm
                },
                'theoretical_size_mm': self.standard_qr_size_mm,
                'theoretical_size_px': theoretical_px,
                'error_mm': error_mm,
                'error_percent': error_percent,
                'local_dpi': local_dpi,
                'paper_position': (paper_x_ratio, paper_y_ratio)
            }
            
            print(f"   QR{qr['id']}: {actual_avg_mm:.1f}mm ({error_percent:+.1f}%) [corrected]")
    
    def get_paper_relative_position(self, qr_x, qr_y, corners):
        """計算 QR 碼在紙張中的相對位置"""
        tl, tr, br, bl = corners
        
        # 使用雙線性插值找到相對位置
        # 首先找到 QR 在紙張座標系中的位置
        
        # 計算紙張的向量
        top_vector = tr - tl
        left_vector = bl - tl
        
        # QR 點相對於左上角的向量
        qr_vector = np.array([qr_x, qr_y]) - tl
        
        # 投影到上邊和左邊來估算相對位置
        # 這是一個簡化的方法，假設紙張是近似矩形
        
        # X 方向相對位置 (0-1)
        top_length = np.linalg.norm(top_vector)
        if top_length > 0:
            x_ratio = np.dot(qr_vector, top_vector / top_length) / top_length
        else:
            x_ratio = 0.5
        
        # Y 方向相對位置 (0-1)
        left_length = np.linalg.norm(left_vector)
        if left_length > 0:
            y_ratio = np.dot(qr_vector, left_vector / left_length) / left_length
        else:
            y_ratio = 0.5
        
        # 限制在 0-1 範圍內
        x_ratio = max(0, min(1, x_ratio))
        y_ratio = max(0, min(1, y_ratio))
        
        return x_ratio, y_ratio
    
    def display_results(self, original, mask, paper_contour, paper_info, qr_results, prompt_points, save_path=None):
        """顯示結果"""
        print("📊 Creating visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. 原圖
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')
        
        # 2. SAM2分割
        colored_mask = np.zeros_like(original)
        if mask.dtype != bool:
            mask = mask.astype(bool)
        
        colored_mask[mask, 0] = 0    # Red
        colored_mask[mask, 1] = 255  # Green
        colored_mask[mask, 2] = 0    # Blue
        
        result = cv2.addWeighted(original, 0.4, colored_mask, 0.6, 0)
        
        # 繪製提示點
        for point in prompt_points:
            cv2.circle(result, tuple(point.astype(int)), 8, (255, 0, 0), -1)
            cv2.circle(result, tuple(point.astype(int)), 10, (255, 255, 255), 2)
        
        axes[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1].set_title('SAM2 Segmentation\n(Green: Paper, Blue: Prompt)', fontweight='bold')
        axes[1].axis('off')
        
        # 3. A4檢測
        a4_img = original.copy()
        if paper_contour is not None:
            cv2.drawContours(a4_img, [paper_contour], -1, (255, 0, 0), 3)
            
            # 標記角點
            if paper_info and 'corners' in paper_info:
                corners = paper_info['corners']
                corner_labels = ['TL', 'TR', 'BR', 'BL']
                for i, corner in enumerate(corners):
                    cv2.circle(a4_img, tuple(corner.astype(int)), 8, (0, 255, 255), -1)
                    cv2.putText(a4_img, corner_labels[i], 
                               tuple(corner.astype(int) + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        axes[2].imshow(cv2.cvtColor(a4_img, cv2.COLOR_BGR2RGB))
        if paper_info:
            title = f'A4 Paper Detection\n{paper_info["orientation"]} | {paper_info["dpi"]:.0f} DPI'
        else:
            title = 'A4 Paper Detection\n(Failed)'
        axes[2].set_title(title, fontweight='bold')
        axes[2].axis('off')
        
        # 4. QR檢測結果
        qr_img = original.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                  (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]
        
        for i, qr in enumerate(qr_results):
            color = colors[i % len(colors)]
            pts = np.array(qr['corners'], dtype=np.int32)
            cv2.polylines(qr_img, [pts], True, color, 3)
            cv2.circle(qr_img, qr['center'], 15, color, -1)
            cv2.putText(qr_img, f"{qr['id']}", (qr['center'][0] - 8, qr['center'][1] + 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        axes[3].imshow(cv2.cvtColor(qr_img, cv2.COLOR_BGR2RGB))
        axes[3].set_title(f'QR Code Detection\nFound: {len(qr_results)} codes', fontweight='bold')
        axes[3].axis('off')
        
        # 5. 尺寸比較
        if qr_results and paper_info:
            qr_ids = [f"QR{qr['id']}" for qr in qr_results]
            actual_sizes = [qr['analysis']['actual_size_mm']['avg'] for qr in qr_results]
            theoretical_sizes = [qr['analysis']['theoretical_size_mm'] for qr in qr_results]
            
            x = np.arange(len(qr_ids))
            width = 0.35
            
            axes[4].bar(x - width/2, actual_sizes, width, label='Actual', color='skyblue')
            axes[4].bar(x + width/2, theoretical_sizes, width, label='Theoretical', color='lightcoral')
            axes[4].set_xlabel('QR Codes')
            axes[4].set_ylabel('Size (mm)')
            axes[4].set_title('Size Comparison', fontweight='bold')
            axes[4].set_xticks(x)
            axes[4].set_xticklabels(qr_ids)
            axes[4].legend()
            axes[4].grid(True, alpha=0.3)
            
            # 標註數值
            for i, (actual, theoretical) in enumerate(zip(actual_sizes, theoretical_sizes)):
                axes[4].text(i - width/2, actual + 1, f'{actual:.1f}', ha='center', va='bottom', fontsize=9)
                axes[4].text(i + width/2, theoretical + 1, f'{theoretical:.1f}', ha='center', va='bottom', fontsize=9)
        else:
            axes[4].text(0.5, 0.5, 'No QR codes or\nno paper info', ha='center', va='center', 
                        transform=axes[4].transAxes, fontsize=16)
            axes[4].set_title('Size Comparison', fontweight='bold')
            axes[4].axis('off')
        
        # 6. 誤差分析
        if qr_results and paper_info:
            errors = [abs(qr['analysis']['error_percent']) for qr in qr_results]
            colors_error = ['green' if e < 2 else 'orange' if e < 5 else 'red' for e in errors]
            
            bars = axes[5].bar(qr_ids, errors, color=colors_error)
            axes[5].set_xlabel('QR Codes')
            axes[5].set_ylabel('Error (%)')
            axes[5].set_title('Measurement Error', fontweight='bold')
            axes[5].axhline(y=2, color='green', linestyle='--', alpha=0.7, label='<2%')
            axes[5].axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='<5%')
            axes[5].legend()
            axes[5].grid(True, alpha=0.3)
            
            # 標註數值
            for bar, error in zip(bars, errors):
                height = bar.get_height()
                axes[5].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{error:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            axes[5].text(0.5, 0.5, 'No error analysis\navailable', ha='center', va='center', 
                        transform=axes[5].transAxes, fontsize=16)
            axes[5].set_title('Error Analysis', fontweight='bold')
            axes[5].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Results saved to: {save_path}")
        
        plt.show()
    
    def process_image(self, image_path, save_results=True):
        """處理圖像的主要函數"""
        print("="*60)
        print("🚀 Simple SAM2 QR Detection System")
        print("="*60)
        
        # 讀取圖像
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Cannot read image")
            return None
        
        print(f"📖 Image: {image.shape[1]}×{image.shape[0]} px")
        
        # 步驟1: SAM2分割紙張
        mask, score, prompt_points = self.segment_paper(image)
        if score < 0.5:
            print("❌ Low segmentation confidence")
            return None
        
        # 步驟2: 獲取紙張角點並計算DPI
        paper_contour, paper_corners = self.get_paper_corners(mask)
        paper_info = self.calculate_paper_dpi(paper_corners)
        
        # 步驟3: 直接在原圖上檢測QR碼
        qr_results = self.detect_qr_codes_on_original(image)
        
        # 步驟4: 計算QR碼尺寸（如果有紙張資訊）
        if paper_info:
            self.calculate_qr_sizes_with_perspective_correction(qr_results, paper_info)
        
        # 步驟5: 顯示結果
        save_path = None
        if save_results:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{base_name}_simple_sam2_qr_results_{timestamp}.png"
        
        self.display_results(image, mask, paper_contour, paper_info, qr_results, prompt_points, save_path)
        
        
        return {
            'paper_info': paper_info,
            'qr_results': qr_results,
            'save_path': save_path
        }

# 主程序
if __name__ == "__main__":
    # 配置
    checkpoint_path = "/home/itrib30156/llm_vision/sam/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg_path = "/home/itrib30156/llm_vision/sam/sam2.1_hiera_l.yaml"
    image_path = "/home/itrib30156/llm_vision/712224.jpg"
    standard_qr_size_mm = 53.5  # QR碼標準尺寸（mm）
    
    # 檢查環境
    print("🎯 Starting Simple SAM2 QR Detection System...")
    if torch.cuda.is_available():
        print("✅ CUDA available")
    else:
        print("⚠️ Using CPU (slower)")
    
    try:
        # 初始化檢測器
        detector = SimpleSAM2QRDetector(checkpoint_path, model_cfg_path, standard_qr_size_mm)
        
        # 處理圖像
        results = detector.process_image(image_path, save_results=True)
        
        if results:
            print("\n🎉 Processing completed successfully!")
            print(f"📊 Results: {len(results['qr_results'])} QR codes detected")
            if results['save_path']:
                print(f"💾 Saved to: {results['save_path']}")
        else:
            print("\n❌ Processing failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()