#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速棉被檢測器 - 簡化版本
"""

from ultralytics import YOLO
import cv2
import time
import os

def quick_blanket_detect(image_path, conf_threshold=0.3):
    """快速檢測棉被並保存結果"""
    
    print(f"=== 快速棉被檢測 ===")
    print(f"圖片: {image_path}")
    print(f"置信度閾值: {conf_threshold}")
    
    # 加載模型
    model = YOLO('yolov8x-oiv7.pt')
    
    # 執行檢測
    results = model(image_path, conf=conf_threshold)
    
    # 定義關鍵詞
    blanket_keywords = ['blanket', 'quilt', 'comforter', 'duvet', 'throw']
    furniture_keywords = ['bed', 'couch', 'sofa', 'chair', 'furniture']
    
    # 統計變數
    total_detections = 0
    blanket_count = 0
    furniture_count = 0
    other_count = 0
    
    print(f"\n檢測結果:")
    
    # 分析檢測結果
    for r in results:
        total_detections = len(r.boxes)
        
        if total_detections > 0:
            for i, box in enumerate(r.boxes):
                class_name = model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                
                # 分類統計
                class_name_lower = class_name.lower()
                if any(keyword in class_name_lower for keyword in blanket_keywords):
                    blanket_count += 1
                    category = "🛏️ 棉被"
                elif any(keyword in class_name_lower for keyword in furniture_keywords):
                    furniture_count += 1
                    category = "🪑 家具"
                else:
                    other_count += 1
                    category = "📦 其他"
                
                print(f"  {i+1}. {class_name} - 置信度:{confidence:.2f} {category}")
    
    # 結果統計
    print(f"\n=== 統計結果 ===")
    print(f"總檢測數: {total_detections}")
    print(f"棉被相關: {blanket_count}")
    print(f"家具相關: {furniture_count}")
    print(f"其他物體: {other_count}")
    
    # 保存結果圖片
    timestamp = int(time.time())
    output_filename = f"detection_result_{timestamp}.jpg"
    
    if total_detections > 0:
        # 保存帶標註的檢測結果
        results[0].save(output_filename)
        print(f"✅ 檢測結果已保存: {output_filename}")
        
        # 結果判斷
        if blanket_count > 0:
            print(f"🎉 發現 {blanket_count} 個棉被！")
        elif furniture_count > 0:
            print(f"🏠 發現 {furniture_count} 個家具，可能包含棉被")
        else:
            print(f"📋 發現 {total_detections} 個物體，但沒有棉被相關物品")
    else:
        # 沒檢測到任何物體，保存原圖
        img = cv2.imread(image_path)
        cv2.imwrite(output_filename, img)
        print(f"❌ 沒有檢測到任何物體")
        print(f"📸 原圖已保存: {output_filename}")
    
    return {
        'total_detections': total_detections,
        'blanket_count': blanket_count,
        'furniture_count': furniture_count,
        'other_count': other_count,
        'output_file': output_filename
    }

# 主程序
if __name__ == "__main__":
    print("🚀 快速棉被檢測器")
    
    # 獲取輸入
    image_path = input("請輸入圖片路徑: ")
    
    conf_input = input("置信度閾值 (0.1-1.0，按Enter使用0.3): ")
    if conf_input.strip() == "":
        conf_threshold = 0.3
    else:
        try:
            conf_threshold = float(conf_input)
            if not 0.1 <= conf_threshold <= 1.0:
                conf_threshold = 0.3
                print("置信度超出範圍，使用默認值 0.3")
        except:
            conf_threshold = 0.3
            print("無效輸入，使用默認值 0.3")
    
    # 執行檢測
    try:
        result = quick_blanket_detect(image_path, conf_threshold)
        
        print(f"\n🎯 檢測完成！")
        print(f"總物體: {result['total_detections']}")
        print(f"棉被: {result['blanket_count']}")
        print(f"家具: {result['furniture_count']}")
        print(f"其他: {result['other_count']}")
        print(f"結果圖片: {result['output_file']}")
        
    except FileNotFoundError:
        print("❌ 找不到圖片文件，請檢查路徑是否正確")
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")