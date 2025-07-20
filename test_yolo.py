import cv2
import torch
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO)

print("检查模型文件...")
model_path = 'yolov5s.pt'
if not os.path.exists(model_path):
    print(f"错误：模型文件 {model_path} 不存在")
else:
    print(f"模型文件 {model_path} 存在，大小: {os.path.getsize(model_path) / (1024*1024):.2f} MB")

print("\n尝试直接使用 torch 加载模型...")
try:
    # 尝试直接使用 torch 加载模型
    model = torch.load(model_path, map_location='cpu')
    print("成功加载模型！")
    print(f"模型类型: {type(model)}")
    
    # 打印模型结构的一些信息
    if hasattr(model, 'names'):
        print(f"模型类别: {model.names}")
    
except Exception as e:
    print(f"加载模型失败: {str(e)}")

print("\n尝试使用 yolov5 库加载模型...")
try:
    import yolov5
    
    # 尝试使用 yolov5 库加载模型
    model = yolov5.YOLOv5(model_path, device='cpu')
    print("成功使用 yolov5 库加载模型！")
    
    # 测试模型是否可用
    img = cv2.imread('demo.gif')
    if img is None:
        print("无法读取测试图像")
    else:
        print(f"测试图像尺寸: {img.shape}")
        results = model.predict(img)
        print("模型推理成功！")
        print(f"检测结果: {results}")
        
except Exception as e:
    print(f"使用 yolov5 库加载模型失败: {str(e)}")