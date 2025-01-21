import cv2
import yolov5
from HolisticDemo import process_image
import logging
import os
import torch

# 配置日志
logging.basicConfig(level=logging.INFO)

# 配置CPU线程数以优化性能
torch.set_num_threads(4)  # 设置PyTorch CPU线程数，根据CPU核心数调整

# 检查模型文件是否存在
# Check if model file exists
model_path = 'yolov5s.pt'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件 {model_path} 不存在")

# 加载YOLOv5模型
# Load the YOLOv5 model
yolov5_model = yolov5.YOLOv5(
    model_path,
    device='cpu',
    load_on_init=True
)

# 设置推理参数
yolov5_model.conf = 0.5  # 设置置信度阈值
yolov5_model.iou = 0.45  # 设置NMS IOU阈值
yolov5_model.agnostic = False  # 关闭类别无关的NMS
yolov5_model.multi_label = False  # 单标签预测
yolov5_model.max_det = 10  # 允许检测多个人

# 加载视频源（可以是摄像头或视频文件）
# Load the video
video = cv2.VideoCapture(0)  # 0表示使用默认摄像头
if not video.isOpened():
    raise RuntimeError("无法打开摄像头")

# 获取视频的宽度、高度和帧率
# Get the video's width, height, and frames per second (fps)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 30  # 使用默认帧率

# 创建VideoWriter对象用于保存处理后的视频
# Create a VideoWriter object to save the video
output_file = 'output_video.mp4'  # 指定输出视频文件名
# 尝试不同的编码器
try:
    video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
except:
    try:
        video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    except:
        video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

if not video_writer.isOpened():
    raise RuntimeError("无法创建视频写入器")

# 逐帧处理视频
# Process each frame of the video
while True:
  # 读取下一帧
  success, frame = video.read()
  if not success:
    break

  # 使用YOLOv5进行目标检测
  # Use YOLOv5 for object detection
  results = yolov5_model.predict(frame)
  
  # 获取检测结果
  boxes = results.xyxy[0]  # 获取第一帧的检测框
  
  # 处理每个检测到的人
  for i, box in enumerate(boxes):
      x1, y1, x2, y2 = box[:4]  # 前4个值是边界框坐标
      confidence = box[4]
      class_id = box[5]
      
      # 只处理人的检测结果
      if class_id == 0 and confidence >= 0.5:  # class_id 0 表示人
          # 计算目标中心点坐标
          centroid_x = int((x1 + x2) // 2)
          centroid_y = int((y1 + y2) // 2)
          
          # 在图像上绘制边界框
          cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
          
          # 设置padding并提取人物区域
          padding = 25
          person = frame[int(y1):int(y2), int(x1):int(x2)]
          
          try:  # 使用mediapipe处理人物图像
              shoulder_y, jump_info = process_image(person, i)  # 传入人物ID
              
              # 根据跳跃状态选择颜色
              color = (0, 0, 255) if "Jump" in jump_info else (0, 255, 0)  # 跳跃时为红色，否则为绿色
              
              # 在每个人物上方显示信息
              text = f"Person {i+1} - Y: {shoulder_y:.2f}"
              # 先显示人物ID和Y坐标
              cv2.putText(frame, text, 
                         (int(x1), int(y1) - 45),  # 位置上移
                         cv2.FONT_HERSHEY_SIMPLEX, 
                         1.0,  # 字体大小增大
                         (0, 255, 0),  # 绿色
                         2)
              
              # 再显示跳跃状态
              cv2.putText(frame, jump_info, 
                         (int(x1), int(y1) - 10),  # 显示在下方
                         cv2.FONT_HERSHEY_SIMPLEX, 
                         1.2,  # 跳跃状态字体更大
                         color,  # 使用根据状态确定的颜色
                         3)  # 更粗的线条
              
              # 同时在控制台打印
              print(f"Person {i+1} - Shoulder Y: {shoulder_y:.2f}, Jump: {jump_info}")
          except Exception as e:
              logging.error(f"处理图像时发生错误: {str(e)}")

  # 显示处理后的帧
  # Display the frame
  cv2.imshow("Video", frame)
  video_writer.write(frame)  # 将处理后的帧写入输出视频
  # 按'q'键退出程序
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# 释放资源
# Release the video capture object
video.release()
video_writer.release()

# 关闭所有窗口
cv2.destroyAllWindows()
