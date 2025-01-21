import cv2
import mediapipe as mp
import numpy as np
import uuid
import logging
from typing import Tuple, Optional

# 配置日志
logging.basicConfig(level=logging.INFO)

# MediaPipe 初始化
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

class JumpDetector:
    """跳跃检测器类，用于检测和计数人物的跳跃动作。
    
    使用肩部位置的垂直变化来检测跳跃动作。需要先进行校准来确定基准位置。
    """
    
    def __init__(self):
        self.jump_count = 0
        self.initial_mid_y = None
        self.calibration_frames = 0
        self.required_calibration_frames = 30  # 校准所需的帧数
        self.lower_bound_offset = 15  # 跳跃检测的上限阈值
        self.upper_bound_offset = 100  # 蹲伏检测的下限阈值
        self.last_state = "Standing"
        self.last_active_time = None  # 用于清理长时间不活动的检测器
        self.max_jump_height = 0  # 记录最大跳跃高度
        self.last_y = None  # 记录上一帧的y坐标
        
    def calculate_shoulder_mid_y(self, landmarks, image_height: int) -> int:
        """计算两肩中点的y坐标
        
        Args:
            landmarks: MediaPipe姿态关键点
            image_height: 图像高度
            
        Returns:
            两肩中点的y坐标
        """
        try:
            left_y = int(landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height)
            right_y = int(landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height)
            mid_y = (left_y + right_y) // 2
            
            # 打印当前肩部中心点的y坐标
            if self.initial_mid_y is not None:
                relative_y = self.initial_mid_y - mid_y  # 相对于初始位置的高度变化
                logging.info(f'肩部中心点Y坐标: {mid_y}, 相对高度: {relative_y}')
            
            return mid_y
        except Exception as e:
            logging.error(f"计算肩部位置时出错: {str(e)}")
            raise
    
    def detect_jump(self, landmarks, image) -> Tuple[str, Optional[int]]:
        """检测跳跃动作
        
        Args:
            landmarks: MediaPipe姿态关键点
            image: 输入图像
            
        Returns:
            状态字符串和跳跃计数的元组
        """
        height, width, _ = image.shape
        current_mid_y = self.calculate_shoulder_mid_y(landmarks, height)
        
        # 校准阶段
        if self.initial_mid_y is None:
            if self.calibration_frames < self.required_calibration_frames:
                self.calibration_frames += 1
                if self.calibration_frames == self.required_calibration_frames:
                    self.initial_mid_y = current_mid_y
                    logging.info(f"校准完成，初始Y坐标: {self.initial_mid_y}")
                return "Calibrating", None
        
        # 设置检测阈值
        lower_bound = self.initial_mid_y - self.lower_bound_offset
        upper_bound = self.initial_mid_y + self.upper_bound_offset
        
        # 计算相对高度变化
        height_change = self.initial_mid_y - current_mid_y if self.initial_mid_y is not None else 0
        
        # 状态检测
        if current_mid_y < lower_bound:
            if self.last_state != "Jumping":
                self.jump_count += 1
                self.last_state = "Jumping"
                # 更新最大跳跃高度
                if height_change > self.max_jump_height:
                    self.max_jump_height = height_change
                    logging.info(f'新的最大跳跃高度: {self.max_jump_height}像素')
                return "Jumping!", self.jump_count
            return "Jumping", self.jump_count
        elif current_mid_y > upper_bound:
            self.last_state = "Crouching"
            return "Crouching", self.jump_count
        else:
            if self.last_state == "Jumping":
                logging.info(f'跳跃结束，最大高度: {self.max_jump_height}像素')
                self.max_jump_height = 0  # 重置最大跳跃高度
            self.last_state = "Standing"
            return "Standing", self.jump_count

# 检测器字典，用于跟踪多个人
detectors = {}

# 创建MediaPipe Pose检测器
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def clean_inactive_detectors(max_inactive_time: int = 300):
    """清理长时间不活动的检测器
    
    Args:
        max_inactive_time: 最大不活动时间（秒）
    """
    current_time = cv2.getTickCount() / cv2.getTickFrequency()
    inactive_ids = [pid for pid, detector in detectors.items() 
                   if detector.last_active_time and current_time - detector.last_active_time > max_inactive_time]
    for pid in inactive_ids:
        del detectors[pid]

def process_image(image):
    # 创建 Holistic 对象
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        
        # 转换颜色空间
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 处理图像
        results = holistic.process(image_rgb)
        
        # 获取肩部中心点的Y坐标
        shoulder_y = None
        jump_info = "No Jump"
        
        if results.pose_landmarks:
            # 获取左右肩的坐标
            left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            
            # 计算肩部中心点的Y坐标
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # 跳跃检测
            if hasattr(process_image, 'last_y'):
                y_change = process_image.last_y - shoulder_y
                if y_change > 0.1:  # 向上移动超过阈值
                    jump_info = "Jumping Up"
                elif y_change < -0.1:  # 向下移动超过阈值
                    jump_info = "Landing"
                else:
                    jump_info = "Standing"
            
            process_image.last_y = shoulder_y
        
        return shoulder_y if shoulder_y is not None else 0.0, jump_info

if __name__ == "__main__":
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("无法打开摄像头")
            
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                logging.warning("跳过空帧")
                continue

            # 处理图像
            try:
                shoulder_y, jump_info = process_image(image)
            except Exception as e:
                logging.error(f"处理图像时出错: {str(e)}")
                continue

            # 显示结果
            cv2.putText(image, f"Shoulder Y: {shoulder_y:.2f}, Jump Info: {jump_info}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow('MediaPipe Jump Detection', cv2.flip(image, 1))
            if cv2.waitKey(1) & 0xFF == 27:  # 按ESC退出
                break
                
    except Exception as e:
        logging.error(f"程序运行出错: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        pose.close()  # 释放MediaPipe资源
