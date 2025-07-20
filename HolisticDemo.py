import cv2
import mediapipe as mp
import numpy as np
import uuid
import logging
import math
from typing import Tuple, Optional, Dict, Any

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

# 用于存储每个人的上一次Y坐标
last_y_positions = {}

# 用于存储每个人的手部状态
hand_states = {}

# 用于存储每个人的双手合十状态
hands_joined_states = {}

def calculate_hand_distance(hand_landmarks, image_width, image_height):
    """计算手部关键点之间的距离，用于判断手是否紧握
    
    Args:
        hand_landmarks: MediaPipe手部关键点
        image_width: 图像宽度
        image_height: 图像高度
        
    Returns:
        手指尖与手掌中心的平均距离
    """
    if not hand_landmarks:
        return 0
    
    # 获取手掌中心点
    palm_center_x = hand_landmarks.landmark[9].x * image_width  # 中指掌指关节
    palm_center_y = hand_landmarks.landmark[9].y * image_height
    
    # 计算各指尖到手掌中心的距离
    finger_tips = [4, 8, 12, 16, 20]  # 拇指、食指、中指、无名指、小指的指尖索引
    distances = []
    
    for tip in finger_tips:
        tip_x = hand_landmarks.landmark[tip].x * image_width
        tip_y = hand_landmarks.landmark[tip].y * image_height
        distance = math.sqrt((tip_x - palm_center_x)**2 + (tip_y - palm_center_y)**2)
        distances.append(distance)
    
    # 返回平均距离
    return sum(distances) / len(distances) if distances else 0

def is_hands_clenched(results, image_width, image_height):
    """检测双手是否紧握
    
    Args:
        results: MediaPipe处理结果
        image_width: 图像宽度
        image_height: 图像高度
        
    Returns:
        是否检测到双手紧握
    """
    # 检查是否同时检测到左右手
    if not results.left_hand_landmarks or not results.right_hand_landmarks:
        return False
    
    # 计算左右手的手指距离
    left_distance = calculate_hand_distance(results.left_hand_landmarks, image_width, image_height)
    right_distance = calculate_hand_distance(results.right_hand_landmarks, image_width, image_height)
    
    # 设置阈值，当距离小于阈值时认为手是紧握的
    # 这个阈值需要根据实际情况调整
    threshold = image_width * 0.05  # 图像宽度的5%作为阈值
    
    # 当左右手都紧握时返回True
    is_left_clenched = left_distance < threshold
    is_right_clenched = right_distance < threshold
    
    logging.info(f"左手距离: {left_distance:.2f}, 右手距离: {right_distance:.2f}, 阈值: {threshold:.2f}")
    
    return is_left_clenched and is_right_clenched

def check_hands_joined(results, image_width, image_height):
    """检测双手是否合十
    
    Args:
        results: MediaPipe处理结果
        image_width: 图像宽度
        image_height: 图像高度
        
    Returns:
        是否检测到双手合十
    """
    # 检查是否同时检测到左右手
    if not results.left_hand_landmarks or not results.right_hand_landmarks:
        return False
    
    # 获取左右手的关键点
    left_hand_landmarks = results.left_hand_landmarks
    right_hand_landmarks = results.right_hand_landmarks
    
    # 计算左右手的距离
    # 我们使用食指指尖(8)、中指指尖(12)和无名指指尖(16)的平均位置来判断
    left_index_tip = np.array([left_hand_landmarks.landmark[8].x * image_width,
                              left_hand_landmarks.landmark[8].y * image_height])
    left_middle_tip = np.array([left_hand_landmarks.landmark[12].x * image_width,
                               left_hand_landmarks.landmark[12].y * image_height])
    left_ring_tip = np.array([left_hand_landmarks.landmark[16].x * image_width,
                             left_hand_landmarks.landmark[16].y * image_height])
    
    right_index_tip = np.array([right_hand_landmarks.landmark[8].x * image_width,
                               right_hand_landmarks.landmark[8].y * image_height])
    right_middle_tip = np.array([right_hand_landmarks.landmark[12].x * image_width,
                                right_hand_landmarks.landmark[12].y * image_height])
    right_ring_tip = np.array([right_hand_landmarks.landmark[16].x * image_width,
                              right_hand_landmarks.landmark[16].y * image_height])
    
    # 计算左右手指尖之间的距离
    index_distance = np.linalg.norm(left_index_tip - right_index_tip)
    middle_distance = np.linalg.norm(left_middle_tip - right_middle_tip)
    ring_distance = np.linalg.norm(left_ring_tip - right_ring_tip)
    
    # 计算平均距离
    avg_distance = (index_distance + middle_distance + ring_distance) / 3
    
    # 设置阈值，当距离小于阈值时认为双手合十
    # 这个阈值需要根据实际情况调整
    threshold = image_width * 0.1  # 图像宽度的10%作为阈值
    
    # 记录距离信息
    logging.info(f"双手指尖平均距离: {avg_distance:.2f}, 阈值: {threshold:.2f}")
    
    # 当双手指尖距离小于阈值时，认为双手合十
    return avg_distance < threshold

def process_image(image, person_id):
    """处理图像并检测跳跃和手势
    
    Args:
        image: 输入图像
        person_id: 人物ID，用于跟踪不同的人
        
    Returns:
        tuple: (shoulder_y, jump_info, hands_clenched, hands_joined) 肩部Y坐标、跳跃状态、双手紧握状态和双手合十状态
    """
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
        hands_clenched = False
        hands_joined = False
        
        # 获取图像尺寸
        height, width, _ = image.shape
        
        if results.pose_landmarks:
            # 获取左右肩的坐标
            left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            
            # 计算肩部中心点的Y坐标
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # 跳跃检测
            if person_id in last_y_positions:
                y_change = last_y_positions[person_id] - shoulder_y
                if y_change > 0.1:  # 向上移动超过阈值
                    jump_info = "Jumping Up"
                elif y_change < -0.1:  # 向下移动超过阈值
                    jump_info = "Landing"
                else:
                    jump_info = "Standing"
            
            # 更新该人物的上一次Y坐标
            last_y_positions[person_id] = shoulder_y
        
        # 检测双手紧握
        if results is not None:
            hands_clenched = is_hands_clenched(results, width, height)
            
            # 检测双手合十
            hands_joined = check_hands_joined(results, width, height)
            
        return shoulder_y if shoulder_y is not None else 0.0, jump_info, hands_clenched, hands_joined

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
                shoulder_y, jump_info, hands_clenched, hands_joined = process_image(image, 0)  # 使用固定person_id 0
            except Exception as e:
                logging.error(f"处理图像时出错: {str(e)}")
                continue

            # 显示结果
            cv2.putText(image, f"Shoulder Y: {shoulder_y:.2f}, Jump Info: {jump_info}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image, f"Hands Clenched: {hands_clenched}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, f"Hands Joined: {hands_joined}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
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
