import cv2

# 尝试打开摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头，请检查权限设置")
else:
    print("摄像头已成功打开")
    
    # 读取一帧
    ret, frame = cap.read()
    
    if ret:
        print(f"成功读取帧，尺寸: {frame.shape}")
        
        # 显示帧
        cv2.imshow('Camera Test', frame)
        print("按任意键关闭窗口")
        cv2.waitKey(0)
    else:
        print("无法读取帧")
    
    # 释放摄像头
    cap.release()
    cv2.destroyAllWindows()