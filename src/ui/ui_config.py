import cv2

# ui/ui_config.py
# 중앙집중형 UI 설정 파일
# 색상은 BGR 튜플로 지정 (OpenCV uses BGR)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICK = 1
LINE_TYPE = cv2.LINE_AA

# Colors (B, G, R)
COLOR_OK = (0, 200, 0)
COLOR_NG = (0, 0, 200)
COLOR_NEUTRAL = (200, 200, 200)
COLOR_BG = (30, 30, 30)
COLOR_ROI = (0, 180, 255)
COLOR_ROI_ACTIVE = (0, 255, 0)
COLOR_TEXT = (255, 255, 255)

# ROI box visual settings
ROI_THICK = 2
ROI_LABEL_BG = (50,50,50)
ROI_LABEL_PADDING = 6

# status bar
STATUS_HEIGHT = 28
STATUS_BG = (40,40,40)
STATUS_TEXT_POS = (10, 20)

# general margins
MARGIN = 8
