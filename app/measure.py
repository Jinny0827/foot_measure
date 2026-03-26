import cv2
import numpy as np

# A4 용지 실제 크기 (mm) - 픽셀→mm 변환 기준값
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297

def load_image(path):
    """
    이미지 파일을 읽어서 반환
    - path: 이미지 파일 경로
    - 파일이 없거나 읽기 실패 시 예외 발생
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다.: {path}")
    return img

def preprocess(img):
    """
    이미지 전처리
    - BGR → HSV 변환: 색상 기반 분리에 HSV가 더 유리함
    - GaussianBlur: 노이즈 제거 (커널 5x5)
    """
    img = np.array(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (5,5), 0)
    return np.array(blurred)

def get_pixel_per_mm(paper_bbox):
    """
    프론트에서 받은 가이드 박스 좌표로 1mm당 픽셀 수 계산
    - paper_bbox: (x, y, w, h) - 프론트 가이드 박스 좌표
    - 세로가 더 긴 쪽을 A4 높이(297mm)로 사용
    - 반환: (가로 비율, 세로 비율, paper_bbox)
    """
    x, y, w, h = paper_bbox

    # 세로가 더 긴 쪽이 A4 높이(297mm)
    if w > h:
        w, h = h, w

    pixel_per_mm_x = w / A4_WIDTH_MM
    pixel_per_mm_y = h / A4_HEIGHT_MM

    return pixel_per_mm_x, pixel_per_mm_y, paper_bbox

def find_foot_contour(img, paper_bbox):
    """
    발 윤곽선 검출
    - A4 영역 안(ROI)에서만 탐색해서 오탐 줄임
    - 피부색 HSV 범위로 마스크 생성
    - morphologyEx(CLOSE): 발 내부 구멍 메우기
    - 주의: 조명/피부톤에 따라 범위 조정 필요할 수 있음
    """
    px, py, pw, ph = paper_bbox

    # A4 영역만 잘라서 ROI로 사용
    roi = img[py:py+ph, px:px+pw]

    # 피부색 범위 (HSV 기준) - 조명에 따라 튜닝 필요(인식이 안될 가능성 있음)
    lower_skin = np.array([0,  50,  80])
    upper_skin = np.array([25, 200, 255])
    mask = cv2.inRange(roi, lower_skin, upper_skin)

    # 모폴로지 클로징 : 발 윤곽 내부 빈 공간 채우기
    kernel = np.ones((15,15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("발을 찾지 못했습니다.")

    # 면적이 가장 큰 윤곽선
    foot = max(contours, key=cv2.contourArea)

    # ROI 오프셋 반환 (원본 이미지 좌표로 변환 시 필요)
    return foot, (px, py)


def measure_foot(foot_contour, roi_offset, pixel_per_mm_x, pixel_per_mm_y):
    """
    발 bounding box로 실제 길이/너비 계산
    - bounding box 높이 → 발 길이
    - bounding box 너비 → 발볼 너비
    - 픽셀 수 ÷ 픽셀/mm 비율 = 실제 mm
    """
    x, y, w, h = cv2.boundingRect(foot_contour)
    ox, oy = roi_offset

    # 픽셀 → mm 변환
    foot_length_mm = h / pixel_per_mm_y
    foot_width_mm = w / pixel_per_mm_x

    return {
        "발 길이 (mm)": round(foot_length_mm, 1),
        "발볼 너비 (mm)": round(foot_width_mm, 1),
        "발 길이 (cm)": round(foot_length_mm / 10, 1),
        "발볼 너비 (cm)": round(foot_width_mm / 10, 1),
        # ROI 오프셋 더해서 원본 이미지 기준 좌표로 변환
        "bounding_box": (x + ox, y + oy, w, h)
    }

def draw_result(img, paper_bbox, result):
    """
    결과 시각화
    - 초록색: A4 가이드 박스 영역
    - 빨간색: 발 bounding box
    - 텍스트: 발 길이 / 발볼 너비
    """
    out = img.copy()

    # A4 가이드 박스 표시 (초록)
    px, py, pw, ph = paper_bbox
    cv2.rectangle(out, (px, py), (px + pw, py + ph), (0, 255, 0), 2)

    # 발 bounding box 표시 (빨강)
    bx, by, bw, bh = result["bounding_box"]
    cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)

    # 측정값 텍스트 표시
    cv2.putText(out, f"Length: {result['발 길이 (cm)']}cm",
                (bx, by - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(out, f"Width: {result['발볼 너비 (cm)']}cm",
                (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return out
