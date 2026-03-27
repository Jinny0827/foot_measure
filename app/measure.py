import cv2
import numpy as np
import base64

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

def refine_paper_bbox(img_bgr, guide_bbox):
    """
    가이드 박스를 힌트로 실제 용지 경계를 정밀 탐색
    - 가이드 박스 ROI + 패딩 안에서 흰색 픽셀로 실제 용지 검출
    - 검출 실패 또는 너무 작으면 가이드 박스 그대로 반환 (fallback)
    """
    px, py, pw, ph = guide_bbox
    h, w = img_bgr.shape[:2]

    # 가이드 박스 주변 10% 패딩으로 탐색 범위 확장
    pad = int(min(pw, ph) * 0.1)
    x1 = max(0, px - pad)
    y1 = max(0, py - pad)
    x2 = min(w, px + pw + pad)
    y2 = min(h, py + ph + pad)

    roi = img_bgr[y1:y2, x1:x2]
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 흰색 범위 (용지 감지)
    lower_white = np.array([0,   0, 170])
    upper_white = np.array([180, 40, 255])
    mask = cv2.inRange(roi_hsv, lower_white, upper_white)

    # 발이 덮은 구멍 채우기 → 외부 노이즈 제거
    k_close = np.ones((30, 30), np.uint8)
    k_open  = np.ones((10, 10), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return guide_bbox  # fallback

    largest = max(contours, key=cv2.contourArea)

    # 가이드 박스 면적의 30% 미만이면 신뢰하지 않고 fallback
    if cv2.contourArea(largest) < pw * ph * 0.3:
        return guide_bbox

    rx, ry, rw, rh = cv2.boundingRect(largest)

    # ROI 오프셋 더해서 원본 이미지 좌표로 변환
    return (x1 + rx, y1 + ry, rw, rh)


def detect_a4_paper(img, guide_bbox):
    """
    이미지에서 A4 용지를 실제로 감지하여 pixel_per_mm 계산

    실패 케이스별 원인 메시지 반환:
      - 흰 바닥: "바닥 색상이 A4 용지와 비슷합니다..."
      - 형태 매칭 실패: "A4 용지 경계를 인식하지 못했습니다..."
      - 용지 없음: "A4 용지를 찾지 못했습니다..."
    """
    px, py, pw, ph = guide_bbox
    img_h, img_w = img.shape[:2]

    # ── ROI: 가이드박스 + 소폭 여유 ──────────────────────────
    m   = max(20, pw // 15)
    rx1 = max(0, px - m);        ry1 = max(0, py - m)
    rx2 = min(img_w, px+pw+m);   ry2 = min(img_h, py+ph+m)
    roi = img[ry1:ry2, rx1:rx2]
    roi_area = (rx2 - rx1) * (ry2 - ry1)

    # ── HSV 흰색 마스크: 고명도(V>175) + 저채도(S<60) ────────
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([0,   0, 175]),
                       np.array([180, 60, 255]))

    # CLOSE(발이 덮은 구멍 메우기) → OPEN(노이즈 제거)
    k_big   = np.ones((25, 25), np.uint8)
    k_small = np.ones((9,  9),  np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_big,   iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_small, iterations=1)

    # ── 흰색 비율로 바닥 색상 판별 ───────────────────────────
    white_ratio = float(np.sum(mask > 0)) / mask.size

    if white_ratio > 0.78:
        raise ValueError(
            "바닥 색상이 A4 용지와 비슷합니다. "
            "어두운 색 바닥 위에서 다시 촬영해주세요."
        )

    # ── 윤곽 검출 ─────────────────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(
            "A4 용지를 찾지 못했습니다. "
            "가이드 박스 안에 A4 전체가 보이도록 맞춰주세요."
        )

    # 면적 5% 이상인 흰색 영역만 사용
    large = [c for c in contours if cv2.contourArea(c) > roi_area * 0.05]
    if not large:
        if white_ratio > 0.40:
            raise ValueError(
                "A4 용지 경계를 인식하지 못했습니다. "
                "바닥이 밝은 색이라면 어두운 바닥에서 촬영해주세요."
            )
        raise ValueError(
            "A4 용지를 찾지 못했습니다. "
            "가이드 박스 안에 A4 전체가 보이도록 맞춰주세요."
        )

    # ── 모든 큰 흰색 영역 합산 → 전체 A4 범위 추정 ───────────
    # (발이 가운데를 가려도 양쪽 끝이 보이면 전체 크기 추정 가능)
    all_pts = np.vstack(large)
    rect    = cv2.minAreaRect(all_pts)
    (_, _), (w_px, h_px), _ = rect

    # portrait 보정: 짧은 쪽 = 210mm, 긴 쪽 = 297mm
    if w_px > h_px:
        w_px, h_px = h_px, w_px

    # ── A4 비율 검증 (210:297 ≈ 0.707) ──────────────────────
    ratio    = w_px / h_px if h_px > 0 else 0
    a4_ratio = A4_WIDTH_MM / A4_HEIGHT_MM   # ≈ 0.707

    if abs(ratio - a4_ratio) > 0.25:
        if white_ratio > 0.45:
            raise ValueError(
                "A4 용지 경계를 인식하지 못했습니다. "
                "바닥이 밝은 색이라면 어두운 바닥에서 촬영해주세요."
            )
        raise ValueError(
            "A4 용지 비율이 맞지 않습니다. "
            "용지 전체가 가이드 박스 안에 들어오는지 확인해주세요."
        )

    px_per_mm_x = w_px / A4_WIDTH_MM
    px_per_mm_y = h_px / A4_HEIGHT_MM
    return px_per_mm_x, px_per_mm_y


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

    # 피부색 범위 (HSV 기준) - S/V 하한 올려서 그림자 제외
    lower_skin = np.array([0,  80, 110])
    upper_skin = np.array([25, 220, 255])
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

def measure_arch_side(img_path, guide_x, guide_y, guide_w, guide_h, foot_length_mm=0.0):
    """
    옆면 촬영에서 아치 높이 및 평발 수준 측정
    - guide_bbox: 프론트 가이드 박스 좌표 (옆면 가이드)
    - foot_length_mm: Step1에서 측정한 발 길이 (스케일 기준값)
    """
    img = load_image(img_path)
    preprocessed = preprocess(img)
    guide_bbox = (guide_x, guide_y, guide_w, guide_h)

    # 발 윤곽 검출 (기존 함수 재사용)
    foot, roi_offset = find_foot_contour(preprocessed, guide_bbox)
    ox, oy = roi_offset

    # ROI 기준 → 절대 좌표
    abs_foot = foot + np.array([[ox, oy]])
    x, y, w, h = cv2.boundingRect(foot)       # ROI 기준 bbox
    foot_abs_x = x + ox

    # pixel_per_mm: Step1 발 길이를 스케일로 사용 (없으면 A4 너비 fallback)
    if foot_length_mm > 10:
        pixel_per_mm = w / foot_length_mm
    else:
        pixel_per_mm = guide_w / 210.0         # A4 너비 210mm

    # 발 마스크 생성 (절대 좌표 기준)
    img_h, img_w = img.shape[:2]
    foot_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.drawContours(foot_mask, [abs_foot], -1, 255, -1)

    # 각 열에서 최하단 발 픽셀 수집 → 바닥 프로파일
    bottom_profile = []
    for col in range(foot_abs_x, min(foot_abs_x + w, img_w)):
        pix = np.where(foot_mask[:, col] > 0)[0]
        if len(pix) > 0:
            bottom_profile.append((col, int(pix.max())))

    if len(bottom_profile) < 10:
        raise ValueError("발 윤곽을 충분히 감지하지 못했습니다.")

    # 바닥 기준선: 뒤꿈치(앞 15%) + 발끝(뒤 15%) 평균 y
    heel_x  = foot_abs_x + int(w * 0.15)
    toe_x   = foot_abs_x + int(w * 0.85)
    floor_pts = [p[1] for p in bottom_profile if p[0] <= heel_x or p[0] >= toe_x]
    floor_y = int(np.mean(floor_pts)) if floor_pts else max(p[1] for p in bottom_profile)

    # 아치 구간(20~70%): 바닥에서 가장 높이 떠있는(y 최소) 지점
    arch_x_s = foot_abs_x + int(w * 0.20)
    arch_x_e = foot_abs_x + int(w * 0.70)
    arch_pts  = [p for p in bottom_profile if arch_x_s <= p[0] <= arch_x_e]
    if not arch_pts:
        raise ValueError("아치 구간을 감지하지 못했습니다.")

    arch_peak       = min(arch_pts, key=lambda p: p[1])
    arch_height_px  = max(0, floor_y - arch_peak[1])
    arch_height_mm  = arch_height_px / pixel_per_mm

    # 아치 등급 분류
    if arch_height_mm < 4:
        arch_level, arch_score = '평발',    0
    elif arch_height_mm < 10:
        arch_level, arch_score = '저아치',  1
    elif arch_height_mm < 20:
        arch_level, arch_score = '정상',    2
    else:
        arch_level, arch_score = '높은 아치', 3

    # 결과 이미지 시각화
    out = img.copy()
    cv2.drawContours(out, [abs_foot], -1, (0, 255, 0), 2)
    # 바닥 기준선 (노란)
    cv2.line(out, (foot_abs_x, floor_y), (foot_abs_x + w, floor_y), (0, 255, 255), 2)
    # 아치 높이 수직선 (하늘색)
    cv2.line(out, (arch_peak[0], arch_peak[1]), (arch_peak[0], floor_y), (255, 180, 0), 3)
    # 텍스트
    mid_y = (arch_peak[1] + floor_y) // 2
    cv2.putText(out, f'{arch_height_mm:.1f}mm ({arch_level})',
                (arch_peak[0] + 8, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)

    _, buf      = cv2.imencode('.jpg', out)
    result_b64  = base64.b64encode(buf).decode('utf-8')

    return {
        'arch_height_mm': round(arch_height_mm, 1),
        'arch_level':     arch_level,
        'arch_score':     arch_score,
        'result_image':   result_b64,
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

    # 발끝(상단) 가로선 - 노란색
    cv2.line(out, (px, by), (px + pw, by), (0, 255, 255), 2)
    # 뒤꿈치(하단) 가로선 - 노란색
    cv2.line(out, (px, by + bh), (px + pw, by + bh), (0, 255, 255), 2)
    # 발볼 왼쪽 세로선 - 노란색
    cv2.line(out, (bx, py), (bx, py + ph), (0, 255, 255), 2)
    # 발볼 오른쪽 세로선 - 노란색
    cv2.line(out, (bx + bw, py), (bx + bw, py + ph), (0, 255, 255), 2)

    # 측정값 텍스트 표시
    cv2.putText(out, f"Length: {result['발 길이 (cm)']}cm",
                (bx, by - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(out, f"Width: {result['발볼 너비 (cm)']}cm",
                (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return out
