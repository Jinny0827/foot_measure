# 👟 foot_measure — 발 측정 기반 신발 추천 서비스

스마트폰 카메라 2장(위 + 옆)으로 발 형태를 분석하고, 신발 사이즈 및 타입을 추천하는 웹 서비스입니다.

🌐 **서비스 URL**: [budget.bowling-manager.com/foot-measure](https://budget.bowling-manager.com/foot-measure)

> ⚠️ 본 서비스는 참고용 측정 도구입니다. 의학적 진단을 목적으로 하지 않습니다.

---

## 📌 주요 기능

### Step 1 — 위에서 촬영 (발 길이 / 발볼)
- A4 용지 실제 감지 (HSV 흰색 마스크 + 비율 검증)
- 흰 바닥 환경 자동 구별 → 원인별 에러 메시지 반환
- 발 길이 / 발볼 너비 측정 (mm 단위)
- 결과 이미지 시각화 (측정선 포함)

### Step 2 — 옆에서 촬영 (아치)
- 피부색 감지 기반 발 인식 → 셔터 자동 활성화
- 아치 높이 측정 (mm)
- 평발 등급 분류: 평발 / 저아치 / 정상 / 높은 아치

### 결과 화면
- 위 / 옆 사진 나란히 표시
- 신발 사이즈 추천 (KR mm 기준)
- 발볼 핏 분류: 슬림 / 보통 / 와이드
- 아치 등급별 신발 타입 + 인솔 추천

---

## 🏗️ 배포 구조

```
프론트 (React)
└─ S3 + CloudFront
      budget.bowling-manager.com/foot-measure

백엔드 (Flask)
└─ API Gateway
      └─ Lambda (python3.11 / 1024MB / 60s)
            ├─ POST /measure        위에서 촬영
            └─ POST /measure/side   옆에서 촬영
```

---

## 🛠️ 기술 스택

| 영역 | 기술 |
|---|---|
| 백엔드 | Python + Flask + OpenCV |
| 이미지 처리 | OpenCV (HSV 마스킹, A4 감지, 형태학적 연산) |
| 프론트엔드 | React + TypeScript + Vite |
| 백엔드 배포 | AWS Lambda (SAM CLI) + API Gateway |
| 프론트 배포 | S3 + CloudFront |

---

## 📁 프로젝트 구조

```
foot_measure/                    # 백엔드
├─ app/
│   ├─ measure.py               # 핵심 이미지 처리 로직
│   └─ routes.py                # API 엔드포인트
├─ main.py                      # Lambda 핸들러 (aws-wsgi)
├─ requirements.txt
└─ template.yaml                # SAM 배포 설정
```

---

## 📡 API 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | /health | 서버 상태 확인 |
| POST | /measure | 위에서 촬영 — 발 길이 / 발볼 측정 |
| POST | /measure/side | 옆에서 촬영 — 아치 높이 측정 |

### POST /measure 응답
```json
{
  "발 길이 (mm)": 255.3,
  "발 길이 (cm)": 25.5,
  "발볼 너비 (mm)": 98.1,
  "발볼 너비 (cm)": 9.8,
  "result_image": "base64 JPEG..."
}
```

### POST /measure/side 응답
```json
{
  "arch_height_mm": 18.4,
  "arch_level": "정상",
  "arch_score": 2,
  "result_image": "base64 JPEG..."
}
```

---

## 🔬 핵심 알고리즘

### A4 감지 알고리즘
1. 가이드박스 ROI 추출 (±여유 20px)
2. HSV 흰색 마스크 (V > 175, S < 60)
3. MORPH_CLOSE (25×25, 3회) — 발이 덮은 구멍 채우기
4. MORPH_OPEN (9×9, 1회) — 노이즈 제거
5. white_ratio > 78% → 흰 바닥 에러
6. 면적 5% 이상 흰색 영역 수집 → minAreaRect
7. 비율 검증 (210:297 ± 25%)
8. pixel_per_mm 계산

### 신발 추천 로직 (프론트 자체 계산)
```
발볼 비율 = 발볼너비 / 발길이
  < 0.37      → 슬림
  0.37~0.41   → 보통
  > 0.41      → 와이드

신발 사이즈 = ceil((발길이 + 10mm) / 5) × 5

아치 등급별 추천:
  평발    → 모션 컨트롤 슈즈 + 높은 아치 지지 인솔
  저아치  → 스태빌리티 슈즈 + 미디엄 아치 지지 인솔
  정상    → 뉴트럴 슈즈 + 기본 쿠셔닝 인솔
  높은아치 → 쿠셔닝 슈즈 + 충격 흡수 인솔
```

---

## 🚀 로컬 실행

### 백엔드
```bash
cd foot_measure
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python main.py
```

### 배포
```bash
# 백엔드 (SAM)
sam build
sam deploy

# 프론트엔드
.\deploy.ps1 frontend
```

---

## 🔑 환경변수

```env
# 프론트엔드
VITE_FOOT_API_URL=http://127.0.0.1:5000/measure   # 로컬
```

---

## 🚧 TODO

- [ ] RunRepeat 크롤러 (Playwright — 신발 DB 수집, 주 1회 배치)
- [ ] `/recommend` 매칭 API (아치 등급 + 발볼 비율 기반 상위 10개 반환)
- [ ] 원근 보정 (4점 Perspective Transform) — 발 길이 오차 개선
- [ ] 회원 시스템 (측정 이력 저장)
- [ ] A4 감지 실패 시 Claude Vision API fallback
