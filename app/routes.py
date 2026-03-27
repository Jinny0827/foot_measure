from flask import Blueprint, request, jsonify
import cv2
import os
import base64
from app.measure import (
    load_image,
    preprocess,
    get_pixel_per_mm,
    find_foot_contour,
    measure_foot,
    draw_result,
    measure_arch_side,
)

# Blueprint: Flask 라우트를 모듈별로 분리하기 위한 단위
bp = Blueprint('main', __name__)

@bp.route('/health', methods = ['GET'])
def health():
    """
    서버 상태 확인용 엔드포인트
    - 배포 후 서버가 정상 동작하는지 확인할 때 사용
    """
    return jsonify({"status" : "ok"}), 200

@bp.route('/measure', methods=['POST'])
def measure():
    """
    발 측정 메인 엔드포인트
    - 요청: multipart/form-data 형식으로 이미지 + 가이드 박스 좌표 전송
      - image: 이미지 파일
      - paper_x, paper_y, paper_w, paper_h: 프론트 가이드 박스 좌표 (픽셀)
    - 응답: 발 길이 / 발볼 너비 JSON 반환
    """
    if 'image' not in request.files:
        return jsonify({"error": "이미지 파일이 없습니다."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "파일명이 비어있습니다."}), 400

    # 가이드 박스 좌표 파싱
    try:
        paper_x = int(request.form.get('paper_x', 0))
        paper_y = int(request.form.get('paper_y', 0))
        paper_w = int(request.form.get('paper_w', 0))
        paper_h = int(request.form.get('paper_h', 0))
    except ValueError:
        return jsonify({"error": "가이드 박스 좌표가 올바르지 않습니다."}), 400

    if paper_w == 0 or paper_h == 0:
        return jsonify({"error": "가이드 박스 좌표가 필요합니다. (paper_x, paper_y, paper_w, paper_h)"}), 400

    paper_bbox = (paper_x, paper_y, paper_w, paper_h)

    # Lambda는 /tmp/만 쓰기 가능, 로컬은 data/ 사용
    tmp_dir = '/tmp' if os.path.exists('/tmp') else 'data'
    save_path = os.path.join(tmp_dir, file.filename)
    file.save(save_path)

    try:
        # 측정 파이프라인 실행
        img = load_image(save_path)
        preprocessed = preprocess(img)

        # 가이드 박스 = A4 기준 (프론트에서 정확한 A4 비율로 계산됨)
        px_per_mm_x, px_per_mm_y, _ = get_pixel_per_mm(paper_bbox)

        foot, roi_offset = find_foot_contour(preprocessed, paper_bbox)
        result = measure_foot(foot, roi_offset, px_per_mm_x, px_per_mm_y)

        # 결과 이미지 저장 (시각화용)
        output_img = draw_result(img, paper_bbox, result)
        output_path = os.path.join(tmp_dir, 'result.jpg')
        cv2.imwrite(output_path, output_img)

        # 결과 이미지 base64 인코딩
        result_image_b64 = None
        if os.path.exists(output_path):
            with open(output_path, 'rb') as f:
                result_image_b64 = base64.b64encode(f.read()).decode('utf-8')

        # bounding_box는 내부 데이터라 응답에서 제외
        response = {
            "발 길이 (mm)": result["발 길이 (mm)"],
            "발볼 너비 (mm)": result["발볼 너비 (mm)"],
            "발 길이 (cm)": result["발 길이 (cm)"],
            "발볼 너비 (cm)": result["발볼 너비 (cm)"],
            "result_image": result_image_b64
        }

        return jsonify(response), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    except Exception as e:
        return jsonify({"error": f"처리 중 오류 발생: {str(e)}"}), 500

    finally:
        import gc
        gc.collect()
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except PermissionError:
                pass


@bp.route('/measure/side', methods=['POST'])
def measure_side():
    """
    발 옆면 촬영 - 아치 높이 / 평발 수준 측정
    - image: 이미지 파일
    - paper_x/y/w/h: 프론트 가이드 박스 좌표
    - foot_length_mm: Step1에서 측정한 발 길이 (스케일 기준)
    """
    if 'image' not in request.files:
        return jsonify({"error": "이미지 파일이 없습니다."}), 400

    file = request.files['image']

    try:
        paper_x       = int(request.form.get('paper_x', 0))
        paper_y       = int(request.form.get('paper_y', 0))
        paper_w       = int(request.form.get('paper_w', 0))
        paper_h       = int(request.form.get('paper_h', 0))
        foot_length_mm = float(request.form.get('foot_length_mm', 0))
    except ValueError:
        return jsonify({"error": "파라미터가 올바르지 않습니다."}), 400

    if paper_w == 0 or paper_h == 0:
        return jsonify({"error": "가이드 박스 좌표가 필요합니다."}), 400

    tmp_dir   = '/tmp' if os.path.exists('/tmp') else 'data'
    fname     = file.filename or 'side.jpg'
    save_path = os.path.join(tmp_dir, fname)
    file.save(save_path)

    try:
        result = measure_arch_side(
            save_path, paper_x, paper_y, paper_w, paper_h, foot_length_mm
        )
        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    except Exception as e:
        return jsonify({"error": f"처리 중 오류 발생: {str(e)}"}), 500

    finally:
        import gc
        gc.collect()
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except PermissionError:
                pass
