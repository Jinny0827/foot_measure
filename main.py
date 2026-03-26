from app import create_app

#Flask 앱 인스턴스 생성
app = create_app()

if __name__ == '__main__':
    """
    로컬 실행 진입점
    - debug=True: 코드 변경 시 자동 재시작, 에러 상세 출력
    - port=5000: 기본 포트 (변경 가능)
    """

    app.run(debug=True, port=5000)