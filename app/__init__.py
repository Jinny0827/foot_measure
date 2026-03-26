from flask import Flask

def create_app():
    """
    Flask 앱 팩토리 함수
    - 앱 인스턴스 생성 및 설정
    - Blueprint 등록
    """
    app = Flask(__name__)

    # routes.py의 Blueprint 등록
    from app.routes import bp
    app.register_blueprint(bp)

    return app
