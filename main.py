from flask import Flask
from routes import pdf_bp


def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.register_blueprint(pdf_bp)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
