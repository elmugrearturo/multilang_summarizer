from flask import Flask

app = Flask(__name__)


@app.route('/')
def flask_service():
    return 'Hello, World!'
