import os

from flask import Flask, render_template, request, url_for, redirect

from flaskr.lemmatizer import lemma_index, language_index, nltk_stopwords

lemmatizers = {}
for key in language_index:
    if key in nltk_stopwords.keys():
        lemmatizers[key] = lemma_index(key)

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return render_template('index.html', name="Arturo")

    @app.route('/form', methods=['POST', 'GET'])
    def bio_data_form():
        if request.method == "POST":
            username = request.form['username']
            age = request.form['age']
            email = request.form['email']
            hobbies = request.form['hobbies']
            return redirect(url_for('showbio',
                                    username=username,
                                    age=age,
                                    email=email,
                                    hobbies=hobbies))
        return render_template("form.html")

    @app.route('/showbio', methods=['GET'])
    def showbio():
        username = request.args.get('username')
        age = request.args.get('age')
        email = request.args.get('email')
        hobbies = request.args.get('hobbies')
        return render_template("show_bio.html",
                               username=username,
                               age=age,
                               email=email,
                               hobbies=hobbies
                               )

    return app
