import os

from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__)

    import matplotlib
    import matplotlib.style

    matplotlib.use("SVG")
    matplotlib.style.use(os.path.join(os.path.dirname(__file__), "match.mplstyle"))

    import schemdraw

    schemdraw.use("svg")

    from . import match

    app.register_blueprint(match.bp)

    return app
