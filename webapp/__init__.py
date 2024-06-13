import os
import secrets

from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix


def create_app() -> Flask:
    import matplotlib
    import matplotlib.style

    matplotlib.use("SVG")
    matplotlib.style.use(os.path.join(os.path.dirname(__file__), "match.mplstyle"))

    import schemdraw

    schemdraw.use("svg")

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1000 * 1000
    app.secret_key = secrets.token_hex(32)

    from . import match

    app.register_blueprint(match.bp)

    return app


def create_proxied_app() -> Flask:
    app = create_app()
    app.wsgi_app = ProxyFix(  # type: ignore[method-assign]
        app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1
    )
    return app
