import os
import re
from unicodedata import normalize

from dotenv import load_dotenv
from flask import Flask
from flask_smorest import Api
from resources import AllBlueprint

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)


OPENSLIDE_PATH = "E://openslide-win64-20230414/bin"
os.add_dll_directory(OPENSLIDE_PATH)
import openslide  # noqa
from openslide import ImageSlide, open_slide  # noqa
from openslide.deepzoom import DeepZoomGenerator  # noqa

UPLOADS_DIR = "./files_uploaded"
RESULTS_DIR = "./files_results"


def slugify(text):
    text = normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return re.sub("[^a-z0-9]+", "-", text)


def create_app(config=None, config_file=None):
    app = Flask(__name__)
    app.config["API_TITLE"] = "Lymphoma REST API"
    app.config["API_VERSION"] = os.environ.get("API_VERSION")
    app.config["OPENAPI_VERSION"] = "3.0.3"
    app.config["OPENAPI_URL_PREFIX"] = "/"
    app.config["OPENAPI_SWAGGER_UI_PATH"] = "/swagger-ui"
    app.config[
        "OPENAPI_SWAGGER_UI_URL"
    ] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"
    app.config["PROPAGATE_EXCEPTIONS"] = True
    app.config.from_mapping(
        DEEPZOOM_SLIDE=None,
        DEEPZOOM_FORMAT="jpeg",
        DEEPZOOM_TILE_SIZE=254,
        DEEPZOOM_OVERLAP=1,
        DEEPZOOM_LIMIT_BOUNDS=True,
        DEEPZOOM_TILE_QUALITY=75,
    )
    app.config.from_envvar("DEEPZOOM_TILER_SETTINGS", silent=True)
    if config_file is not None:
        app.config.from_pyfile(config_file)
    if config is not None:
        app.config.from_mapping(config)

    app.slides = dict()

    api = Api(app)

    api.register_blueprint(AllBlueprint)

    config_map = {
        "DEEPZOOM_TILE_SIZE": "tile_size",
        "DEEPZOOM_OVERLAP": "overlap",
        "DEEPZOOM_LIMIT_BOUNDS": "limit_bounds",
    }
    opts = {v: app.config[k] for k, v in config_map.items()}
    app.slides = dict()
    app.slide_properties = dict()
    app.slide_mpps = dict()
    for filename in os.listdir(UPLOADS_DIR):
        file_path = os.path.join(UPLOADS_DIR, filename)
        try:
            slide = open_slide(file_path)
            id = filename.rsplit(".", 1)[0].lower()
            slugified_id = slugify(id)
            app.slides[slugified_id] = DeepZoomGenerator(slide, **opts)
            try:
                mpp_x = slide.properties[openslide.PROPERTY_NAME_MPP_X]
                mpp_y = slide.properties[openslide.PROPERTY_NAME_MPP_Y]
                app.slide_mpps[slugified_id] = (float(mpp_x) + float(mpp_y)) / 2
            except (KeyError, ValueError):
                app.slide_mpps[slugified_id] = 0
        except Exception as e:
            app.logger.warn(e)
    for filename in os.listdir(RESULTS_DIR):
        if filename.rsplit(".", 1)[1].lower() == "txt":
            continue
        file_path = os.path.join(RESULTS_DIR, filename)
        try:
            slide = open_slide(file_path)
            id = filename.rsplit(".", 1)[0].lower()
            slugified_id = slugify(id)
            app.slides[slugified_id] = DeepZoomGenerator(slide, **opts)
        except Exception as e:
            app.logger.warn(e)

    return app
