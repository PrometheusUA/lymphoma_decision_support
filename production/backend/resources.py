import json
import os
from io import BytesIO
from queue import Queue
from threading import Thread

import numpy as np
import PIL
import torch
from flask import current_app, make_response, url_for
from flask.views import MethodView
from flask_smorest import Blueprint, abort
from prediction.dataset import DatasetTest
from prediction.simplenet import SimpleNet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from schemas import FileSchema, MessageIDSchema
from segmentation.segmentation import segment_slide
from torch.utils.data import DataLoader
from tqdm import tqdm

PIL.Image.MAX_IMAGE_PIXELS = 933120000

AllBlueprint = Blueprint("all", "all", description="All resources")

SUBIMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 0
NUM_CLASSES = 3

OPENSLIDE_PATH = "E://openslide-win64-20230414/bin"
os.add_dll_directory(OPENSLIDE_PATH)
import openslide  # noqa
from openslide import ImageSlide, open_slide  # noqa
from openslide.deepzoom import DeepZoomGenerator  # noqa

UPLOADS_DIR = "./files_uploaded"
RESULTS_DIR = "./files_results"

ALLOWED_EXTENSIONS = [
    "svs",  # Aperio
    "tif",  # Aperio, Trestle, Ventana, Generic tiled TIFF
    "ndpi",  # Hamamatsu
    "vms",  # Hamamatsu
    "vmu",  # Hamamatsu
    "scn",  # Leica
    "mrxs",  # MIRAX
    "tiff",  # Philips
    "svslide",  # Sakura
    "bif",  # Ventana
]


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def wsi_gradcam(grayscale_cams, target_class_id, reconstruction_info):
    target_grayscale_cams = grayscale_cams[target_class_id]
    grayscale_cams_list, xs_list, ys_list = zip(*target_grayscale_cams)
    grayscale_cams_np = np.concatenate(grayscale_cams_list)
    xs_np = torch.cat(xs_list).numpy()
    ys_np = torch.cat(ys_list).numpy()

    (
        target_image_height,
        target_image_width,
        count_width,
        count_height,
        d_width,
        d_height,
        subimage_size,
    ) = reconstruction_info

    grayscale_full = np.zeros((target_image_height, target_image_width))
    grayscale_intersections = np.zeros((target_image_height, target_image_width))

    for grayscale_cam, x, y in tqdm(zip(grayscale_cams_np, xs_np, ys_np)):
        # subimage_id = y * count_width + x
        subimage_startx = int(x * d_width)
        subimage_starty = int(y * d_height)
        subimage_endx = int(subimage_startx + subimage_size)
        subimage_endy = int(subimage_starty + subimage_size)

        grayscale_full[subimage_starty:subimage_endy, subimage_startx:subimage_endx] = (
            grayscale_intersections[
                subimage_starty:subimage_endy, subimage_startx:subimage_endx
            ]
            * grayscale_full[
                subimage_starty:subimage_endy, subimage_startx:subimage_endx
            ]
            + grayscale_cam
        ) / (
            grayscale_intersections[
                subimage_starty:subimage_endy, subimage_startx:subimage_endx
            ]
            + 1
        )

        grayscale_intersections[
            subimage_starty:subimage_endy, subimage_startx:subimage_endx
        ] += 1

    return grayscale_full


@AllBlueprint.after_request
def after_request(response):
    header = response.headers
    header["Access-Control-Allow-Origin"] = "*"
    return response


@AllBlueprint.route("/")
class SendFile(MethodView):
    @AllBlueprint.arguments(FileSchema, location="files")
    @AllBlueprint.response(
        200, MessageIDSchema, description="Successfully uploaded and processed"
    )
    @AllBlueprint.alt_response(400, description="Error occured")
    def post(self, file_data):
        current_app.logger.info(file_data["file"].filename)
        file = file_data["file"]
        if file.filename == "" or not allowed_file(file.filename):
            abort(400, message="File not found or wrong format")
        id = np.random.randint(low=1e6, high=1e9)
        file_path = os.path.join(
            UPLOADS_DIR, f"{id}.{file.filename.rsplit('.', 1)[1].lower()}"
        )
        txt_path = os.path.join(RESULTS_DIR, f"{id}.txt")
        while os.path.isfile(txt_path):
            id = np.random.randint(low=1e6, high=1e13)
            txt_path = os.path.join(RESULTS_DIR, f"{id}.txt")
            file_path = os.path.join(
                UPLOADS_DIR, f"{id}.{file.filename.rsplit('.', 1)[1].lower()}"
            )
        file.save(file_path)
        current_app.logger.info(f"{file_data['file'].filename} uploaded")
        try:
            wsi_openslide = openslide.OpenSlide(file_path)
            current_app.logger.info(f"{file_data['file'].filename} opened")
        except Exception as e:
            current_app.logger.info(e)
            abort(400, message="File can't be read by openslide")

        wsi_segmented = segment_slide(file_path)

        if wsi_segmented is None:
            abort(400, message="Something went wrong with segmentation.")

        current_app.logger.info("Segmented successfully")

        wsi_width, wsi_height = wsi_openslide.level_dimensions[1]
        vertical_subims_count = int(np.ceil(wsi_height / SUBIMAGE_SIZE))
        horizontal_subims_count = int(np.ceil(wsi_width / SUBIMAGE_SIZE))
        d_wsi_height = (wsi_height - SUBIMAGE_SIZE) / (vertical_subims_count - 1)
        d_wsi_width = (wsi_width - SUBIMAGE_SIZE) / (horizontal_subims_count - 1)

        segmented_height, segmented_width = wsi_segmented.shape
        subimage_height_segmented = segmented_height / wsi_height * SUBIMAGE_SIZE
        subimage_width_segmented = segmented_width / wsi_width * SUBIMAGE_SIZE

        d_segmented_height = (segmented_height - subimage_height_segmented) / (
            vertical_subims_count - 1
        )
        d_segmented_width = (segmented_width - subimage_width_segmented) / (
            horizontal_subims_count - 1
        )

        t = tqdm(total=horizontal_subims_count * vertical_subims_count)

        def add_to_valid_multiple(pairs, queue):
            for x, y in pairs:
                start_point_seg_y = int(y * d_segmented_height)
                start_point_seg_x = int(x * d_segmented_width)

                segmentation_map = wsi_segmented[
                    int(start_point_seg_y) : int(
                        start_point_seg_y + subimage_height_segmented
                    ),
                    int(start_point_seg_x) : int(
                        start_point_seg_x + subimage_width_segmented
                    ),
                ]

                map_size = int(subimage_height_segmented) * int(
                    subimage_width_segmented
                )
                valid_pixels_count = np.sum(segmentation_map)

                if valid_pixels_count / map_size > 0.1:
                    queue.put((x, y))

                t.update()

        pairs = []
        for x in range(horizontal_subims_count):
            for y in range(vertical_subims_count):
                pairs.append((x, y))

        valid_images = []
        valid_images_q = Queue()

        threads_num = 10
        threads = []

        for thread_i in range(threads_num):
            start_i = int(thread_i / threads_num * len(pairs))
            end_i = (
                len(pairs)
                if thread_i + 1 == threads_num
                else int((thread_i + 1) / threads_num * len(pairs))
            )
            pairs_thread = pairs[start_i:end_i]
            threads.append(
                Thread(
                    target=add_to_valid_multiple, args=(pairs_thread, valid_images_q)
                )
            )

        for i in range(threads_num):
            threads[i].start()

        for i in range(threads_num):
            threads[i].join()
        t.close()

        while not valid_images_q.empty():
            valid_images.append(valid_images_q.get())

        current_app.logger.info("Valid subimages chosen")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleNet()
        model = model.to(device)

        model.load_state_dict(
            torch.load("./prediction/simplenet_1.5.pt", map_location=device)
        )

        dataset = DatasetTest(
            valid_images, wsi_openslide, d_wsi_height, d_wsi_width, SUBIMAGE_SIZE
        )
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

        target_layers = [model.conv3]
        cam = GradCAM(
            model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available()
        )

        grayscale_cams = [[], [], []]

        current_app.logger.info("Ready for gradcam and prediction")

        outputs = []
        for images, xs, ys in tqdm(dataloader):
            with torch.no_grad():
                images = images.to(device)
                output = model(images)
                outputs.append(output.cpu().detach())
            for class_id in range(3):
                targets = [
                    ClassifierOutputTarget(class_id) for _ in range(images.size()[0])
                ]
                grayscale_cams_batch = cam(input_tensor=images, targets=targets)
                grayscale_cams[class_id].append((grayscale_cams_batch, xs, ys))

        current_app.logger.info("Subimages predictions successfull")

        outputs = torch.cat(outputs)

        outputs_means = torch.mean(outputs, dim=0)
        probs = torch.softmax(outputs_means, dim=0)

        with open(txt_path, "w") as file:
            file.write(str(probs.tolist()))

        current_app.logger.info("Predictions saved")

        for class_idx in range(NUM_CLASSES):
            wsi_openslide_pil = wsi_openslide.read_region(
                (0, 0), 2, wsi_openslide.level_dimensions[2]
            )
            wsi_openslide_pil = wsi_openslide_pil.convert(mode="RGB")
            grayscale_full = wsi_gradcam(
                grayscale_cams,
                class_idx,
                (
                    wsi_height,
                    wsi_width,
                    vertical_subims_count,
                    horizontal_subims_count,
                    d_wsi_height,
                    d_wsi_width,
                    SUBIMAGE_SIZE,
                ),
            )

            wsi_segmented_reshaped = np.array(
                PIL.Image.fromarray(wsi_segmented.astype(np.int8), mode="L").resize(
                    (grayscale_full.shape[1], grayscale_full.shape[0])
                )
            )
            grayscale_sharpen = (grayscale_full * wsi_segmented_reshaped * 255).astype(
                np.int8
            )

            grayscale_image = np.array(
                PIL.Image.fromarray(grayscale_sharpen, mode="L").resize(
                    wsi_openslide.level_dimensions[2]
                )
            )

            mapped_wsi = show_cam_on_image(
                np.array(wsi_openslide_pil).astype(np.float64) / 255.0,
                grayscale_image,
                use_rgb=True,
            )

            output_path = os.path.join(RESULTS_DIR, f"{id}_{class_idx}.tiff")
            PIL.Image.fromarray(mapped_wsi, mode="RGB").save(output_path)

            current_app.logger.info(f"GradCam for class {class_idx} saved")

        return {"message": "Файл оброблено", "id": id}


@AllBlueprint.route("/<id>", methods=["GET"])
def getById(id):
    if id not in current_app.slides:
        abort(404, message="Case with this id wasn't found")

    text_path = os.path.join(RESULTS_DIR, f"{id}.txt")
    if not os.path.isfile(text_path):
        abort(400, message="Results not found")

    with open(text_path) as file:
        resultObj = json.loads(file.read())
    current_app.logger.info(current_app.slides)
    map_urls = [url_for("all.dzi", slug=f"{id}-{i}") for i in range(NUM_CLASSES)]
    return {
        "slide_url": url_for("all.dzi", slug=id),
        "map_urls": map_urls,
        "slide_mpp": current_app.slide_mpps[id],
        "class_probs": resultObj,
    }, 200


@AllBlueprint.route("/deepzoom/<slug>.dzi", methods=["GET"])
def dzi(slug):
    
    if slug not in current_app.slides:
        abort(404, message="File with this slug wasn't found")

    format = current_app.config["DEEPZOOM_FORMAT"]

    resp = make_response(current_app.slides[slug].get_dzi(format))
    resp.mimetype = "application/xml"
    return resp


@AllBlueprint.route("/deepzoom/<slug>_files/<int:level>/<int:col>_<int:row>.<format>")
def tile(slug, level, col, row, format):
    format = format.lower()
    if format != "jpeg" and format != "png":
        abort(404, message="Format not supported")
    if slug not in current_app.slides:
        abort(404, message="File with this slug wasn't found")
    try:
        tile = current_app.slides[slug].get_tile(level, (col, row))
    except ValueError:
        abort(404, message="Invalid level coordinates.")
    buf = BytesIO()
    tile.save(buf, format, quality=current_app.config["DEEPZOOM_TILE_QUALITY"])
    resp = make_response(buf.getvalue())
    resp.mimetype = "image/%s" % format
    return resp
