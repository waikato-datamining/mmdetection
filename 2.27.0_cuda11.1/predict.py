import numpy as np
import os
import argparse
from datetime import datetime
from PIL import Image
from image_complete import auto
import torch
import traceback

import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.datasets.dataset import determine_classes
from opex import ObjectPredictions, ObjectPrediction, BBox, Polygon
from sfp import Poller
from wai.annotations.image_utils import image_to_numpyarray, remove_alpha_channel, mask_to_polygon, polygon_to_minrect, polygon_to_lists, lists_to_polygon, polygon_to_bbox
from wai.annotations.core import ImageInfo
from wai.annotations.roi import ROIObject
from wai.annotations.roi.io import ROIWriter

SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
""" supported file extensions (lower case). """

OUTPUT_ROIS = "rois"
OUTPUT_OPEX = "opex"
OUTPUT_FORMATS = [OUTPUT_ROIS, OUTPUT_OPEX]


def check_image(fname, poller):
    """
    Check method that ensures the image is valid.

    :param fname: the file to check
    :type fname: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: True if complete
    :rtype: bool
    """
    result = auto.is_image_complete(fname)
    poller.debug("Image complete:", fname, "->", result)
    return result


def process_image(fname, output_dir, poller):
    """
    Method for processing an image.

    :param fname: the image to process
    :type fname: str
    :param output_dir: the directory to write the image to
    :type output_dir: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: the list of generated output files
    :rtype: list
    """
    result = []

    try:
        image = Image.open(fname)
        image = remove_alpha_channel(image)
        image_array = image_to_numpyarray(image)
        detection = inference_detector(model, image_array)

        assert isinstance(poller.params.class_names, (tuple, list))
        if isinstance(detection, tuple):
            bbox_result, segm_result = detection
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = detection, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        output_path = "{}/{}{}".format(output_dir, os.path.splitext(os.path.basename(fname))[0], poller.params.suffix)
        img_path = "{}/{}-mask.png".format(output_dir, os.path.splitext(os.path.basename(fname))[0])

        # predictions
        pred_objs = []
        mask_comb = None
        masks = None
        if segm_result is not None:
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                masks = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                masks = np.stack(segms, axis=0)

        for index in range(len(bboxes)):
            x0, y0, x1, y1, score = bboxes[index]
            label = labels[index]
            label_str = poller.params.class_names[label]

            # Ignore this roi if the score is less than the provided threshold
            if score < poller.params.score_threshold:
                continue

            # Translate roi coordinates into original image coordinates (before combining)
            x0n = x0 / image.width
            y0n = y0 / image.height
            x1n = x1 / image.width
            y1n = y1 / image.height

            px = None
            py = None
            pxn = None
            pyn = None
            bw = None
            bh = None

            if segm_result is not None:
                px = []
                py = []
                pxn = []
                pyn = []
                bw = ""
                bh = ""

                mask = masks[index].astype(bool)
                poly = mask_to_polygon(mask, poller.params.mask_threshold, mask_nth=poller.params.mask_nth, view=(x0, y0, x1, y1),
                                       view_margin=poller.params.view_margin, fully_connected=poller.params.fully_connected)
                if len(poly) > 0:
                    px, py = polygon_to_lists(poly[0], swap_x_y=True, normalize=False)
                    pxn, pyn = polygon_to_lists(poly[0], swap_x_y=True, normalize=True, img_width=image.width, img_height=image.height)
                    if poller.params.output_minrect:
                        bw, bh = polygon_to_minrect(poly[0])
                    if poller.params.bbox_as_fallback >= 0:
                        if len(px) >= 3:
                            p_x0n, p_y0n, p_x1n, p_y1n = polygon_to_bbox(lists_to_polygon(pxn, pyn))
                            p_area = (p_x1n - p_x0n) * (p_y1n - p_y0n)
                            b_area = (x1n - x0n) * (y1n - y0n)
                            if (b_area > 0) and (p_area / b_area < poller.params.bbox_as_fallback):
                                px = [float(i) for i in [x0, x1, x1, x0]]
                                py = [float(i) for i in [y0, y0, y1, y1]]
                                pxn = [float(i) for i in [x0n, x1n, x1n, x0n]]
                                pyn = [float(i) for i in [y0n, y0n, y1n, y1n]]
                        else:
                            px = [float(i) for i in [x0, x1, x1, x0]]
                            py = [float(i) for i in [y0, y0, y1, y1]]
                            pxn = [float(i) for i in [x0n, x1n, x1n, x0n]]
                            pyn = [float(i) for i in [y0n, y0n, y1n, y1n]]
                        if poller.params.output_minrect:
                            bw = x1 - x0 + 1
                            bh = y1 - y0 + 1
                    if poller.params.fit_bbox_to_polygon:
                        if len(px) >= 3:
                            x0, y0, x1, y1 = polygon_to_bbox(lists_to_polygon(px, py))
                            x0n, y0n, x1n, y1n = polygon_to_bbox(lists_to_polygon(pxn, pyn))

                if poller.params.output_mask_image:
                    mask_img = mask.copy()
                    mask_img[mask_img < poller.params.mask_threshold] = 0
                    mask_img[mask_img >= poller.params.mask_threshold] = label+1  # first label is 0
                    if mask_comb is None:
                        mask_comb = mask_img
                    else:
                        tmp = np.where(mask_comb == 0, mask_img, mask_comb)
                        mask_comb = tmp

            if poller.params.output_format == OUTPUT_ROIS:
                roi_obj = ROIObject(x0, y0, x1, y1, x0n, y0n, x1n, y1n, label, label_str, score=score,
                                    poly_x=px, poly_y=py, poly_xn=pxn, poly_yn=pyn,
                                    minrect_w=bw, minrect_h=bh)
                pred_objs.append(roi_obj)
            elif poller.params.output_format == OUTPUT_OPEX:
                bbox = BBox(left=int(x0), top=int(y0), right=int(x1), bottom=int(y1))
                points = []
                for x, y in zip(px, py):
                    points.append((int(x), int(y)))
                poly = Polygon(points=points)
                opex_pred = ObjectPrediction(score=score, label=label, bbox=bbox, polygon=poly)
                pred_objs.append(opex_pred)
            else:
                poller.error("Unknown output format: %s" % poller.params.output_format)

        if poller.params.output_format == OUTPUT_ROIS:
            info = ImageInfo(os.path.basename(fname))
            roi_ext = (info, pred_objs)
            options = ["--output", output_dir, "--no-images"]
            if poller.params.output_width_height:
                options.append("--size-mode")
            roi_writer = ROIWriter(options)
            roi_writer.save([roi_ext])
        elif poller.params.output_format == OUTPUT_OPEX:
            opex_preds = ObjectPredictions(id=id, timestamp=str(datetime.now()), objects=pred_objs)
            opex_preds.save_json_to_file(output_path)
        else:
            poller.error("Unknown output format: %s" % poller.params.output_format)

        result.append(output_path)

        if mask_comb is not None:
            im = Image.fromarray(np.uint8(mask_comb), 'P')
            im.save(img_path, "PNG")
            result.append(img_path)
    except KeyboardInterrupt:
        poller.keyboard_interrupt()
    except:
        poller.error("Failed to process image: %s\n%s" % (fname, traceback.format_exc()))
    return result


def predict_on_images(input_dir, model, output_dir, tmp_dir, class_names, score_threshold=0.0,
                      poll_wait=1.0, continuous=False, use_watchdog=False, watchdog_check_interval=10.0,
                      delete_input=False, mask_threshold=0.1, mask_nth=1, output_format=OUTPUT_ROIS, suffix="-rois.csv",
                      output_minrect=False, view_margin=2, fully_connected='high', fit_bbox_to_polygon=False,
                      output_width_height=False, bbox_as_fallback=-1.0, output_mask_image=False,
                      verbose=False, quiet=False):
    """
    Method for performing predictions on images.

    :param input_dir: the directory with the images
    :type input_dir: str
    :param model: the mmdetection trained model
    :param output_dir: the output directory to move the images to and store the predictions
    :type output_dir: str
    :param tmp_dir: the temporary directory to store the predictions until finished, use None if not to use
    :type tmp_dir: str
    :param class_names: labels or class names
    :type class_names: list[str]
    :param score_threshold: the minimum score predictions have to have
    :type score_threshold: float
    :param poll_wait: the amount of seconds between polls when not in watchdog mode
    :type poll_wait: float
    :param continuous: whether to poll continuously
    :type continuous: bool
    :param use_watchdog: whether to react to file creation events rather than use fixed-interval polling
    :type use_watchdog: bool
    :param watchdog_check_interval: the interval for the watchdog process to check for files that were missed due to potential race conditions
    :type watchdog_check_interval: float
    :param delete_input: whether to delete the input images rather than moving them to the output directory
    :type delete_input: bool
    :param mask_threshold: the threshold to use for determining the contour of a mask
    :type mask_threshold: float
    :param mask_nth: to speed up polygon computation, use only every nth row and column from mask
    :type mask_nth: int
    :param output_format: the output format to generate (see OUTPUT_FORMATS)
    :type output_format: str
    :param suffix: the suffix to use for the prediction files, incl extension
    :type suffix: str
    :param output_minrect: when predicting polygons, whether to output the minimal rectangles around the objects as well
    :type output_minrect: bool
    :param view_margin: the margin in pixels to use around the masks
    :type view_margin: int
    :param fully_connected: whether regions of 'high' or 'low' values should be fully-connected at isthmuses
    :type fully_connected: str
    :param fit_bbox_to_polygon: whether to fit the bounding box to the polygon
    :type fit_bbox_to_polygon: bool
    :param output_width_height: whether to output x/y/w/h instead of x0/y0/x1/y1
    :type output_width_height: bool
    :param bbox_as_fallback: if ratio between polygon-bbox and bbox is smaller than this value, use bbox as fallback polygon, ignored if < 0
    :type bbox_as_fallback: float
    :param output_mask_image: when generating masks, whether to output a combined mask image as well
    :type output_mask_image: bool
    :param verbose: whether to output more logging information
    :type verbose: bool
    :param quiet: whether to suppress output
    :type quiet: bool
    """

    if fit_bbox_to_polygon and (bbox_as_fallback >= 0):
        raise Exception("fit_bbox_to_polygon and bbox_as_fallback cannot be used together!")

    poller = Poller()
    poller.model = model
    poller.input_dir = input_dir
    poller.output_dir = output_dir
    poller.tmp_dir = tmp_dir
    poller.extensions = SUPPORTED_EXTS
    poller.delete_input = delete_input
    poller.progress = not quiet
    poller.verbose = verbose
    poller.check_file = check_image
    poller.process_file = process_image
    poller.poll_wait = poll_wait
    poller.continuous = continuous
    poller.use_watchdog = use_watchdog
    poller.watchdog_check_interval = watchdog_check_interval
    poller.params.output_format = output_format
    poller.params.suffix = suffix
    poller.params.class_names = class_names
    poller.params.score_threshold = score_threshold
    poller.params.mask_threshold = mask_threshold
    poller.params.mask_nth = mask_nth
    poller.params.view_margin = view_margin
    poller.params.fully_connected = fully_connected
    poller.params.output_minrect = output_minrect
    poller.params.bbox_as_fallback = bbox_as_fallback
    poller.params.fit_bbox_to_polygon = fit_bbox_to_polygon
    poller.params.output_mask_image = output_mask_image
    poller.params.output_width_height = output_width_height
    poller.poll()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='Path to the trained model checkpoint', required=True, default=None)
    parser.add_argument('--config', help='Path to the config file', required=True, default=None)
    parser.add_argument('--device', help='The CUDA device to use', default="cuda:0")
    parser.add_argument('--prediction_in', help='Path to the test images', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the output csv files folder', required=True, default=None)
    parser.add_argument('--prediction_tmp', help='Path to the temporary csv files folder', required=False, default=None)
    parser.add_argument('--prediction_format', choices=OUTPUT_FORMATS, help='The type of output format to generate', default=OUTPUT_ROIS, required=False)
    parser.add_argument('--prediction_suffix', metavar='SUFFIX', help='The suffix to use for the prediction files', default="-rois.csv", required=False)
    parser.add_argument('--labels', help='ignored, use MMDET_CLASSES environment variable', required=False, default=None)
    parser.add_argument('--score', type=float, help='Score threshold to include in csv file', required=False, default=0.0)
    parser.add_argument('--mask_threshold', type=float, help='The threshold (0-1) to use for determining the contour of a mask', required=False, default=0.1)
    parser.add_argument('--mask_nth', type=int, help='To speed polygon detection up, use every nth row and column only', required=False, default=1)
    parser.add_argument('--output_minrect', action='store_true', help='When outputting polygons whether to store the minimal rectangle around the objects in the CSV files as well', required=False, default=False)
    parser.add_argument('--fit_bbox_to_polygon', action='store_true', help='Whether to fit the bounding box to the polygon', required=False, default=False)
    parser.add_argument('--bbox_as_fallback', default=-1.0, type=float,
                        help='When outputting polygons the bbox can be used as fallback polygon. This happens if the ratio '
                             + 'between the surrounding bbox of the polygon and the bbox is smaller than the specified value. '
                             + 'Turned off if < 0.', required=False)
    parser.add_argument('--poll_wait', type=float, help='poll interval in seconds when not using watchdog mode', required=False, default=1.0)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--use_watchdog', action='store_true', help='Whether to react to file creation events rather than performing fixed-interval polling', required=False, default=False)
    parser.add_argument('--watchdog_check_interval', type=float, help='check interval in seconds for the watchdog', required=False, default=10.0)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--view_margin', default=2, type=int, required=False, help='The number of pixels to use as margin around the masks when determining the polygon')
    parser.add_argument('--fully_connected', default='high', choices=['high', 'low'], required=False, help='When determining polygons, whether regions of high or low values should be fully-connected at isthmuses')
    parser.add_argument('--output_width_height', action='store_true', help="Whether to output x/y/w/h instead of x0/y0/x1/y1 in the ROI CSV files", required=False, default=False)
    parser.add_argument('--output_mask_image', action='store_true', help="Whether to output a mask image (PNG) when predictions generate masks", required=False, default=False)
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parser.add_argument('--quiet', action='store_true', help='Whether to suppress output', required=False, default=False)
    parsed = parser.parse_args()

    if parsed.fit_bbox_to_polygon and (parsed.bbox_as_fallback >= 0):
        raise Exception("Options --fit_bbox_to_polygon and --bbox_as_fallback cannot be used together!")

    try:
        # This is the actual model that is used for the object detection
        model = init_detector(parsed.config, parsed.checkpoint, device=parsed.device)
        
        # Get class names
        class_names = determine_classes()

        # Performing the prediction and producing the csv files
        predict_on_images(parsed.prediction_in, model, parsed.prediction_out, parsed.prediction_tmp, class_names,
                          score_threshold=parsed.score, continuous=parsed.continuous,
                          use_watchdog=parsed.use_watchdog, watchdog_check_interval=parsed.watchdog_check_interval,
                          delete_input=parsed.delete_input,
                          mask_threshold=parsed.mask_threshold, mask_nth=parsed.mask_nth,
                          output_format=parsed.prediction_format, suffix=parsed.prediction_suffix,
                          output_minrect=parsed.output_minrect, view_margin=parsed.view_margin,
                          fully_connected=parsed.fully_connected, fit_bbox_to_polygon=parsed.fit_bbox_to_polygon,
                          output_width_height=parsed.output_width_height, bbox_as_fallback=parsed.bbox_as_fallback,
                          output_mask_image=parsed.output_mask_image, verbose=parsed.verbose, quiet=parsed.quiet)

    except Exception as e:
        print(traceback.format_exc())
