from datetime import datetime
import io
from PIL import Image
import torch
import traceback

from mmdet.apis import init_detector, inference_detector
from mmdet.datasets.ext_dataset import determine_classes
from mmdet.structures.mask import BitmapMasks, PolygonMasks, bitmap_to_polygon
from wai.annotations.image_utils import image_to_numpyarray, remove_alpha_channel, polygon_to_minrect, polygon_to_lists, lists_to_polygon, polygon_to_bbox
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from opex import ObjectPredictions, ObjectPrediction, Polygon, BBox


def process_image(msg_cont):
    """
    Processes the message container, loading the image from the message and forwarding the predictions.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config

    try:
        start_time = datetime.now()
        image = Image.open(io.BytesIO(msg_cont.message['data']))
        image = remove_alpha_channel(image)
        image_array = image_to_numpyarray(image)
        detection = inference_detector(model, image_array)
        if 'pred_instances' in detection:
            pred_instances = detection.pred_instances
            pred_instances = pred_instances[pred_instances.scores > config.score_threshold]
        else:
            log("No 'pred_instances' in detection data for: %s" % str(start_time))
            return

        assert isinstance(config.class_names, (tuple, list))
        labels = pred_instances.labels
        scores = pred_instances.scores
        bboxes = None
        masks = None
        if 'bboxes' in pred_instances:
            bboxes = pred_instances.bboxes
        if 'masks' in pred_instances:
            masks = pred_instances.masks
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            elif isinstance(masks, (PolygonMasks, BitmapMasks)):
                masks = masks.to_ndarray()
            masks = masks.astype(bool)

        objs = []
        for index in range(len(bboxes)):
            x0, y0, x1, y1, score = bboxes[index]
            x0 = float(x0)
            y0 = float(y0)
            x1 = float(x1)
            y1 = float(y1)
            score = float(scores[index])
            label = int(labels[index])
            label_str = config.class_names[label]

            # Translate roi coordinates into original image coordinates (before combining)
            x0n = x0 / image.width
            y0n = y0 / image.height
            x1n = x1 / image.width
            y1n = y1 / image.height

            px = None
            py = None

            if masks is not None:
                px = []
                py = []
                poly, _ = bitmap_to_polygon(masks[index])
                if len(poly) > 0:
                    px, py = polygon_to_lists(poly[0], swap_x_y=False, normalize=False)
                    pxn, pyn = polygon_to_lists(poly[0], swap_x_y=False, normalize=True, img_width=image.width, img_height=image.height)
                    if config.bbox_as_fallback >= 0:
                        if len(px) >= 3:
                            p_x0n, p_y0n, p_x1n, p_y1n = polygon_to_bbox(lists_to_polygon(pxn, pyn))
                            p_area = (p_x1n - p_x0n) * (p_y1n - p_y0n)
                            b_area = (x1n - x0n) * (y1n - y0n)
                            if (b_area > 0) and (p_area / b_area < config.bbox_as_fallback):
                                px = [float(i) for i in [x0, x1, x1, x0]]
                                py = [float(i) for i in [y0, y0, y1, y1]]
                        else:
                            px = [float(i) for i in [x0, x1, x1, x0]]
                            py = [float(i) for i in [y0, y0, y1, y1]]
                    if config.fit_bbox_to_polygon:
                        if len(px) >= 3:
                            x0, y0, x1, y1 = polygon_to_bbox(lists_to_polygon(px, py))

            bbox = BBox(left=int(x0), top=int(y0), right=int(x1), bottom=int(y1))
            p = []
            if px is None:
                px = [x0, x1, x1, x0]
                py = [y0, y0, y1, y1]
            for i in range(len(px)):
                p.append([int(px[i]), int(py[i])])
            poly = Polygon(points=p)
            pred = ObjectPrediction(label=label_str, score=float(score), bbox=bbox, polygon=poly)
            objs.append(pred)

        preds = ObjectPredictions(id=str(start_time), timestamp=str(start_time), objects=objs)
        msg_cont.params.redis.publish(msg_cont.params.channel_out, preds.to_json_string())
        if config.verbose:
            log("process_images - predictions string published: %s" % msg_cont.params.channel_out)
            end_time = datetime.now()
            processing_time = end_time - start_time
            processing_time = int(processing_time.total_seconds() * 1000)
            log("process_images - finished processing image: %d ms" % processing_time)

    except KeyboardInterrupt:
        msg_cont.params.stopped = True
    except:
        log("process_images - failed to process: %s" % traceback.format_exc())


if __name__ == '__main__':
    parser = create_parser('MMDetection - Prediction (Redis)', prog="mmdet_predict_redis", prefix="redis_")
    parser.add_argument('--checkpoint', help='Path to the trained model checkpoint', required=True, default=None)
    parser.add_argument('--config', help='Path to the config file', required=True, default=None)
    parser.add_argument('--device', help='The CUDA device to use', default="cuda:0")
    parser.add_argument('--score', type=float, help='Score threshold to include in csv file', required=False, default=0.0)
    parser.add_argument('--fit_bbox_to_polygon', action='store_true', help='Whether to fit the bounding box to the polygon', required=False, default=False)
    parser.add_argument('--bbox_as_fallback', default=-1.0, type=float,
                        help='When outputting polygons the bbox can be used as fallback polygon. This happens if the ratio '
                             + 'between the surrounding bbox of the polygon and the bbox is smaller than the specified value. '
                             + 'Turned off if < 0.', required=False)
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parsed = parser.parse_args()

    if parsed.fit_bbox_to_polygon and (parsed.bbox_as_fallback >= 0):
        raise Exception("Options --fit_bbox_to_polygon and --bbox_as_fallback cannot be used together!")

    try:
        # This is the actual model that is used for the object detection
        model = init_detector(parsed.config, parsed.checkpoint, device=parsed.device)

        # Get class names
        class_names = determine_classes()
        if parsed.verbose:
            print("Classes: %s" % str(class_names))

        config = Container()
        config.class_names = class_names
        config.model = model
        config.score_threshold = parsed.score
        config.fit_bbox_to_polygon = parsed.fit_bbox_to_polygon
        config.bbox_as_fallback = parsed.bbox_as_fallback
        config.verbose = parsed.verbose

        params = configure_redis(parsed, config=config)
        run_harness(params, process_image)

    except Exception as e:
        print(traceback.format_exc())
