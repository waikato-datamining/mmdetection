# Copyright (C) 2019-2020 University of Waikato, Hamilton, NZ
#
# Performs predictions on combined images from all images present in the folder passed as "prediction_in", then outputs the results as
# csv files in the folder passed as "prediction_out"
#
# The number of images to combine at a time before prediction can be specified (passed as a parameter)
# Score threshold can be specified (passed as a parameter) to ignore all rois with low score
# Can run in a continuous mode, where it will run indefinitely

import numpy as np
import os
import argparse
from PIL import Image
from datetime import datetime
import time
from image_complete import auto
import traceback

import mmcv
import pycocotools.mask as maskUtils
from mmdet.apis import init_detector, inference_detector
from wai.annotations.image_utils import image_to_numpyarray, remove_alpha_channel, mask_to_polygon, polygon_to_minrect, polygon_to_lists, lists_to_polygon, polygon_to_bbox
from wai.annotations.core import ImageInfo
from wai.annotations.roi import ROIObject
from wai.annotations.roi.io import ROIWriter

SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
""" supported file extensions (lower case). """

MAX_INCOMPLETE = 3
""" the maximum number of times an image can return 'incomplete' status before getting moved/deleted. """


def predict_on_images(input_dir, model, output_dir, tmp_dir, class_names, score_threshold, 
                      num_imgs, inference_times, delete_input, mask_threshold, mask_nth, 
                      output_minrect, view_margin, fully_connected, fit_bbox_to_polygon,
                      output_width_height):
    """
    Method performing predictions on all images ony by one or combined as specified by the int value of num_imgs.

    :param input_dir: the directory with the images
    :type input_dir: str
    :param model: the mmdetection trained model
    :param output_dir: the output directory to move the images to and store the predictions
    :type output_dir: str
    :param tmp_dir: the temporary directory to store the predictions until finished
    :type tmp_dir: str
    :param class_names: labels or class names
    :type class_names: list[str]
    :param score_threshold: the minimum score predictions have to have
    :type score_threshold: float
    :param num_imgs: the number of images to combine into one before presenting to graph
    :type num_imgs: int
    :param inference_times: whether to output a CSV file with the inference times
    :type inference_times: bool
    :param delete_input: whether to delete the input images rather than moving them to the output directory
    :type delete_input: bool
    :param mask_threshold: the threshold to use for determining the contour of a mask
    :type mask_threshold: float
    :param mask_nth: to speed up polygon computation, use only every nth row and column from mask
    :type mask_nth: int
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
    """

    # Iterate through all files present in "input_dir"
    total_time = 0
    times = list()
    times.append("Image(s)_file_name(s),Total_time(ms),Number_of_images,Time_per_image(ms)\n")

    # counter for keeping track of images that cannot be processed
    incomplete_counter = dict()

    while True:
        start_time = datetime.now()
        im_list = []
        # Loop to pick up images equal to num_imgs or the remaining images if less
        for image_path in os.listdir(input_dir):
            # Load images only
            ext_lower = os.path.splitext(image_path)[1]
            if ext_lower in SUPPORTED_EXTS:
                full_path = os.path.join(input_dir, image_path)
                if auto.is_image_complete(full_path):
                    im_list.append(full_path)
                else:
                    if not full_path in incomplete_counter:
                        incomplete_counter[full_path] = 1
                    else:
                        incomplete_counter[full_path] = incomplete_counter[full_path] + 1

            # remove images that cannot be processed
            remove_from_blacklist = []
            for k in incomplete_counter:
                if incomplete_counter[k] == MAX_INCOMPLETE:
                    print("%s - %s" % (str(datetime.now()), os.path.basename(k)))
                    remove_from_blacklist.append(k)
                    try:
                        if delete_input:
                            print("  flagged as incomplete {} times, deleting\n".format(MAX_INCOMPLETE))
                            os.remove(k)
                        else:
                            print("  flagged as incomplete {} times, skipping\n".format(MAX_INCOMPLETE))
                            os.rename(k, os.path.join(output_dir, os.path.basename(k)))
                    except:
                        print(traceback.format_exc())

            for k in remove_from_blacklist:
                del incomplete_counter[k]

            if len(im_list) == num_imgs:
                break

        if len(im_list) == 0:
            time.sleep(1)
            break
        else:
            print("%s - %s" % (str(datetime.now()), ", ".join(os.path.basename(x) for x in im_list)))

        try:
            # Combining picked up images
            i = len(im_list)
            combined = []
            comb_img = None
            if i > 1:
                while i != 0:
                    if comb_img is None:
                        img2 = Image.open(im_list[i-1])
                        img1 = Image.open(im_list[i-2])
                        i -= 1
                        combined.append(os.path.join(output_dir, "combined.png"))
                    else:
                        img2 = comb_img
                        img1 = Image.open(im_list[i-1])
                    i -= 1
                    # Remove alpha channel if present
                    img1 = remove_alpha_channel(img1)
                    img2 = remove_alpha_channel(img2)
                    w1, h1 = img1.size
                    w2, h2 = img2.size
                    comb_img = np.zeros((h1+h2, max(w1, w2), 3), np.uint8)
                    comb_img[:h1, :w1, :3] = img1
                    comb_img[h1:h1+h2, :w2, :3] = img2
                    comb_img = Image.fromarray(comb_img)

            if comb_img is None:
                im_name = im_list[0]
                image = Image.open(im_name)
                image = remove_alpha_channel(image)
            else:
                im_name = combined[0]
                image = remove_alpha_channel(comb_img)

            image_array = image_to_numpyarray(image)
            result = inference_detector(model, image_array)
            
            assert isinstance(class_names, (tuple, list))
            if isinstance(result, tuple):
                bbox_result, segm_result = result
            else:
                bbox_result, segm_result = result, None
            bboxes = np.vstack(bbox_result)
            labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in 
                            enumerate(bbox_result)]
            labels = np.concatenate(labels)

            # Code for splitting rois to multiple csv's, one csv per image before combining
            max_height = 0
            prev_min = 0
            for i in range(len(im_list)):
                img = Image.open(im_list[i])
                img_height = img.height
                min_height = prev_min
                max_height += img_height
                prev_min = max_height
                roi_path = "{}/{}-rois.csv".format(output_dir, os.path.splitext(os.path.basename(im_list[i]))[0])
                if tmp_dir is not None:
                    roi_path_tmp = "{}/{}-rois.tmp".format(tmp_dir, os.path.splitext(os.path.basename(im_list[i]))[0])
                else:
                    roi_path_tmp = "{}/{}-rois.tmp".format(output_dir, os.path.splitext(os.path.basename(im_list[i]))[0])

                # rois
                roiobjs = []
                for index in range(len(bboxes)):
                    x0, y0, x1, y1, score = bboxes[index]
                    label = labels[index]
                    label_str = class_names[label - 1]

                    # Ignore this roi if the score is less than the provided threshold
                    if score < score_threshold:
                        continue

                    if y0 > max_height or y1 > max_height:
                        continue
                    elif y0 < min_height or y1 < min_height:
                        continue

                    # Translate roi coordinates into original image coordinates (before combining)
                    y0 -= min_height
                    y1 -= min_height
                    x0n = x0 / img.width
                    y0n = y0 / img.height
                    x1n = x1 / img.width
                    y1n = y1 / img.height

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
                        segms = mmcv.concat_list(segm_result)
                        if isinstance(segms, tuple):
                            mask = segms[0][index]
                            score = segms[1][index]
                        else:
                            mask = segms[index]
                        mask = maskUtils.decode(mask).astype(np.int)
                        poly = mask_to_polygon(mask, mask_threshold, mask_nth=mask_nth, view=(x0, y0, x1, y1),
                                               view_margin=view_margin, fully_connected=fully_connected)
                        if len(poly) > 0:
                            px, py = polygon_to_lists(poly[0], swap_x_y=True, normalize=False, as_string=True)
                            pxn, pyn = polygon_to_lists(poly[0], swap_x_y=True, normalize=True, img_width=image.width, img_height=image.height, as_string=True)
                            if output_minrect:
                                bw, bh = polygon_to_minrect(poly[0])
                            if fit_bbox_to_polygon and (len(px) >= 3):
                                x0, y0, x1, y1 = polygon_to_bbox(lists_to_polygon(px, py))
                                x0n, y0n, x1n, y1n = polygon_to_bbox(lists_to_polygon(pxn, pyn))

                    roiobj = ROIObject(x0, y0, x1, y1, x0n, y0n, x1n, y1n, label, label_str, score=score,
                                       poly_x=px, poly_y=py, poly_xn=pxn, poly_yn=pyn,
                                       minrect_w=bw, minrect_h=bh)
                    roiobjs.append(roiobj)

                info = ImageInfo(os.path.basename(im_list[i]))
                roiext = (info, roiobjs)
                options = ["--output", str(tmp_dir if tmp_dir is not None else output_dir), "--no-images"]
                if output_width_height:
                    options.append("--size-mode")
                roiwriter = ROIWriter(options)
                roiwriter.save([roiext])
                if tmp_dir is not None:
                    os.rename(roi_path_tmp, roi_path)
        except:
            print("Failed processing images: {}".format(",".join(im_list)))
            print(traceback.format_exc())

        # Move finished images to output_path or delete it
        for i in range(len(im_list)):
            if delete_input:
                os.remove(im_list[i])
            else:
                os.rename(im_list[i], os.path.join(output_dir, os.path.basename(im_list[i])))

        end_time = datetime.now()
        inference_time = end_time - start_time
        inference_time = int(inference_time.total_seconds() * 1000)
        time_per_image = int(inference_time / len(im_list))
        if inference_times:
            l = ""
            for i in range(len(im_list)):
                l += ("{}|".format(os.path.basename(im_list[i])))
            l += ",{},{},{}\n".format(inference_time, len(im_list), time_per_image)
            times.append(l)
        print("  Inference + I/O time: {} ms\n".format(inference_time))
        total_time += inference_time

    if inference_times:
        with open(os.path.join(output_dir, "inference_time.csv"), "w") as time_file:
            for l in times:
                time_file.write(l)
        with open(os.path.join(output_dir, "total_time.txt"), "w") as total_time_file:
            total_time_file.write("Total inference and I/O time: {} ms\n".format(total_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='Path to the trained model checkpoint', required=True, default=None)
    parser.add_argument('--config', help='Path to the config file', required=True, default=None)
    parser.add_argument('--prediction_in', help='Path to the test images', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the output csv files folder', required=True, default=None)
    parser.add_argument('--prediction_tmp', help='Path to the temporary csv files folder', required=False, default=None)
    parser.add_argument('--labels', help='Path to text file with comma seperated labels', required=True, default=None)
    parser.add_argument('--score', type=float, help='Score threshold to include in csv file', required=False, default=0.0)
    parser.add_argument('--mask_threshold', type=float, help='The threshold (0-1) to use for determining the contour of a mask', required=False, default=0.1)
    parser.add_argument('--mask_nth', type=int, help='To speed polygon detection up, use every nth row and column only', required=False, default=1)
    parser.add_argument('--output_minrect', action='store_true', help='When outputting polygons whether to store the minimal rectangle around the objects in the CSV files as well', required=False, default=False)
    parser.add_argument('--fit_bbox_to_polygon', action='store_true', help='Whether to fit the bounding box to the polygon', required=False, default=False)
    parser.add_argument('--num_imgs', type=int, help='Number of images to combine', required=False, default=1)
    parser.add_argument('--status', help='file path for predict exit status file', required=False, default=None)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--output_inference_time', action='store_true', help='Whether to output a CSV file with inference times in the --prediction_output directory', required=False, default=False)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--view_margin', default=2, type=int, required=False, help='The number of pixels to use as margin around the masks when determining the polygon')
    parser.add_argument('--fully_connected', default='high', choices=['high', 'low'], required=False, help='When determining polygons, whether regions of high or low values should be fully-connected at isthmuses')
    parser.add_argument('--output_width_height', action='store_true', help="Whether to output x/y/w/h instead of x0/y0/x1/y1 in the ROI CSV files", required=False, default=False)
    parsed = parser.parse_args()

    try:
        # This is the actual model that is used for the object detection
        model = init_detector(parsed.config, parsed.checkpoint)
        
        # Get class names
        with open(parsed.labels, "r") as labels_file:
            class_names = labels_file.read()
            class_names = class_names.split(",")

        while True:
            # Performing the prediction and producing the csv files
            predict_on_images(parsed.prediction_in, model, parsed.prediction_out, parsed.prediction_tmp, class_names,
                                            parsed.score, parsed.num_imgs, parsed.output_inference_time,
                                            parsed.delete_input, parsed.mask_threshold, parsed.mask_nth,
                                            parsed.output_minrect, parsed.view_margin, parsed.fully_connected,
                                            parsed.fit_bbox_to_polygon, parsed.output_width_height)

            # Exit if not continuous
            if not parsed.continuous:
                break
        if parsed.status is not None:
            with open(parsed.status, 'w') as f:
                f.write("Success")

    except Exception as e:
        print(e)
        if parsed.status is not None:
            with open(parsed.status, 'w') as f:
                f.write(str(e))
