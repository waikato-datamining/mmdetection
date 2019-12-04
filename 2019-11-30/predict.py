# Copyright (C) 2019 University of Waikato, Hamilton, NZ
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

import mmcv
from skimage import measure
import pycocotools.mask as maskUtils
from mmdet.apis import init_detector, inference_detector

OUTPUT_COMBINED = False
""" Whether to output CSV file with ROIs for combined images as well (only for debugging). """


def load_image_into_numpy_array(image):
    """
    Method to convert the image into a numpy array.
    faster solution via np.fromstring found here:
    https://stackoverflow.com/a/42036542/4698227

    :param image: the image object to convert
    :type image: Image
    :return: the numpy array
    :rtype: nd.array
    """

    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
    return im_arr


def remove_alpha_channel(image):
    """
    Converts the Image object to RGB.

    :param image: the image object to convert if necessary
    :type image: Image
    :return: the converted object
    :rtype: Image
    """
    if image.mode is 'RGBA' or 'ARGB':
        return image.convert('RGB')
    else:
        return image


def predict_on_images(input_dir, model, output_dir, class_names, score_threshold, num_imgs, inference_times, delete_input):
    """
    Method performing predictions on all images ony by one or combined as specified by the int value of num_imgs.

    :param input_dir: the directory with the images
    :type input_dir: str
    :param model: the mmdetection trained model
    :param output_dir: the output directory to move the images to and store the predictions
    :type output_dir: str
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
    """

    # Iterate through all files present in "input_dir"
    total_time = 0
    times = list()
    times.append("Image(s)_file_name(s),Total_time(ms),Number_of_images,Time_per_image(ms)\n")
    while True:
        start_time = datetime.now()
        im_list = []
        # Loop to pick up images equal to num_imgs or the remaining images if less
        for image_path in os.listdir(input_dir):
            # Load images only, currently supporting only jpg and png
            # TODO image complete?
            if image_path.lower().endswith(".jpg") or image_path.lower().endswith(".png"):
                im_list.append(os.path.join(input_dir, image_path))
            if len(im_list) == num_imgs:
                break

        if len(im_list) == 0:
            time.sleep(1)
            break
        else:
            print("%s - %s" % (str(datetime.now()), ", ".join(os.path.basename(x) for x in im_list)))

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

        image_array = load_image_into_numpy_array(image)
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

        if OUTPUT_COMBINED:
            roi_path = "{}/{}-rois-combined.csv".format(output_dir, os.path.splitext(os.path.basename(im_name))[0])
            with open(roi_path, "w") as roi_file:
                # File header
                roi_file.write("file,x0,y0,x1,y1,x0n,y0n,x1n,y1n,label,label_str,score")
                if segm_result is not None:
                    roi_file.write(",poly_x,poly_y,poly_xn,poly_yn,mask_score\n")
                else:
                    roi_file.write("\n")
                for index in range(len(bboxes)):
                    x0, y0, x1, y1, score = bboxes[index]
                    label = labels[index]
                    label_str = class_names[label - 1]

                    # Ignore this roi if the score is less than the provided threshold
                    if score < score_threshold:
                        continue

                    # Translate roi coordinates into image coordinates
                    x0n = x0 / image.width
                    y0n = y0 / image.height
                    x1n = x1 / image.width
                    y1n = y1 / image.height

                    roi_file.write(
                        "{},{},{},{},{},{},{},{},{},{},{},{}".format(os.path.basename(im_name), x0, y0, x1, y1,
                                                                       x0n, y0n, x1n, y1n, label, label_str, score))

                    if segm_result is not None:
                        segms = mmcv.concat_list(segm_result)
                        if isinstance(segms, tuple):
                            mask = segms[0][index]
                            mask_score = segms[1][index]
                        else:
                            mask = segms[index]
                            mask_score = score
                        mask = maskUtils.decode(mask).astype(np.int)
                        mask = measure.find_contours(mask, 0.5)
                        if mask:
                            roi_file.write(",\"")
                            for c in mask[0]:
                                roi_file.write("{},".format(c[1]))
                            roi_file.write("\",\"")
                            for c in mask[0]:
                                roi_file.write("{},".format(c[0]))
                            roi_file.write("\",\"")
                            for c in mask[0]:
                                roi_file.write("{},".format(c[1] / image.width))
                            roi_file.write("\",\"")
                            for c in mask[0]:
                                roi_file.write("{},".format(c[0] / image.height))
                            roi_file.write("\",{}\n".format(mask_score))
                        else:
                            roi_file.write("\n")
                    else:
                        roi_file.write("\n")

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
            roi_path_tmp = "{}/{}-rois.tmp".format(output_dir, os.path.splitext(os.path.basename(im_list[i]))[0])
            with open(roi_path_tmp, "w") as roi_file:
                # File header
                roi_file.write("file,x0,y0,x1,y1,x0n,y0n,x1n,y1n,label,label_str,score")
                if segm_result is not None:
                    roi_file.write(",poly_x,poly_y,poly_xn,poly_yn,mask_score\n")
                else:
                    roi_file.write("\n")
                # rois
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

                    # output
                    roi_file.write("{},{},{},{},{},{},{},{},{},{},{},{}".format(os.path.basename(im_list[i]),
                                                                                  x0, y0, x1, y1, x0n, y0n, x1n, y1n,
                                                                                  label, label_str, score))
                    if segm_result is not None:
                        segms = mmcv.concat_list(segm_result)
                        if isinstance(segms, tuple):
                            mask = segms[0][index]
                            mask_score = segms[1][index]
                        else:
                            mask = segms[index]
                            mask_score = score
                        mask = maskUtils.decode(mask).astype(np.int)
                        mask = measure.find_contours(mask, 0.5)
                        if mask:
                            roi_file.write(",\"")
                            for c in mask[0]:
                                roi_file.write("{},".format(c[1]))
                            roi_file.write("\",\"")
                            for c in mask[0]:
                                roi_file.write("{},".format(c[0]-min_height))
                            roi_file.write("\",\"")
                            for c in mask[0]:
                                roi_file.write("{},".format(c[1] / img.width))
                            roi_file.write("\",\"")
                            for c in mask[0]:
                                roi_file.write("{},".format((c[0]-min_height) / img.height))
                            roi_file.write("\",{}\n".format(mask_score))
                        else:
                            roi_file.write("\n")
                    else:
                        roi_file.write("\n")
            os.rename(roi_path_tmp, roi_path)

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
    parser.add_argument('--labels', help='Path to text file with comma seperated labels', required=True, default=None)
    parser.add_argument('--score', type=float, help='Score threshold to include in csv file', required=False, default=0.0)
    parser.add_argument('--num_imgs', type=int, help='Number of images to combine', required=False, default=1)
    parser.add_argument('--status', help='file path for predict exit status file', required=False, default=None)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--output_inference_time', action='store_true', help='Whether to output a CSV file with inference times in the --prediction_output directory', required=False, default=False)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
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
            predict_on_images(parsed.prediction_in, model, parsed.prediction_out, class_names,
                                            parsed.score, parsed.num_imgs, parsed.output_inference_time,
                                            parsed.delete_input)

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