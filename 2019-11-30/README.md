# MMDetection

Allows processing of images with MMDetection.

# Version

MMDetection github repo hash:

```
b7894cbdcbe114e3e9efdd1a6a229419a552c807
```

and timestamp:

```
Sat Nov 30 04:28:00 2019 +1300
```

## Installation & Usage on Linux with Docker

* Build the image from Docker file (from within /path_to/mmdetection/2019-11-30)

  ```commandline
  sudo docker build -t mmdet .
  ```
  
* Run the container

  ```commandline
  sudo docker run --runtime=nvidia --shm-size 8G -v /local:/container -it mmdet
  ```
  "/local:/container" maps a local disk directory into a directory inside the container

* Convert annotations (in ADAMS report format) to MS COCO JSON format (See https://github.com/waikato-datamining/mscocodata). 
  Conversion must be done twice, once for training set and again for validation set.
  
* Store class names or label strings in an environment variable called "CLASSES" **(inside the container)**:

  ```commandline
  export CLASSES=\'class1\',\'class2\',...
  ```

* Copy the config file (of the model you want to train) from /mmdetection/configs (inside the container) or from [here](https://github.com/open-mmlab/mmdetection/tree/b7894cbdcbe114e3e9efdd1a6a229419a552c807/configs) to local disk, then follow [these instructions](#config).

* Train

  ```commandline
  mmdet_train /path_to/your_data_config.py --autoscale-lr
  ```

* Predict and produce csv files

  ```commandline
  mmdet_predict --checkpoint /path_to/epoch_n.pth --config /path_to/your_data_config.py \
    --prediction_in /path_to/test_imgs/ --prediction_out /path_to/test_results/ \
    --labels /path_to/your_data/labels.txt --score 0 --num_imgs 3 --output_inference_time
  ```
  Run with -h for all available options.
  
#### <a name="config">Preparing the config file</a>

1. Change num_classes to labels + 1 (BG).
2. In train_cfg & test_cfg: change nms_pre, nms_post, & max_num to the preferred values.
3. Change dataset_type to 'Dataset'
4. Change data_root to the root path of your dataset (the directory containing train & val directories).
5. Copy & paste train_pipeline = [...] and change it to val_pipeline.
6. In train_pipeline, val_pipeline, & test_pipeline: change img_scale to preferred values. Image will be scaled to the smaller value between (larger_scale/larger_image_side) & (smaller_scale/smaller_image_side).
7. Change ann_file (after data_root +) to train/annotations.json (train) or val/annotations.json (val & test).
8. Change img_prefix (after data_root +) to train/ (train) or val/ (val & test).
9. Change pipeline for val to val_pipeline.
10. Interval in checkpoint_config will determine the frequency of saving models while training (10 for example will save a model after every 10 epochs).
11. Change total_epochs to how many epochs you want to train the model for.
12. Change work_dir to the path where you want to save the trained models to.
13. Add , ('val', 1) to workflow.

_You don't have to copy the config file back, just point at it when training._

## Docker Image in aml-repo

* Build

  ```commandline
  docker build -t mmdetection:2019-11-30 .
  ```
  
* Tag

  ```commandline
  docker tag \
    mmdetection:2019-11-30 \
    public-push.aml-repo.cms.waikato.ac.nz:443/mmdetection:2019-11-30
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/mmdetection:2019-11-30
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/mmdetection:2019-11-30
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/mmdetection:2019-11-30 \
    mmdetection:2019-11-30
  ```
  
* <a name="run">Run</a>

  ```commandline
  docker run --runtime=nvidia --shm-size 8G -v /local:/container -it mmdetection:2019-11-30
  ```
  "/local:/container" maps a local disk directory into a directory inside the container
