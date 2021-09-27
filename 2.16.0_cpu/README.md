# MMDetection

Allows processing of images with [MMDetection](https://github.com/open-mmlab/mmdetection).

Uses PyTorch 1.9.0 and [CPU support](https://mmdetection.readthedocs.io/en/v2.16.0/get_started.html#install-without-gpu-support).

## Version

MMDetection github repo tag/hash:

```
v2.16.0
7bd39044f35aec4b90dd797b965777541a8678ff
```

and timestamp:

```
August 31st, 2021
```

## Docker

### Quick start

* Log into registry using *public* credentials:

  ```commandline
  docker login -u public -p public public.aml-repo.cms.waikato.ac.nz:443 
  ```

* Pull and run image (adjust volume mappings `-v`):

  ```commandline
  docker run --shm-size 8G \
    -v /local/dir:/container/dir \
    -it public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2.16.0_cpu
  ```

* If need be, remove all containers and images from your system:

  ```commandline
  docker stop $(docker ps -a -q) && docker rm $(docker ps -a -q) && docker system prune -a
  ```

### Build local image

* Build the image from Docker file (from within /path_to/mmdetection/2.16.0_cpu)

  ```commandline
  docker build -t mmdet .
  ```
  
* Run the container

  ```commandline
  docker run --shm-size 8G -v /local/dir:/container/dir -it mmdet
  ```
  `/local/dir:/container/dir` maps a local disk directory into a directory inside the container

### Usage

* Convert annotations (in ADAMS report format) to MS COCO JSON format using 
  [wai.annotations](https://github.com/waikato-ufdl/wai-annotations). 
  Conversion must be done twice, once for training set and again for validation set.
  
* Store class names or label strings in an environment variable called `MMDET_CLASSES` **(inside the container)**:

  ```commandline
  export MMDET_CLASSES=\'class1\',\'class2\',...
  ```
  
* Alternatively, have the labels stored in a text file with the labels separated by commas and the `MMDET_CLASSES`
  environment variable point at the file.
  
  * The labels are stored in `/data/labels.txt` as follows:

    ```commandline
    class1,class2,...
    ``` 
  
  * Export `MMDET_CLASSES` as follows:

    ```commandline
    export MMDET_CLASSES=/data/labels.txt
    ```

* Copy the config file (of the model you want to train) from /mmdetection/configs (inside the container) or 
  from [here](https://github.com/open-mmlab/mmdetection/tree/7bd39044f35aec4b90dd797b965777541a8678ff/configs) 
  to local disk, then follow [these instructions](#config).

* Train

  Training is not possible on a CPU, only inference.

* Predict and produce CSV files

  ```commandline
  mmdet_predict --checkpoint /path_to/epoch_n.pth --config /path_to/your_data_config.py \
    --prediction_in /path_to/test_imgs/ --prediction_out /path_to/test_results/ \
    --score 0
  ```
  Run with -h for all available options.

  You may also need to specify the following options:

  * `--mask_threshold` - if using another threshold than the default of 0.1
  * `--mask_nth` - use every nth row/col of mask to speed up computation of polygon
  * `--output_minrect`

* Predict via Redis backend

  You need to start the docker container with the `--net=host` option if you are using the host's Redis server.

  The following command listens for images coming through on channel `images` and broadcasts
  predictions in [opex format](https://github.com/WaikatoLink2020/objdet-predictions-exchange-format):

  ```commandline
  mmdet_predict_redis --checkpoint /path_to/epoch_n.pth --config /path_to/your_data_config.py \
    --redis_in images --redis_out predictions --score 0
  ```
  
  Run with `-h` for all available options.

## Pre-built images

* Build

  ```commandline
  docker build -t open-mmlab/mmdetection:2.16.0_cpu .
  ```
  
* Tag

  ```commandline
  docker tag \
    mmdetection:2.16.0_cpu \
    public-push.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2.16.0_cpu
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2.16.0_cpu
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2.16.0_cpu
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2.16.0_cpu \
    open-mmlab/mmdetection:2.16.0_cpu
  ```
  
* <a name="run">Run</a>

  ```commandline
  docker run --shm-size 8G \
    -v /local/dir:/container/dir -it open-mmlab/mmdetection:2.16.0_cpu
  ```
  `/local/dir:/container/dir` maps a local disk directory into a directory inside the container


## Example config files

You can output example config files using:

```commandline
mmdet_config PATH_TO_CONFIG.py
```

You can browse the config files [here](https://github.com/open-mmlab/mmdetection/blob/v2.16.0/docs/model_zoo.md).


## <a name="config">Preparing the config file</a>

1. If necessary, change `num_classes` to labels + 1 (BG).
2. In `train_cfg` and `test_cfg`: change `nms_pre`, `nms_post`, and `max_num` to the preferred values.
3. Change `dataset_type` to `Dataset`
4. Change `data_root` to the root path of your dataset (the directory containing train and val directories).
5. Copy/paste `train_pipeline = [...]` and rename it to `val_pipeline`.
6. Change `pipeline` for `val` to `val_pipeline`.
7. In `train_pipeline`, `val_pipeline` and `test_pipeline`: change `img_scale` to preferred values. 
   Image will be scaled to the smaller value between (larger_scale/larger_image_side) and (smaller_scale/smaller_image_side).
8. Adapt `ann_file` and `img_prefix` to suit your dataset.
9. Interval in `checkpoint_config` will determine the frequency of saving models while training 
   (10 for example will save a model after every 10 epochs).
10. Change `total_epochs` to how many epochs you want to train the model for.
11. Change `work_dir` to the path where you want to save the trained models to.
12. If you want to include the validation set, add `, ('val', 1)` to `workflow`.

_You don't have to copy the config file back, just point at it when training._

**NB:** A fully expanded config file will get placed in the output directory with the same
name as the config plus the extension *.full*.


## Permissions

When running the docker container as regular use, you will want to set the correct
user and group on the files generated by the container (aka the user:group launching
the container):

```commandline
docker run -u $(id -u):$(id -g) -e USER=$USER ...
```

## Caching models

PyTorch downloads base models, if necessary. However, by using Docker, this means that 
models will get downloaded with each Docker image, using up unnecessary bandwidth and
slowing down the startup. To avoid this, you can map a directory on the host machine
to cache the base models for all processes (usually, there would be only one concurrent
model being trained):  

```
-v /somewhere/local/cache:/.cache
```

Or specifically for PyTorch:

```
-v /somewhere/local/cache/torch:/.cache/torch
```

**NB:** When running the container as root rather than a specific user, the internal directory will have to be
prefixed with `/root`. 
