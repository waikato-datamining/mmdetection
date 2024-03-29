# MMDetection

Allows processing of images with [MMDetection](https://github.com/open-mmlab/mmdetection).

Uses PyTorch 1.9.0 on CPU.

## Version

MMDetection github repo tag/hash:

```
v2.27.0
94afcdc6aa1e2c849c9d8334ec7730bb52b17922
```

and timestamp:

```
January 5th, 2023
```

## Quick start

### Inhouse registry

* Log into registry using *public* credentials:

  ```bash
  docker login -u public -p public public.aml-repo.cms.waikato.ac.nz:443 
  ```

* Pull and run image (adjust volume mappings `-v`):

  ```bash
  docker run --shm-size 8G \
    -v /local/dir:/container/dir \
    -it public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2.27.0_cpu
  ```

### Docker hub

* Pull and run image (adjust volume mappings `-v`):

  ```bash
  docker run --shm-size 8G \
    -v /local/dir:/container/dir \
    -it waikatodatamining/mmdetection:2.27.0_cpu
  ```

### Build local image

* Build the image from Docker file (from within /path_to/mmdetection/2.27.0_cpu)

  ```bash
  docker build -t mmdet .
  ```
  
* Run the container

  ```bash
  docker run --shm-size 8G -v /local/dir:/container/dir -it mmdet
  ```
  `/local/dir:/container/dir` maps a local disk directory into a directory inside the container

## Publish images

### Build

```bash
docker build -t mmdetection:2.27.0_cpu .
```

### Inhouse registry  

* Tag

  ```bash
  docker tag \
    mmdetection:2.27.0_cpu \
    public-push.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2.27.0_cpu
  ```
  
* Push

  ```bash
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2.27.0_cpu
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```

### Docker hub  

* Tag

  ```bash
  docker tag \
    mmdetection:2.27.0_cpu \
    waikatodatamining/mmdetection:2.27.0_cpu
  ```
  
* Push

  ```bash
  docker push waikatodatamining/mmdetection:2.27.0_cpu
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login 
  ```

## Scripts

The following scripts are available:

* `mmdet_config` - for exporting default configurations into separate files
* `mmdet_train` - for training a model (though not all operators supported)
* `mmdet_predict` - for applying a model to images (uses file-polling)
* `mmdet_predict_redis` - for applying a model to images (via [Redis](https://redis.io/) backend)
* `mmdet_onnx` - for exporting a trained PyTorch model to [ONNX](https://onnx.ai/)


## Usage

* The data must be in COCO format. You can use [wai.annotations](https://github.com/waikato-ufdl/wai-annotations) 
  to convert your data from other formats.
  
* Store class names or label strings in an environment variable called `MMDET_CLASSES` **(inside the container)**:

  ```bash
  export MMDET_CLASSES=\'class1\',\'class2\',...
  ```
  
* Alternatively, have the labels stored in a text file with the labels separated by commas and the `MMDET_CLASSES`
  environment variable point at the file.
  
  * The labels are stored in `/data/labels.txt` as follows:

    ```bash
    class1,class2,...
    ``` 
  
  * Export `MMDET_CLASSES` as follows:

    ```bash
    export MMDET_CLASSES=/data/labels.txt
    ```

* Use `mmdet_config` to export the config file (of the model you want to train) from `/mmdetection/configs` 
  (inside the container), then follow [these instructions](#config).

* Train

  ```bash
  mmdet_train /path_to/your_data_config.py
  ```

* Predict and produce CSV files

  ```bash
  mmdet_predict \
      --checkpoint /path_to/epoch_n.pth \
      --config /path_to/your_data_config.py \
      --prediction_in /path_to/test_imgs/ \
      --prediction_out /path_to/test_results/ \
      --score 0.0
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

  ```bash
  mmdet_predict_redis \
      --checkpoint /path_to/epoch_n.pth \
      --config /path_to/your_data_config.py \
      --redis_in images \
      --redis_out predictions \
      --score 0.0
  ```
  
  Run with `-h` for all available options.



## Example config files

You can output example config files using (stored under `/mmdetection/configs` for the various network types):

```bash
mmdet_config /path/to/my_config.py
```

You can browse the config files [here](https://github.com/open-mmlab/mmdetection/blob/v2.27.0/docs/model_zoo.md).


## <a name="config">Preparing the config file</a>

1. If necessary, change `num_classes` to number of labels (background not counted).
2. In `train_cfg` and `test_cfg`: change `nms_pre`, `nms_post`, and `max_num` to the preferred values.
3. Change `dataset_type` to `Dataset` and any occurrences of `type` in the `train`, `test`, `val` sections of 
   the `data` dictionary.
4. Change `data_root` to the root path of your dataset (the directory containing train and val directories).
5. In `train_pipeline`, `val_pipeline` and `test_pipeline`: change `img_scale` to preferred values. 
   Image will be scaled to the smaller value between (larger_scale/larger_image_side) and (smaller_scale/smaller_image_side).
6. Adapt `ann_file` and `img_prefix` to suit your dataset.
7. Interval in `checkpoint_config` will determine the frequency of saving models while training 
   (10 for example will save a model after every 10 epochs).
8. In the `runner` property, change `max_epochs` to how many epochs you want to train the model for.
9. Change `work_dir` to the path where you want to save the trained models to.
10. Change `load_from` to the file name of the pre-trained network that you downloaded from the model zoo.
11. If you want to include the validation set, add `, ('val', 1)` to `workflow`.

_You don't have to copy the config file back, just point at it when training._

**NB:** A fully expanded config file will get placed in the output directory with the same
name as the config plus the extension *.full*.


## Permissions

When running the docker container as regular use, you will want to set the correct
user and group on the files generated by the container (aka the user:group launching
the container):

```bash
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


## Testing Redis

You can use [simple-redis-helper](https://pypi.org/project/simple-redis-helper/) to broadcast images 
and listen for image segmentation results when testing.


## Troubleshooting

* `PermissionError: [Errno 13] Permission denied: '/mmdetection/work_dirs'`

  The top-level `work_dir` parameter is missing from your config file. This directory will contain log files
  and checkpoints and final model. Either add this parameter to the config file or supply it on the 
  command-line:
  
  ```bash
  mmdet_train /path/to/my_config.py --cfg-options work_dir=/some/where/output 
  ```

* `RuntimeError: Expected 4-dimensional input for 4-dimensional weight [64, 3, 7, 7], but got 5-dimensional input of size [1, 2, 3, 800, 800] instead`
  
  Though training support on CPU is in principle available, not all operators are supported
  that your network architecture may require. For more details:
  
  https://github.com/open-mmlab/mmdetection/blob/95747b18fc684491a7921deac1c619679ac6d3fb/docs/en/get_started.md
