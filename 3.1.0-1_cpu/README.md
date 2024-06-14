# MMDetection

Allows processing of images with [MMDetection](https://github.com/open-mmlab/mmdetection). Uses fast-opex for OPEX output.

Uses PyTorch 1.12.0 on CPU.

## Version

MMDetection github repo tag/hash:

```
v3.1.0
f78af7785ada87f1ced75a2313746e4ba3149760
```

and timestamp:

```
June 30th, 2023
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
    -it public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:3.1.0-1_cpu
  ```

### Docker hub

* Pull and run image (adjust volume mappings `-v`):

  ```bash
  docker run --shm-size 8G \
    -v /local/dir:/container/dir \
    -it waikatodatamining/mmdetection:3.1.0-1_cpu
  ```

### Build local image

* Build the image from Docker file (from within /path_to/mmdetection/3.1.0-1_cpu)

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
docker build -t mmdetection:3.1.0-1_cpu .
```

### Inhouse registry  

* Tag

  ```bash
  docker tag \
    mmdetection:3.1.0-1_cpu \
    public-push.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:3.1.0-1_cpu
  ```
  
* Push

  ```bash
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:3.1.0-1_cpu
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```

### Docker hub  

* Tag

  ```bash
  docker tag \
    mmdetection:3.1.0-1_cpu \
    waikatodatamining/mmdetection:3.1.0-1_cpu
  ```
  
* Push

  ```bash
  docker push waikatodatamining/mmdetection:3.1.0-1_cpu
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
  (inside the container).

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
mmdet_config /path/from/my_config.py --save-path /path/to/my_saved_config.py
```

You can browse the config files [here](https://github.com/open-mmlab/mmdetection/tree/v3.1.0/configs).


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
  
  https://github.com/open-mmlab/mmdetection/blob/f78af7785ada87f1ced75a2313746e4ba3149760/docs/en/get_started.md
