# Training with MMDetection

Allows processing of images with MMDetection.

## Version

MMDetection github repo hash:

```
51df8a9b7ad5f25ebd75cf8e0969c3b728bde08d
```

and timestamp:

```
March 1st, 2020
```

## Docker

### Build local image

* Build the image from Docker file (from within /path_to/mmdetection/2020-03-01/train)

  ```commandline
  sudo docker build -t mmdet_train .
  ```
  
* Run the container

  ```commandline
  sudo docker run --runtime=nvidia --shm-size 8G -v /local:/container -it \
    -e CLASSES=\'class1\',\'class2\',... mmdet_train /path_to/your_data_config.py --autoscale-lr
  ```
  "/local:/container" maps a local disk directory into a directory inside the container


### Pre-built images

* Build

  ```commandline
  docker build -t open-mmlab/mmdetection:2020-03-01_train .
  ```
  
* Tag

  ```commandline
  docker tag \
    open-mmlab/mmdetection:2020-03-01_train \
    public-push.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2020-03-01_train
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2020-03-01_train
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2020-03-01_train
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2020-03-01_train \
    open-mmlab/mmdetection:2020-03-01_train
  ```

* <a name="run">Run</a>

  ```commandline
  docker run --runtime=nvidia --shm-size 8G -v /local:/container -it \
    -e CLASSES=\'class1\',\'class2\',... open-mmlab/mmdetection:2020-03-01_train \
    /path_to/your_data_config.py --autoscale-lr
  ```
  "/local:/container" maps a local disk directory into a directory inside the container
