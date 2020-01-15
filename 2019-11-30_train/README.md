# Training with MMDetection

Allows processing of images with MMDetection.

## Version

MMDetection github repo hash:

```
b7894cbdcbe114e3e9efdd1a6a229419a552c807
```

and timestamp:

```
Sat Nov 30 04:28:00 2019 +1300
```

## Docker

### Build local image

* Build the image from Docker file (from within /path_to/mmdetection/2019-11-30_train)

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
  docker build -t mmdetection:2019-11-30_train .
  ```
  
* Tag

  ```commandline
  docker tag \
    mmdetection:2019-11-30_train \
    public-push.aml-repo.cms.waikato.ac.nz:443/mmdetection:2019-11-30_train
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/mmdetection:2019-11-30_train
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/mmdetection:2019-11-30_train
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/mmdetection:2019-11-30_train \
    mmdetection:2019-11-30_train
  ```

* <a name="run">Run</a>

  ```commandline
  docker run --runtime=nvidia --shm-size 8G -v /local:/container -it \
    -e CLASSES=\'class1\',\'class2\',... mmdetection:2019-11-30_train \
    /path_to/your_data_config.py --autoscale-lr
  ```
  "/local:/container" maps a local disk directory into a directory inside the container

