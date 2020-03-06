# Inference with MMDetection

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

* Build the image from Docker file (from within /path_to/mmdetection/2020-03-01/predict)

  ```commandline
  sudo docker build -t mmdet_predict .
  ```
  
* Run the container

  ```commandline
  sudo docker run --runtime=nvidia --shm-size 8G -v /local:/container -it \
    -e CLASSES=\'class1\',\'class2\',... mmdet_predict \
    --checkpoint /path_to/epoch_n.pth --config /path_to/your_data_config.py \
    --prediction_in /path_to/test_imgs/ --prediction_out /path_to/test_results/ \
    --labels /path_to/your_data/labels.txt --score 0 --num_imgs 3 --output_inference_time
  ```
  "/local:/container" maps a local disk directory into a directory inside the container

### Pre-built images

* Build

  ```commandline
  docker build -t open-mmlab/mmdetection:2020-03-01_predict .
  ```
  
* Tag

  ```commandline
  docker tag \
    open-mmlab/mmdetection:2020-03-01_predict \
    public-push.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2020-03-01_predict
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2020-03-01_predict
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2020-03-01_predict
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2020-03-01_predict \
    open-mmlab/mmdetection:2020-03-01_predict
  ```

* <a name="run">Run</a>

  ```commandline
  docker run --runtime=nvidia --shm-size 8G -v /local:/container -it \
    -e CLASSES=\'class1\',\'class2\',... open-mmlab/mmdetection:2020-03-01_predict \
    --checkpoint /path_to/epoch_n.pth --config /path_to/your_data_config.py \
    --prediction_in /path_to/test_imgs/ --prediction_out /path_to/test_results/ \
    --labels /path_to/your_data/labels.txt --score 0 --num_imgs 3 --output_inference_time
  ```
  "/local:/container" maps a local disk directory into a directory inside the container
