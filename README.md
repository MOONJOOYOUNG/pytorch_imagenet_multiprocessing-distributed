# pytorch_imagenet_multiprocessing-distributed
pytorch_imagenet_multiprocessing-distributed

# python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:2222' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 none --save_path './test/' --gpu_count 4

https://github.com/pytorch/examples/tree/master/imagenet

## 동작
* 사용할 gpu 수만큼 초기 init weight가 같은 네트워크 구성.
  * 배치가 32면, 각 네트워크당 배치크기 : 32
* data sampler를 통해 서로 다른 네트워크에 서로 다른 데이터 배분.
  * 배치가 32고 사용하는 gpu가 4개라면 iter 마다 128개의 데이터를 보게됨. 
* 각 네트워크마다 그레이디언트를 구하고 각각 값을 모아서 평균을 구함.
* 평균을 각 네트워크에 뿌려주고 네트워크의 모델을 업데이트함
 * 결과 뽑을씨 rank를 이용해서 하나의 gpu 값만 확인하고 record 하면됌   
``` 
# 0 : 0번 gpu를 말함
if dist.get_rank() == 0:
        logger.write([epoch, losses.avg, top1.avg, top5.avg])
``` 
