
## Pre-requisites

#### Dependencies
The code base been tested on CentOS Linux 7 using Python3.6.7. The dependencies can be installed using requirement file:

```sh
pip3 install -r requirements.txt
```

## Training configurations
For the training we used default meta-parameters of GCN network provided by authors of the model.

All meta/multi training parameters, like type of training (size, mode, scale or mix), list of training/testing tasks, number of finetuning steps during the training and testing etc. should be provided as arguments of the training script. All arguments have default vaules with a description and we encourage users to take a close look at each of the training/testing scripts.

## Train different models

* Script to train **meta-GCN model** on a variety of graph sizes N, numbers of modes M, scales L
```
	python3 train_gcn_meta.py <arglist>
```

* Scripts to train **multi-GCN**  on a variety of graph sizes N, numbers of modes M, scales L
```
	python3 train_gcn_multi.py <arglist>
```
- Oracle model can be trained by using original GCN code or using multi training on just one task.



## Examples

* To train **meta-GCN** model on graphs of sizes 10, 20, 30 and 50 and test on sizes 80, 120 and 150 with 50 finetuning steps during the training:
```
python3 train_gcn.py --task_type size --train_task_list 10 20 30 50 --test_task_list 80 120 150 --train_finetuning_steps 50
```
* To train **meta-GCN** model on graphs of modes 1, 2 and 5 and test on sizes 3, 4, 6 and 8 with 20 finetuning steps during the training:
```
python3 train_gcn.py --task_type mode --train_task_list 1 2 5 --test_task_list 3 4 6 8 --train_finetuning_steps 20
```


## Pre-trained models
Some pretrained models are in the output/directory and they can be tested with the script 
```
python3 test_gcn.py --trained_model <path_to_the_model> --task_type <size/mode/scale> --test_task_list list_of_tasks
```

For instance, to perform testing on the pre-trained **meta-GCN** model with various modes M, on problems with 3 modes, kindly run:

    `python3 test_gcn.py --trained_model outputs/gcn_tsp/mode.pth --task_type mode --test_task_list 3



## Credits 

Our code is built upon code provided by Joshi et al. https://github.com/chaitjo/graph-convnet-tsp/ and Kool et al. https://github.com/wouterkool/dpdp

