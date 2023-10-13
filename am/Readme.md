## Dependencies
The code was tested using Python 3.7.8 on CentOS Linux 7. The dependencies can be installed using 

`python3 -m pip install -r requirements.txt` 

## Steps to train/validate/test different models.

### Training scripts:
* `run.py` to train **meta-AM** model
* `run_multi.py` used to train **multi-AM** model
* `run_single.py` to train **original-AM** model

###  Model selection scripts:
Selection of the best model based upon performance on the validation dataset across a subset of epochs
* `val.py` for **meta-AM** 
* `val_multi.py` for  **multi-AM** model
*  `val_single.py` for  **original-AM** model

### Testing scripts:
* `testing_with_tuning.py` for testing **meta-AM/multi-AM** model after fine tuning
* `testing_single.py` for testing the **original-AM** model without fine-tuning

### Parameters:
Check `options.py` in which each parameter has a 'help' attribute that gives its precise describtion

## Examples:
* To train the **meta-AM** model on small graphs (N=10,20,30,50) and test on unseen larger graphs:
```sh
python3 run.py --graph_size -1  --run_name <NAME_OF_MODEL>  --variation_type graph_size --baseline_every_Xepochs_for_META 7
python3 val.py --graph_size -1  --run_name temp_val  --val_result_pickle_file <NAME_OF_MODEL> --load_path outputs/SIZE/<NAME_OF_MODEL>/ --variation_type graph_size 
python3 testing_with_tuning.py --graph_size -1  --run_name temp_val --load_path outputs/SIZE/<NAME_OF_MODEL>/  --val_result_pickle_file results_all/validation/SIZE/<NAME_OF_MODEL> --variation_type graph_size --test_result_pickle_file <NAME_OF_MODEL>.pkl  
```
where <NAME_OF_MODEL> could be "SIZE_META". Please keep the name without spaces. 

* To train the **meta-AM** model on graphs of different scales (L=1,2,4) and test on unseen scales:
```sh
python3 run.py --graph_size 40  --run_name <NAME_OF_MODEL>  --variation_type scale --baseline_every_Xepochs_for_META 7
python3 val.py --graph_size 40  --run_name temp_val  --val_result_pickle_file <NAME_OF_MODEL> --load_path outputs/SCALE/<NAME_OF_MODEL>/ --variation_type scale 
python3 testing_with_tuning.py --graph_size 40  --run_name temp_val --load_path outputs/SCALE/<NAME_OF_MODEL>/  --val_result_pickle_file results_all/validation/SCALE/<NAME_OF_MODEL> --variation_type scale --test_result_pickle_file <NAME_OF_MODEL>.pkl  
```
where <NAME_OF_MODEL> could be "SCALE_META". Please keep the name without spaces. 

* To train the **multi-AM** on various numbers of modes M and test on an unseen mode:
```sh
python3 run_multi.py --graph_size 40  --run_name <NAME_OF_MODEL> --variation_type distribution
python3 val_multi.py --graph_size 40  --run_name temp_val  --val_result_pickle_file <NAME_OF_MODEL> --variation_type distribution --load_path outputs/MODE/<NAME_OF_MODEL>/ 
python3 testing_with_tuning.py --graph_size 40   --run_name temp_val  --val_result_pickle_file results_all/validation/MODE/<NAME_OF_MODEL> --variation_type distribution --load_path outputs/MODE/<NAME_OF_MODEL> --test_result_pickle_file GRID_MULTI_40_3_modes.pkl `
```

* To train and test the **original-AM** model on graphs of size N=80:
```sh
python3 run_single.py --graph_size 80  --run_name <NAME_OF_MODEL> --variation_type graph_size 
python3 val_single.py --graph_size 80  --run_name temp_val  --load_path outputs/SIZE/<NAME_OF_MODEL>/ --val_result_pickle_file <NAME_OF_MODEL> --variation_type graph_size
python3 test_single.py --graph_size 80  --run_name temp_val  --val_result_pickle_file results_all/validation/SIZE/<NAME_OF_MODEL> --variation_type graph_size --load_path outputs/SIZE/<NAME_OF_MODEL>/  --test_result_pickle_file GSIZE_SINGLE_80.pkl 
```

**original-AM** for scale L = 5
* ##### Train:
`python3 run_single.py --graph_size 40  --run_name <NAME_OF_MODEL> --variation_type scale --scale 5 `


## Using pre-trained models
We also provide the models (**meta-AM**, **multi-AM** and **original-AM**) that we have trained on our side (for 24 hours using 1 GPU). They are stored in the `outputs_pre_trained` folder and their corresponding validation output files are stored in `results_all/validation_pre_computed/` folder. To use a pre-trained model during testing please modify the 2 parameters below:

* `--val_result_pickle_file` should have a path prefixed with `results_all/validation_pre_computed/` instead of `results_all/validation/`
* `--load_path` should have path prefixed by `pretrained_models` instead of `outputs`.


### Example to test the meta-model for graph size variation using the pre-trained meta-model
```sh
python3 testing_with_tuning.py --graph_size -1  --run_name temp_val --load_path pretrained_models/SIZE/META_10_20_30_50/  --val_result_pickle_file results_all/validation_pre_computed/SIZE/META_10_20_30_50 --variation_type graph_size --test_result_pickle_file GSIZE_META_NORM_k50.pkl 
```


## Credits 

Our code is built upon code provided by Kool  et al. https://github.com/wouterkool/attention-learn-to-route
