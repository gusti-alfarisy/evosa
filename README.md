# Evolutionary Simulated Annealing for Transfer Learning Optimization (CIVEMSA 2022)

Official tensorflow implementation of "Evolutionary Simulated Annealing for Transfer Learning Optimization in Plant-Species Identification Domain"

> **Abstract** - The reuse of the pre-trained deep neural network models has been found successful in improving the classification accuracy for the plant species identification task. However, most of these models have a large number of parameters, and layers and take more storage space which makes them difficult to deploy on embedded or mobile devices for real-time classification. Optimization techniques, such as Simulated Annealing (SA), can help to reduce the number of parameters and the size of these models. However, SA can easily get trapped into local optima when dealing with such complex problems. To solve this problem, we propose a new technique, namely Evolutionary Simulated Annealing (EvoSA), which optimizes the process of transfer learning for the plant-species identification task. We incorporate the genetic operators (e.g., mutation and recombination) on SA to avoid the local optima problem. The technique was tested using the MNetV3-Small as a pre-trained model due to its efficiency on mobile for two plant species data sets (MalayaKew and UBD botanical garden). As compared to the standard SA and Bayesian Optimization techniques, the EvoSA provides the least cost value with a similar number of objective evaluations. Moreover, the EvoSA produces approximately 14x and 6x less cost compared to SA for MalayaKew and UBD botanical data sets, respectively. The results show that the EvoSA can generate solutions with higher test accuracy than typical transfer learning with a competitive number of parameters.

## Environment

**Only for linux**

```angular2html
conda env create -f environment_linux.yml
conda activate evosa_tf
```


## Dataset

Custom datasets have to be put into the **dataset** folder. The convention follows the TensorFlow ImageFolder. The details can be seen below:

```
path/to/image_dir/
  split_name/  # Ex: 'train'
    label1/  # Ex: 'airplane' or '0015'
      xxx.png
      xxy.png
      xxz.png
    label2/
      xxx.png
      xxy.png
      xxz.png
  split_name/  # Ex: 'test'
    ...
```

## Transfer Learning Optimization

- --dataset specify the folder's name of dataset in dataset folder
- --max_trial specify total run of EvoSA
- --iter shows the total iteration in EvoSA for each trial
- --k is the parameter k in k-fold cross validation,  e.g., k = 10 shows 10-fold cross validation.

### Evolutionary Simulated Annealing (EvoSA)
```angular2html
python main_evosa.py --dataset MalayaKew --max_trial 1 --iter 100 --k 10
```
### Simulated Annealing (SA)

```angular2html
python main_sa.py --dataset MalayaKew --max_trial 1 --iter 150 --k 10
```

### Bayesian Optimization

```angular2html
python main_bayes_optim.py --dataset MalayaKew --max_trial 1 --iter 150 --k 10
```

For citation:
```
 
 @INPROCEEDINGS{AlfarisyEvoSA2022,  
 author={Ahmad Fanshuri Alfarisy, Gusti and Ahmed Malik, Owais and Wee Hong, Ong},  
 booktitle={2022 IEEE 9th International Conference on Computational Intelligence and Virtual Environments for Measurement Systems and Applications (CIVEMSA)},   
 title={Evolutionary Simulated Annealing for Transfer Learning Optimization in Plant-Species Identification Domain},   
 year={2022},  
 volume={},  
 number={},  
 ages={1-6},  
 doi={10.1109/CIVEMSA53371.2022.9853679}}
```

## Further Support

You can create a new issue in this github repo. Thank you :)
  
