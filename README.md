## TTA4SR
Official source code for SIGIR 2025 paper: Data Augmentation as Free Lunch: Exploring the Test-Time Augmentation for Sequential Recommendation

## Run the Code

Go to the `src` folder in the `GRU4Rec` and `SASRec` directories, then run the following commands. We provide the original pretrained model.

After running our command, the pretrained model will be loaded. We will first test the performance of the original model, after which the performance will be improved using the proposed TTA method.

## TTA Commands for SASRec

TMask-R
```
python main.py --TTA --data_name=Beauty --model_name=SASRec --TTA_type=TMask-R --sigma=0.5
python main.py --TTA --data_name=Sports --model_name=SASRec --TTA_type=TMask-R --sigma=0.6
python main.py --TTA --data_name=Home --model_name=SASRec --TTA_type=TMask-R --sigma=0.6
python main.py --TTA --data_name=Yelp --model_name=SASRec --TTA_type=TMask-R --sigma=0.7
```

TNoise
```
python main.py --TTA --data_name=Beauty --model_name=SASRec --TTA_type=TNoise --a=1 --b=0.5 --input_num=7 --output_num=5
python main.py --TTA --data_name=Sports --model_name=SASRec --TTA_type=TNoise --a=2 --b=1 --input_num=7 --output_num=5
python main.py --TTA --data_name=Home --model_name=SASRec --TTA_type=TNoise --a=1 --b=0.5 --input_num=7 --output_num=5
python main.py --TTA --data_name=Yelp --model_name=SASRec --TTA_type=TNoise --a=2 --b=1 --input_num=5 --output_num=5
```

## TTA Commands for GRU4Rec

TMask-R
```
python main.py --TTA --data_name=Beauty --model_name=GRU4Rec --TTA_type=TMask-R --sigma=0.5
python main.py --TTA --data_name=Sports --model_name=GRU4Rec --TTA_type=TMask-R --sigma=0.7
python main.py --TTA --data_name=Home --model_name=GRU4Rec --TTA_type=TMask-R --sigma=0.6
python main.py --TTA --data_name=Yelp --model_name=GRU4Rec --TTA_type=TMask-R --sigma=0.7
```


## Commands for Pretraining

We also provide the commands to train the original models.

SASRec
```
python main.py --data_name=Home --star_test=200 --model_name=SASRec
python main.py --data_name=Yelp --star_test=200 --model_name=SASRec
python main.py --data_name=Beauty --star_test=300 --model_name=SASRec 
python main.py --data_name=Sports --star_test=200 --model_name=SASRec
```

GRU4Rec
```
python main.py --data_name=Home --star_test=350 --model_name=GRU4Rec
python main.py --data_name=Yelp --star_test=350 --model_name=GRU4Rec
python main.py --data_name=Beauty --star_test=450 --model_name=GRU4Rec
python main.py --data_name=Sports --star_test=450 --model_name=GRU4Rec
```

## Reference

Please cite our paper if you use this code.
```
@article{dang2025data,
  title={Data Augmentation as Free Lunch: Exploring the Test-Time Augmentation for Sequential Recommendation},
  author={Dang, Yizhou and Liu, Yuting and Yang, Enneng and Huang, Minhan and Guo, Guibing and Zhao, Jianzhe and Wang, Xingwei},
  journal={arXiv preprint arXiv:2504.04843},
  year={2025}
}
```
