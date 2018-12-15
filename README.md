# README

It is a pytorch implemention of paper "BRITS: Bidirectional Recurrent Imputation for Time Series, Wei Cao, Dong Wang, Jian Li, Hao Zhou, Lei Li Yitan Li. (NerIPS 2018)". The paper can be found here. http://papers.nips.cc/paper/7911-brits-bidirectional-recurrent-imputation-for-time-series

To train the BRIST model, first please unzip the PhysioNet data into ***raw*** folder, including the label file ***Outcomes-a.txt***.

To run the model:
* make a empty folder named ***json***, and run inpute_process.py.
* run different models:
    * e.g., RITS_I: python main.py --model rits_i --epochs 1000 --batch_size 64 --impute_weight 0.3 --label_weight 1.0 --hid_size 108
    * for most cases, using impute_weight=0.3 and label_weight=1.0 lead to a good performance. Also adjust hid_size to control the number of parameters

