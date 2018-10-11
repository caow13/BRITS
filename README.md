# README

To train the BRIST model, first please unzip the PhysioNet data into ***raw*** folder, including the label file ***Outcomes-a.txt***.

To run the model:
* make a empty folder named ***json***, and run inpute_process.py.
* run different models:
    * e.g., RITS_I: python main.py --model rits_i --epochs 1000 --batch_size 64 --impute_weight 0.3 --label_weight 1.0 --hid_size 108
    * for most cases, using impute_weight=0.3 and label_weight=1.0 lead to a good performance. Also adjust hid_size to control the number of parameters

