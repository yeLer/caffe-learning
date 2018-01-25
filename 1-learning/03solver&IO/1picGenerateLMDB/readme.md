## pic
1.Adding a new folder named data,there are two folders and one file,which are male female and label_name.txt

2picTest
		--data
			--tarin
				--male
				--female
				
			--test
			--label_name.txt

#### 1.we need to run the command like these to unify the filename.

`$ ls train/female |sed "s:^:female/:" | sed "s:$: 0:" >>t_train.txt`

`$ ls train/male |sed "s:^:male/:" | sed "s:$: 1:" >>t_train.txt`

`$ ls test/female |sed "s:^:female/:" | sed "s:$: 0:" >>t_test.txt`

`$ ls test/male |sed "s:^:male/:" | sed "s:$: 1:" >>t_test.txt`

#### 2.convert your images data to the format of LMDB 
`$ cd ~/caffe/build/tools`

`$ ./convert_imageset  --shuffle --resize_width=40 --resize_height=40 /home/yll/cafworkspace/2picTest/data/train/ ./t_train.txt ./t_train_lmdb`

`$ ./convert_imageset  --shuffle --resize_width=40 --resize_height=40 /home/yll/cafworkspace/2picTest/data/test/ ./t_test.txt ./t_test_lmdb`




























