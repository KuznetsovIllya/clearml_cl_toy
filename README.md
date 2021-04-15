# clearml_cl_toy
Toy example of ClearML project for classification

Complete process consists of 3 stages, each represented in ClearML pipeline as separate process:

1) image_augmentation.py creates process with the same ClearML name. This process augments images from folder 'source', puts augmened images into folder 'destination' simultaneously sorting the files by train/test/eval set and by category insdi the sets.

2) train_1st_nn.py trains 1st NN with batch size 4 on images from the 'destination' folder, checks if its accuracy higher than noted in 'best_score.json', saves the NN pb file and rewrites the score in 'best_score.json'.

3) train_2nd_nn.py trains 2nd NN with batch size 4 on images from the 'destination' folder, checks accuracy score, writes pb file and best score if achieved. 