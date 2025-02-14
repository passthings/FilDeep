# FilDeep
 FilDeep: Learning Large Deformations of Elastic-Plastic Solids with Multi-Fidelity Data


## Installation guide
We recommend that `python>=3`.

```
cd ./FilDeep
pip install -r ./requirements.txt
```

### Dataset

The dataset is large, so we will upload it on the Internet in the near future. Download the data,  extract it and put it in the `./data` folder. 


### Train

Several hyperparameters are recorded in `./config.ini` file. You can change them to train the model. The default settings are corresponding to those used in the paper.

Then use the below command to start the train:

```

# Display training process on the frontend
python main.py

# Display training process on the backend
nohup python main.py >> ./train.log 2>&1 &
```

The training processor can be seen in `tensorboard`:

```
tensorboard --logdir ./logs --port 9999
```
    
Then you can monitor the training and evaluation details by open `localhost:9999` on your browser.

### Test

The model weights in `./checkpoints` folder. 
We provide a trained model weights in the folder `./checkpoints/train_0` for the test.
If you want to test the model, you should first change the mode in `config.ini`:

```
mode = test
```

and the `mode_id` is the folder `./checkpoints/train_{mode_id}` where saves the checkpoints you need to test. Then run

```
python main.py
```

and the results, MAD and TE will be saved in `./data/pred_results/test_{mode_id}`.


If you want to further calculate the 3-Dimensional Intersection over Union (3D IoU), you need to install the software of [Siemens NX](https://plm.sw.siemens.com/en-US/nx/).
Then, add the folder containing the prediction results to the `root` list in  `./scripts/3D_IoU/3D_IoU_eval.py` and run this python file using the playback function of [Siemens NX](https://plm.sw.siemens.com/en-US/nx/).