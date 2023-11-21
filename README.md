
# Requirements
Python == 3.7.0 

Need to make conda environment 

```
conda create -n myenv python=3.7
cd Adversarial-SimCSE
pip install -r Adversarial-SimCSE/requirements.txt
pip install wandb optuna # For logging and hyperparameter tuning purpose
```

# Downloading the datasets

For SentEval(Evaluation data)
```
cd Adversarial-SimCSE/SentEval/data/downstream
bash Adversarial-SimCSE/SentEval/data/downstream/download_dataset.sh
```

For training data (Unsupervised)

```
cd Adversarial-SimCSE/SentEval/data
bash /Adversarial-SimCSE/data/download_wiki.sh
```


# Training Unsupervised setting

```
bash Adversarial-SimCSE/run_unsup_example.sh
```

At the end of the training, it automatically evaluate the training model with STS dataset


# Result Unsupervised V-AdvCSE
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
|-------|-------|-------|-------|-------|--------------|-----------------|-------|
| 69.91 | 82.90 | 75.26 | 82.76 | 79.28 |    79.94     |      70.91      | 77.53 |
# Result Unsupervised SimCSE
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
|-------|-------|-------|-------|-------|--------------|-----------------|-------|
| 68.40 | 82.41 | 74.38 | 80.91 | 78.56|    76.85      |      72.23      | 76.25 |



# Contribution of my work
My implementation is based on SimCSE work https://github.com/princeton-nlp/SimCSE

Most of modification of the code is in the 
```
cd Adversarial-SimCSE/simcse/models.py
```
In this file, I implemented Adversarial loss to modify the original loss objective
```
class SMARTLoss(nn.Module) 
```
In the following code, I modified the original loss function with adding weighted adversarial loss
```
    z1_emb =first_output.last_hidden_state[:,0,:]
    loss_adv = cls.smart_loss(z1_emb,z1,radius,step_size,reduction)

    loss_cont = loss_fct(cos_sim, labels)
    loss = loss_cont + loss_adv*alpha
```
In the train.py file I also added code for logging with Wandb and hyper-parameter tunning with optuna
```
    if wandb_on:
        wandb.init()
        cfig = wandb.config
        alpha = cfig.alpha
        radius = cfig.radius
    elif optuna_on:
        wandb.init()
        alpha = trial.suggest_categorical('alpha',[0.00005, 0.00001, 0.000005, 0.000001, 0.0000005, 0.0000001])
        radius = trial.suggest_categorical('radius',[1])
        step_size = trial.suggest_categorical('step_size',[1e-3])
        learning_rate =  trial.suggest_categorical('learning_rate',[3e-5])
        reduction = "sum"
    else:
        alpha = 0.0001
        radius = 1
        step_size = 1e-3
        reduction = "sum"
        learning_rate = 3e-5

```
```
if __name__ == "__main__":
    wandb_on = False
    optuna_on = False
    if wandb_on:
        sweep_config = dict()
        sweep_config['method'] = 'grid'
        sweep_config['metric'] = {'name': 'test_accuracy', 'goal': 'maximize'}
        sweep_config['parameters'] = {'alpha' : {'values' : [0.5]}, 'K_iter':{'values':[1]}, 'radius':{'values':[5]}}
        sweep_id = wandb.sweep(sweep_config, project = 'Adversarial_SimCSE')
        wandb.agent(sweep_id, main)
    elif optuna_on:
        search_space = {'alpha': [0.00005, 0.00001, 0.000005, 0.000001, 0.0000005, 0.0000001],'radius':[1], 'learning_rate':[3e-5]}
        sampler=optuna.samplers.GridSampler(search_space)
        study = optuna.create_study(study_name = 'bert-unsup_grid_mse_rad1_smaller',storage="sqlite:///db.sqlite24",load_if_exists= True,
                                direction ="maximize",sampler =sampler)
        study.optimize(main, n_trials = 6)
        # study = optuna.create_study(study_name = 'simcse_adv_wandb_13',storage="sqlite:///db.sqlite13",load_if_exists= True,
        #                         direction ="maximize")
    
        # study.optimize(main, n_trials = 100)
        trials = study.best_trial
        print("value: ", trials.value)
        print("parmas :")
        for k, v in trials.params.items():
            print("   {}:      {}".format(k,v))
    else:
        main()

```
