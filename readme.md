# CROSSER

# The code for CROSSER
This repo contains the source code for the CROSSER model.

## Datasets
- This study evaluates the performance of all methods using three real-world trajectory datasets: the T-drive dataset, the Chengdu dataset, and the Porto dataset.

- T-drive dataset: Collected from taxis in Beijing. [1]

- Chengdu dataset: Available at http://outreach.didichuxing.com/research/opendata/.

- Porto dataset: Available at https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i.

## How to run the code for CROSSER

- We take the T-drive dataset as an example.

1. Pre-training

```bash
cd pretrain_GCN
python pretrain.py --dataset chengdu porto
```

2. Estimation Stage

```bash
cd estimation_stage
python train.py --data_name 'T-drive'
```

3. Recovery Stage

```bash
cd recovery_stage
python train.py --data_name 'T-drive'
```

4. Testing

```bash
cd recovery_stage
python test.py --data_name 'T-drive'
```

## Reference

[1]Yuan J, Zheng Y, Xie X, et al. Driving with knowledge from the physical world[C]//Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining. 2011: 316-324.