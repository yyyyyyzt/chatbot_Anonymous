## Prerequisites

make sure python >= 3.7
Install the required packages:

```python
pip install -r requirements.txt
```

We use Yake to extract keywords, if Yake installation fails

```python
pip install git+https://github.com/LIAAD/yake
```

## Data Preparation

we trained on the ConvAI2 and DailyDialog Dataset.
We need prepare the training data from scratch, please follow the steps:

```python
python jsonify_extract_keywords_convai.py # or python jsonify_extract_keywords_dd.py
# get positive and negative example
python sampleing_by_reasoning.py
# use SimCSE to create fake response
python sampleing_by_semantic.py
# get the stastic
python dataset_stastic.py
```

## Training

Note: We used `seed_everything`, so the result is fixed.

To train basemodel, please run the following script:

```python
python basemodel_gpt.py --datatype convai # or daily_dialog
```

To train discriminator, please run the following script:

```python
python discriminator.py --datatype convai # or daily_dialog
```

To finetune by RL, please run the following script:

The path of the trained basemodel and discriminator needs to be passed in
```python
python rl_ppo.py --base_path logs_base/version_0/checkpoints/best.ckpt --disc_path logs_discri/version_0
```

## Evaluation

We evaluate model on a randomly context and target selected from target_set.

please run the following script and the result will save in test_result folder:
```python
python rl_test.py --model_path logs_rl/version_0
```