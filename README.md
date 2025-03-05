# Integrated Communication and Learned Recognizer with Customized RIS Phases and Sensing Durations (CRISense)

This is a PyTorch implementation of the paper "**Integrated Communication and Learned Recognizer with Customized RIS Phases and Sensing Durations**" in *IEEE Transactions on Communications*.
Arxiv link: https://arxiv.org/abs/2503.02244

This paper realizes high-accuracy and fast **target recognition in RIS-aided ISAC systems using wireless signals**.
Specifically, the RIS phases are customized according to the scene, task, quantization, and target priors, and the sensing duration is dynamically determined based on the previously captured information.


# Packages

- python==3.11.5
- pytorch==2.0.1
- numpy==1.26.1
- wandb==0.16.0


# Training

The training scripts come with several options.
An example for training is:

```
python main.py --is_train True --num_glimpses 5 --learned_start True --wandb_project 'CRISense'  
```

# Testing

The codes rely on the .csv files exported from wandb to conduct tests.
An example file name is:

```
wandb_export_2025-03-01T12_00_00.000+08_00.csv  
```

An example for testing is:

```
python main.py --is_train False --test_index 1 --test_wandb_data 'wandb_export_2025-03-01T12_00_00.000+08_00.csv'  
```

# Citation

```
@article{huang2025integrated,
  title={Integrated Communication and Learned Recognizer with Customized {RIS} Phases and Sensing Durations},  
  author={Huang, Yixuan and Yang, Jie and Wen, Chao-Kai and Jin, Shi},
  journal={IEEE Transactions on Communications},
  year={early access, Mar. 2025},
  publisher={IEEE}
}
```

# References

This work is motivated by [Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247) and [Dynamic Computational Time for Recurrent Attention Model](https://arxiv.org/abs/1703.10332) in the computer vision filed.

The codes for CRISense are built on [recurrent-visual-attention](https://github.com/kevinzakka/recurrent-visual-attention) and [DT-RAM](https://github.com/baidu-research/DT-RAM) using wireless communication and sensing knowledge.

