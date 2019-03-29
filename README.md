## ORACLE-Fixed

This is an implementation of the ORACLE-Fixed machine learning algorithm as defined [here](https://arxiv.org/abs/1902.09432)

- uses the EMNIST digits dataset
- 6000 train / 2000 cross validation / 2000 test
- Input layer (M X 784 + 1)
- Theta1 (32 X 784 + 1)
- Theta2 (10 X 32 + 1)

## TO RUN

```bash
git clone [repo]
cd [repo]
make setup
make learn
```
