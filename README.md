## 机器学习 第三次实验 AdaBoost与人脸检测

Requirement and document: [基于AdaBoost算法的人脸检测](https://www.zybuluo.com/liushiya/note/1305548)

## Tip

This implement will output `features.dump` which may take up more than 1.2GB disk space. :open_mouth:

It will load the dump file into memory so make sure there is enough available memory space.

## Usage

```python
python train.py
```

It may take a long time to preprocess all the images and train the classifier.