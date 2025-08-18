[![DOI](https://sandbox.zenodo.org/badge/1033514246.svg)](https://handle.stage.datacite.org/10.5072/zenodo.313889)


Understanding Overfitting in Deep Learning
This project explores the concept of **overfitting** in deep learning, using **Convolutional Neural Networks (CNNs)** implemented in **TensorFlow**. We demonstrate how **Dropout**, a regularization technique, helps improve model generalization and prevents overfitting by comparing two models: one with Dropout and one without.
 What You'll Learn
* What is **Overfitting** and why it happens in deep learning
* How to build a CNN using TensorFlow/Keras
* How to apply **Dropout layers** to prevent overfitting
* How training/validation accuracy and loss behave with and without Dropout
* How to visualize model performance with matplotlib

# Overfitting Explained
Overfitting occurs when a machine learning model performs well on training data but poorly on unseen data (test/validation). This usually happens when the model becomes too complex and starts memorizing patterns, including noise, instead of learning generalizable features.
#  Symptoms of Overfitting
* High training accuracy
* Low validation/test accuracy
* Training loss decreases while validation loss increases
###  Solution: Dropout
**Dropout** is a regularization method where randomly selected neurons are ignored during training. This forces the network to not rely too heavily on specific neurons, improving generalization.
##  Model Architectures
###  Without Dropout:

<img width="732" height="195" alt="image" src="https://github.com/user-attachments/assets/4ed1e465-e579-450b-9d95-d8888aeaa1af" />


### With Dropout:

<img width="805" height="220" alt="image" src="https://github.com/user-attachments/assets/b89043e1-fb35-49d1-9718-6e83a28faee1" />


##  Results and Comparison

###  Validation vs Training Curves (No Dropout)

<img width="1884" height="919" alt="image" src="https://github.com/user-attachments/assets/1fcb3a27-c919-4ad8-b97c-f09819f28560" />


### Validation vs Training Curves (With Dropout)

<img width="1892" height="926" alt="image" src="https://github.com/user-attachments/assets/f55adcbe-fc58-4829-a7ab-a932c8e5538c" />


From the graphs, you can observe that **Dropout** helps to reduce the gap between training and validation accuracy/loss, showing better generalization.

##  Technologies Used
* Python
* TensorFlow / Keras
* Matplotlib
* MNIST dataset
## How to Run
1. Clone this repo:

```bash
git clone https://github.com/gauravkumar2424/Understanding-Overfitting-in-DeepLearning.git
cd Understanding-Overfitting-in-DeepLearning
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the training scripts:

```bash
python Test1.py
python Test2.py
```

## Contributing
Feel free to open issues or submit pull requests. Let’s build an amazing learning hub for deep learning enthusiasts together!

## Contact
Created by Gaurav Kumar. If you'd like to connect, message or email me at gkumar20112000@gmail.com .

##  License
MIT License

Copyright (c) 2025 Gaurav Kumar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



> "The best models are not just those that learn but those that generalize."


