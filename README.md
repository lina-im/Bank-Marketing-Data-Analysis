# Bank-Marketing-Data-Analysis
Using the “Bank Marketing” dataset, “bank-full.csv,” apply and compare the three data analysis methods, logistic regression, shallow neural network, and decision tree on the dataset to predict the outcome variable.

# Description:
  The “bank-full.csv” dataset contains 45212 rows with 16 input variables and 1 
  output variable: age (numeric), job (categorical), martial status (categorical), education 
  (categorical), default (binary), balance (numeric), housing (binary), loan (binary), contact 
  (categorical), day (numeric), month (categorical), duration (numeric), campaign (numeric), 
  pdays (numeric), previous (numeric), and poutcome (categorical). The dataset also 
  includes an output variable, y (binary), which is the desired target. This output variable will 
  be used to measure the accuracy of the prediction.  
  
  The data was gathered through direct marketing campaigns, such as phone calls, of a 
  Portuguese banking institution. After the phone call, they gathered data of whether the 
  client subscribed (yes/no) to a bank term deposit. Whether the client subscribed or not to 
  the bank term deposit is the output variable and is used to compare the accuracy of the 
  classification model’s prediction. 

# II. Brief Description of the Three Methods Used  
  **Logistic regression** is one of the methods used, and it is a classification model that 
  predicts binary outcomes, such as yes and no. This method uses optimization with a 
  sigmoid function, which creates an S-shaped curve. This is used to estimate the probability 
  of an event falling into specific category between 0 and 1 by fitting a linear equation to the 
  dataset. Values above the threshold indicate 1 (yes) and values below the threshold 
  indicate 0 (no). This probability demonstrates the likelihood of an event belonging to a 
  specific category (yes/no). 
  
  **Shallow neural network** is another method used, and this is an artificial neural network 
  with typically one or very few hidden layers for simple tasks such as classification. The 
  input layer receives data, the hidden layer(s) processes it through activation functions, 
  such as ReLU or sigmoid, and the output layer produces the final classification (yes/no). 
  
  **Decision tree** is the third method used, and this acts as a series of if-then conditions. 
  Based on a decision rule through the Gini Impurity calculation, branches are created to 
  represent possible outcomes with the leaves representing the final classification (yes/no). 

# III. Experimental Results  
  _Logistic Regression Results:_
  
  ![1](https://github.com/user-attachments/assets/5332d15d-4258-445c-8fef-57407eee8758)
  
  **Figure 1:** The confusion matrix after performing logistic regression shows that the model 
  predicted 7755 correct “no” and 373 correct “yes” as the final classification, achieving a 
  model accuracy of **89.88%**. 

  _Shallow Neural Network (SNN) Results:_
  
  ![2](https://github.com/user-attachments/assets/baf130c9-7e8f-4190-87dd-1af5517da515)

  **Figure 2:** The confusion matrix after performing SNN shows that the model predicted 7654 
  correct “no” and 513 correct “yes” as the final classification, achieving a model accuracy of 
  **90.31%**. 

  _Decision Tree Results:_
  
  ![3](https://github.com/user-attachments/assets/d17aec26-56ea-4d99-85e4-46ccdecc3251)

  **Figure 3:** The confusion matrix after performing logistic regression shows that the model 
  predicted 7360 correct “no” and 546 correct “yes” as the final classification, achieving a 
  model accuracy of **87.43%**. 

# IV. Discussion of Results 
  When comparing the model accuracy with the three different methods, the model used 
  with SNN displayed the highest accuracy of 90.31%. The second highest accuracy was the 
  model used with logistic regression with an accuracy of 89.88%, and the lowest accuracy 
  was the model used with decision tree with an accuracy of 87.43%. 
  
  When using **logistic regression model**, the input features were standardized to ensure a 
  certain feature from dominating the model. Then, the dataset was split into training (80%) 
  and testing (20%) sets before the logistic regression model was applied. In my 
  experimental results, it was found that the logistic regression model displayed a high 
  accuracy. This indicates the model accurately predicted the binary outcome (yes/no) 
  based on the 16 input variables. This could be due to the model showing a linear 
  relationship or a pattern between the input variables and the output variable, which is ideal 
  for a logistic regression model. To improve this accuracy, I could increase the testing set to 
  40% to reduce overfitting. The current dataset is large with 45212 rows, so increasing the 
  testing set and decreasing the training set might lead to improved generalization of data. 
  
  When using **SNN model**, I used one hidden layer with 10 neurons after standardizing the 
  input features and dividing the dataset into training (80%) and testing (20%) sets. Although 
  training this model took more time than the logistic regression model, this model showed a 
  higher accuracy. The SNN model can process more complex and non-linear relationships 
  between the input and target variables, so it can learn the behavior of the dataset more 
  flexibly which yields to higher accuracy. To improve this model, the number of neurons can 
  be increased from 10 to 20. Each neuron in the hidden layer learns different features of the 
  input variables within the training dataset, so increasing the number of neurons could allow 
  the model to capture more details and complex patterns about the dataset. Additionally, 
  the model could utilize L2 regularization during training to prevent overfitting of the data. 
  This approach adds a penalty to the weights in the neural network to prevent one feature 
  from dominating the training dataset to further improve generalization of the model. 
  
  When using **decision tree model**, I used Gini Impurity for splitting nodes within the tree 
  after standardizing and splitting the dataset into training (80%) and testing (20%) sets. The 
  Gini Impurity was used to grow the tree until all nodes were “pure” (1 label: yes/no) or the 
  node contains a low number of data points. The training dataset begins at the root node 
  and splits into nodes recursively that minimizes the Gini Impurity calculation. Based on the 
  experimental results, this model achieved the lowest accuracy. This could indicate the 
  decision tree grew deep, leading to higher time complexity, and overfitting. To improve this 
  model, bootstrapping could be implemented to sample the original dataset with 
  replacement. This allows the same data point to appear more than once in the subsets to 
  help minimize overfitting. 
