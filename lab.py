# # import numpy as nm  
# # import matplotlib.pyplot as mtp  
# # import pandas as pd  
  
# # #importing datasets  
# # data_set= pd.read_csv("C:/Users/vikas/Downloads/User_Data.csv")  
# # #Extracting Independent and dependent Variable  
# # x= data_set.iloc[:, [2,3]].values  
# # y= data_set.iloc[:, 4].values  
# # from sklearn.model_selection import train_test_split  
# # x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)
# # from sklearn.preprocessing import StandardScaler    
# # st_x= StandardScaler()    
# # x_train= st_x.fit_transform(x_train)    
# # x_test= st_x.transform(x_test)  
# # #Fitting Logistic Regression to the training set  
# # from sklearn.linear_model import LogisticRegression  
# # classifier= LogisticRegression(random_state=0)  
# # classifier.fit(x_train, y_train)
# # #Predicting the test set result  
# # y_pred= classifier.predict(x_test)  
# # # import the metrics class
# # from sklearn import metrics

# # cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
# # print(cnf_matrix)

















# # import necessary libraries
# import pandas as pd
# from sklearn import tree
# from sklearn.preprocessing import LabelEncoder
# from sklearn.naive_bayes import GaussianNB
# # data = pd.read_csv(r"C:\Users\vikas\Downloads\apple_quality (2) (1).csv")
# # print("THe first 5 values of data is :\n",data.head())
# # X = data.iloc[:,:-1]
# # print("\nThe First 5 values of train data is\n",X.head())
# # y = data.iloc[:,-1]
# # #label encoder is used to Convert then in numbers 
# # encoder = LabelEncoder()
# # y = encoder.fit_transform(y)
# # print("\nNow the Train output is\n",y)
# # from sklearn.model_selection import train_test_split
# # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=0)
# # classifier = GaussianNB()
# # classifier.fit(X_train,y_train)
# # from sklearn.metrics import accuracy_score
# # from sklearn.metrics import confusion_matrix 
# # y_pred= classifier.predict(X_test) 
# # print("the y predict is",y_pred)
# # cm = confusion_matrix(y_test, y_pred) 
# # print ("Confusion Matrix : \n", cm)
# # from sklearn.metrics import accuracy_score 
# # print ("Accuracy : ", accuracy_score(y_test, y_pred))
# import pandas as pd

# # Load the data from CSV file
# data = pd.read_csv(r"C:\Users\vikas\Downloads\enjoysport.csv")

# # Convert data to a list for easy row-by-row access
# a = data.values.tolist()

# # Get the number of attributes (excluding the target column)
# num_attributes = len(a[0]) - 1

# # Initialize the most general and specific hypotheses
# print("The most general hypothesis:", ["?"] * num_attributes)
# print("The most specific hypothesis:", ["0"] * num_attributes)

# # Initialize `hypothesis` to the first positive example
# hypothesis = a[0][:-1]
# print("the",hypothesis)

# print("\nFind-S Algorithm: Finding a maximally specific hypothesis")
# for i in range(len(a)):
#     # Only consider positive examples
#     if a[i][num_attributes] == "Yes":
#         for j in range(num_attributes):
#             # Generalize hypothesis if there's a mismatch
#             if a[i][j] != hypothesis[j]:
#                 hypothesis[j] = '?'
#         print("Training example no:", i + 1, "Hypothesis:", hypothesis)

# # Display the final maximally specific hypothesis
# print("\nThe maximally specific hypothesis for the training set is:")
# print(hypothesis)






import numpy as np 
import pandas as pd

# Load the data
data = pd.read_csv(r"C:\Users\vikas\Downloads\enjoysport.csv")
# Extract the features (concepts) and target values
concepts = np.array(data.iloc[:, 0:-1])
target = np.array(data.iloc[:, -1])

print("\nInstances are:\n", concepts)
print("\nTarget Values are:\n", target)

def learn(concepts, target): 
    # Initialize specific hypothesis to the first instance
    specific_h = concepts[0]
    print("\nInitial Specific Hypothesis:", specific_h)
    
    # Initialize the general hypothesis as the most general (all '?')
    # Determine the length of specific_h
    num_attributes = len(specific_h)

# Initialize general_h as a list of lists with "?" symbols
    general_h = []
    for _ in range(num_attributes):
        general_h.append(["?"] * num_attributes)

    print("General Hypothesis Boundary:", general_h)


    # Iterate over each instance and adjust hypotheses based on the target
    for i, instance in enumerate(concepts):
        print("\nInstance", i+1, ":", instance)

        # For positive instances
        if target[i] == "Yes":
            print("Instance is Positive")
            for j in range(len(specific_h)):
                # Update specific hypothesis if there is a mismatch
                if instance[j] != specific_h[j]:                    
                    specific_h[j] = '?'  
                    # Adjust general hypothesis to remain maximally general
                    general_h[j][j] = '?'

        # For negative instances
        if target[i] == "No":
            print("Instance is Negative")
            for j in range(len(specific_h)):
                # Update general hypothesis to specify differences
                if instance[j] != specific_h[j]:                    
                    general_h[j][j] = specific_h[j]                
                else:                    
                    general_h[j][j] = '?'        

        print("Specific Hypothesis after instance", i+1, ":", specific_h)         
        print("General Hypothesis after instance", i+1, ":", general_h)
    
    # Remove fully generic hypotheses from general boundary
    # Filter out rows that are equal to ['?'] * len(specific_h)
    filtered_general_h = []
    for h in general_h:
        if h != ['?'] * len(specific_h):
            filtered_general_h.append(h)

    # Assign the filtered result back to general_h
    general_h = filtered_general_h

    print("Filtered General Hypothesis Boundary:", general_h)
    return specific_h, general_h


s_final, g_final = learn(concepts, target)
print("\nFinal Specific Hypothesis:\n", s_final)
print("\nFinal General Hypothesis:\n", g_final)
