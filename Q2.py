import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense


def main():

    X = np.array([[25000, 600, 200000, 10000, 0], 
        [40000, 700, 300000, 8000, 1], 
        [60000, 750, 500000, 12000, 1],
        [20000, 550, 150000, 15000, 0],
        [80000, 800, 700000, 10000, 1],
        [35000, 650, 250000, 9000, 1],
        [18000, 500, 100000, 12000, 0],
        [90000, 850, 800000, 15000, 1],
        [30000, 580, 200000, 14000, 0],
        [70000, 780, 600000, 10000, 1]
    ])

    y = np.array([ 0, 1, 1, 0, 1,1, 0, 1, 0, 1])


    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)


    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)


    model = Sequential()

    model.add(Dense(8, activation="relu", input_shape=(5,)))

    model.add(Dense(4, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit( X_train, y_train,epochs=50,verbose=0)

    Loss, Accuracy = model.evaluate(X_test,y_test,verbose=0)

    print("Test Accuracy:", Accuracy * 100, "%")

    new_applicant = np.array([[55000, 720, 400000, 10000, 1]])

    new_applicant = scaler.transform(new_applicant)

    Prediction = model.predict(new_applicant,verbose=0)

    if Prediction[0][0] >= 0.5:
        print("Prediction: Loan Approved")
    else:
        print("Prediction: Loan Rejected")

if __name__ == "__main__":
    main()