# Fraud Detection (_Binary Classification_) of BPJS Hackathon Dataset Using CNN

Predicting the potential for fraud in hospital service claims using CNN (Convolutional Neural Networks)

---

### `🔖 What is CNN?`
<a href="https://insightsimaging.springeropen.com/articles/10.1007/s13244-018-0639-9"> <button>CNN</button></a> (Convolutional Neural Networks) is a mathematical construct that is typically composed of three types of layers (or building blocks): convolution, pooling, and fully connected layers.

<a href="cnn architecture"><img src="https://github.com/marceljsh/DaMi-FraudDetection-BPJS-CNN/assets/70984049/b9b177cb-be2f-4d32-87e1-90b22d4d378e" align="center" height="400" width="750" ></a>

An <a href="https://insightsimaging.springeropen.com/articles/10.1007/s13244-018-0639-9/figures/1"> <button>overview</button></a> of a convolutional neural network (CNN) architecture and the training process.

### `🗂️ Dataset`
<a href="https://bpjs-kesehatan.go.id/"> <button>BPJS</button></a> (Badan Penyelenggara Jaminan Sosial) Dataset
Hackathon to train the model, which can be accessed <a
    href="https://github.com/marceljsh/DaMi-FraudDetection-BPJS-CNN/blob/main/dataset.csv"> <button>here</button></a>.


---


### `📚 Business Understanding`
The hospital plays a crucial role as a healthcare service provider for the community, with BPJS Kesehatan significantly impacting the quality of services provided. However, with the increasing utilization of BPJS Kesehatan, there is a risk of fraud that can harm various parties. Perpetrators of fraud can originate from BPJS Kesehatan participants, healthcare facility operators, healthcare service buyers, providers of drugs and medical equipment, as well as other involved parties.<br>

To address this issue, an analysis of fraud prediction is necessary to identify the factors that can lead to fraud. This will enable us to formulate preventive measures and ensure sustainable business growth.

### `📑 Data Understanding`
- **Describe data**: The dataset consists of 53 variables with a total of 200217 observations. The dataset description can be seen in the table below.

  <img width="419" alt="dataset description" src="https://github.com/marceljsh/DaMi-FraudDetection-BPJS-CNN/assets/70984049/0f42f02f-56cf-4c80-bbbd-c6e70df1ed27">

The visualization below shows the correlation between features.

<img width="667" alt="heatmap" src="https://github.com/marceljsh/DaMi-FraudDetection-BPJS-CNN/assets/70984049/5e7b5f86-eb6e-489e-9ebf-ca85b31c8e93">


- **Data Validation**: Evaluation phase, the goal is to ensure completeness and quality. Any missing values or noise in the data may be due to input errors, and it's important to fix these for accurate results.


### `🫧 Data Preparation`
Pada bagian ini akan dijabarkan proses untuk persiapan data, yaitu pemilahan atribut untuk analisis dan pembersihan data.
<br />
**1. Data Selection**
<br />
*Data Selection* atau *Feature Selection* digunakan untuk memilih beberapa fitur untuk membangun model klasifikasi. Proses seleksi dilakukan dengan melakukan penggabungan terhadap fitur yang terkait menjadi satu selanjutnya memilih fitur yang akan digunakan sebagai input fitur.
<br />
**2. Data Cleaning**
<br />
*Data Cleaning* merupakan proses persiapan data dengan cara menghapus atau memodifikasi data yang salah, tidak akurat, tidak terformat maupun duplikat. Data yang rusak tentunya akan berpengaruh pada kinerja pada sistem. Pada tahap ini, kami akan *drop* kolom `visit_id`, `dx2_koo_k93`, `dx2_u00_u99`, dan `procv00_v89`.

### `🔁 Modeling`
Pada bagian ini akan dijelaskan cara modeling dengan penerapan CNN dalam melakukan prediksi jumlah kasus dan unit cost pada sebuah daerah akibat penambahan Rumah Sakit dari 200217 observasi dan banyak variabel. Adapun beberapa fitur yang telah dikembangkan dari hasil encode sebanyak **116** fitur.

Berikut adalah hal apa saja yang telah dilakukan ditahap modeling: 
<br />
**1. `feature selection` for determining `input` and `target features`**
```
X = df.drop('label', axis = 1)
y = df['label']
```
**2. change the scale for each feature using `normalization` so that each value is on a `scale` between `0-1`.**
```
X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
```
**3. divide the dataset into `training data` and `test data`**
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
```
**4. `scaling standardization` focuses on turning raw data into usable information before it is analyzed.**
```
scaler = StandardScaler() 

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
```
**5. Define and build Model CNN**
```
model = Sequential()
model.add(Conv1D(32, 2, activation='relu', input_shape = (111, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Conv1D(64, 2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(1, activation='sigmoid'))
model.summary()
```
**Output :**
<br />
| Layer                      | Output Shape      | Param #   |
| ---------------------------| ------------------| --------- |
| conv1d_2 (Conv1D)           | (None, 110, 32)   | 96        |
| batch_normalization_2       | (None, 110, 32)   | 128       |
| dropout_3 (Dropout)         | (None, 110, 32)   | 0         |
| conv1d_3 (Conv1D)           | (None, 109, 64)   | 4160      |
| batch_normalization_3       | (None, 109, 64)   | 256       |
| dropout_4 (Dropout)         | (None, 109, 64)   | 0         |
| flatten_1 (Flatten)         | (None, 6976)      | 0         |
| dense_2 (Dense)             | (None, 64)        | 446528    |
| dropout_5 (Dropout)         | (None, 64)        | 0         |
| dense_3 (Dense)             | (None, 1)         | 65        |
<br />

**Total params: 451233 (1.72 MB)** 

**Trainable params: 451041 (1.72 MB)** 

**Non-trainable params: 192 (768.00 Byte)**

<br />

**6. Compile dan fit model CNN**

```
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

training_results = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1) 
```
<br />

**Output:**

```
Epoch 1/10
5013/5013 [==============================] - 32s 6ms/step - loss: 0.6848 - accuracy: 0.5502 - val_loss: 0.6697 - val_accuracy: 0.5928
Epoch 2/10
5013/5013 [==============================] - 32s 6ms/step - loss: 0.6730 - accuracy: 0.5698 - val_loss: 0.6556 - val_accuracy: 0.6088
Epoch 3/10
5013/5013 [==============================] - 32s 6ms/step - loss: 0.6680 - accuracy: 0.5757 - val_loss: 0.6590 - val_accuracy: 0.6152
Epoch 4/10
5013/5013 [==============================] - 32s 6ms/step - loss: 0.6655 - accuracy: 0.5806 - val_loss: 0.6583 - val_accuracy: 0.6173
Epoch 5/10
5013/5013 [==============================] - 37s 7ms/step - loss: 0.6633 - accuracy: 0.5855 - val_loss: 0.6503 - val_accuracy: 0.6170
Epoch 6/10
5013/5013 [==============================] - 40s 8ms/step - loss: 0.6612 - accuracy: 0.5880 - val_loss: 0.6448 - val_accuracy: 0.6243
Epoch 7/10
5013/5013 [==============================] - 38s 7ms/step - loss: 0.6603 - accuracy: 0.5911 - val_loss: 0.6456 - val_accuracy: 0.6195
Epoch 8/10
5013/5013 [==============================] - 40s 8ms/step - loss: 0.6581 - accuracy: 0.5937 - val_loss: 0.6435 - val_accuracy: 0.6237
Epoch 9/10
5013/5013 [==============================] - 39s 8ms/step - loss: 0.6569 - accuracy: 0.5963 - val_loss: 0.6461 - val_accuracy: 0.6254
Epoch 10/10
5013/5013 [==============================] - 37s 7ms/step - loss: 0.6548 - accuracy: 0.6002 - val_loss: 0.6399 - val_accuracy: 0.6294
```
<br />

### `🔍 Evaluation`
Pada bagian ini dilakukan tahap Evaluation (Evaluasi) dengan tujuan untuk memprediksi seberapa baik model akhir akan bekerja nantinya sehingga diketahui apakah model tersebut layak digunakan atau tidak dan untuk membantu menemukan model yang paling mewakili pelatihan data <br />
**1.  Uji Coba Model ke Data Testing** 
<br />

```
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,  random_state=0)

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
loss_v, accuracy_v = model.evaluate(X_val, y_val, verbose=1)
```

<br />

**Output:**
```
1254/1254 [==============================] - 2s 1ms/step - loss: 0.6399 - accuracy: 0.6294
1003/1003 [==============================] - 1s 1ms/step - loss: 0.6375 - accuracy: 0.6353<br />
```

<br />

**2.  Validation Data**

<br />

```
print("Validation | accuracy = %f  ;  loss = %f" % (accuracy_v, loss_v))
print("Test       | accuracy = %f  ;  loss = %f" % (accuracy, loss))
```

**Output :**
```
Validation | accuracy = 0.635341  ;  loss = 0.637452
Test       | accuracy = 0.629370  ;  loss = 0.639914
```

**3. CNN Confusion Matrix**
<br />

```
y_pred_cnn = model.predict(X_test)
y_pred_cnn = np.round(y_pred_cnn).astype(int)

cm = confusion_matrix(y_test, y_pred_cnn)
sns.heatmap(cm, annot=True, fmt="d", cbar=False)
plt.title('CNN Confusion Matrix')

plt.show()
```

**Output Gambar**

<img width="667" alt="heatmap" src="https://github.com/marceljsh/DaMi-FraudDetection-BPJS-CNN/blob/main/assets/img/CNN-matrix.png">

<br />

```
print(classification_report(y_test, y_pred_cnn))
```

<br />

**Output :**
<br />

|              | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| Class 0       | 0.66      | 0.55   | 0.60     | 20179   |
| Class 1       | 0.61      | 0.71   | 0.66     | 19923   |
| ------------ | --------- | ------ | -------- | ------- |
| Accuracy     |           |        | 0.63     | 40102   |
| Macro Avg    | 0.63      | 0.63   | 0.63     | 40102   |
| Weighted Avg | 0.63      | 0.63   | 0.63     | 40102   |

```
print('Precision Score:', precision_score(y_test, y_pred_cnn))
print('Accuracy Score:', accuracy_score(y_test, y_pred_cnn))
print('Recall Score:', recall_score(y_test,y_pred_cnn))
```


**Output:**

```
Precision Score: 0.609183497324357
Accuracy Score: 0.6293701062291157
Recall Score: 0.7085278321537921
```

**4. Number of fraud points in the testing dataset**
```
print("Number of fraud points in the testing dataset =", sum(y_test))
```

**Output:**
```
Number of fraud points in the testing dataset = 19923
```



### `🎯 Deployment`
The BPJS Fraud Detection application is deployed using [Streamlit](https://streamlit.io/), a popular open-source Python library that enables rapid development of web applications for machine learning and data science projects.

Excited to see our BPJS Fraud Detection system in action? 🚀 Click [here](https://fraud-detection-web-app-xafy7nzgnqrmcclnopaiu7.streamlit.app/) to explore and experience the live demo! Dive in and discover the innovative features firsthand. Happy exploring! 


![fraud det](https://github.com/marceljsh/DaMi-FraudDetection-BPJS-CNN/assets/70984049/781e189a-6bea-448f-a5ad-cc31c9bfc95a)

---

🗓️ See the timeline <a
    href="https://docs.google.com/spreadsheets/d/1lCm1ovuSqeUQS-WfJKTlWbghmVE-5M6GxSSM4PMYfkw/edit?usp=sharing">
    <button>here</button></a>!<br />

<br />

```
🧞‍♂️ SemelekeTeam

1. 12S20003 - Marcel Joshua
2. 12S20025 - Irma Naomi
2. 12S20031 - Daniel Andres
3. 12S20048 - Jevania

```
