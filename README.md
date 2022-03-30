# Audio-Classification
Classifying the Ubansound8k dataset and making predictions

This dataset can be downloaded from the following link: https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbWxyc21RNDRTLVZ6ekp6ODJTWmY3UkwtNGJSUXxBQ3Jtc0tuckljZmdhVE90RzZiZHItM1Z1TmZLSDMxdnNSQmZVNkRidW9mcWJ4QUVWSy15ZWlRbGFSRC1ldHpNQUNya3FOSFdEYm9hNHZQRHZFWkxoSnVwLXQ5aXRzMDUtS0xvNFFYNFJGVUM4dTIxZlNYSGRZcw&q=https%3A%2F%2Furbansounddataset.weebly.com%2Fdownload-urbansound8k.html&v=uTFU7qThylE

This dataset contains sounds of:

1. engine_idling  - 1000    

2. dog_bark        -    1000

3. street_music     -   1000

4. children_playing  -   1000

5. jackhammer      -    1000

6. drilling         -   1000

7. air_conditioner   -  1000

8. siren              -  929

9. car_horn            - 429

10. gun_shot           -  374

## Steps followed in this project are:

### 1. Performing EDA

Firslty I took a single sound of dog bark and performed analysis by checking the samplerate and wavesamplerate

![image](https://user-images.githubusercontent.com/63282184/160834038-3b0dea4a-3ab7-45ef-9efe-87d20842f885.png)

### 2. Loading the dataset and also the metadata

![image](https://user-images.githubusercontent.com/63282184/160834122-8de6bb43-e50f-4bd6-b674-c37322b3be58.png)


### 3. Extrating the feature using Mel-Frequency Cepstral Coefficients

![image](https://user-images.githubusercontent.com/63282184/160834284-7456dd6b-6419-4b65-8aad-9a2dc8736445.png)

### 4. Converting the extracted features to pandas dataframe

![image](https://user-images.githubusercontent.com/63282184/160834445-d34f8544-49df-45e8-aff5-5482e3d54302.png)

### 5. Split the dataset

### 6. Label encoding the data

### 7. Building the model

![image](https://user-images.githubusercontent.com/63282184/160834712-d315a6a7-aa13-40f1-b055-b5f6ec9ea908.png)

### 8. Training the model
![image](https://user-images.githubusercontent.com/63282184/160834812-870efa51-63ac-4081-8a43-c8b0ce74a28c.png)



