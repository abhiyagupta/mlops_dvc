**Dog Breed Classification: 
**
This project aims to classify dog breeds using a convolutional neural network (CNN) trained on image data. The project is built using PyTorch Lightning, and Hydra is used for configuration management. The repo also includes test evaluation using a checkpointed model and inference capabilities. Additionally, code coverage (using Codecov) and Docker containerization are integrated.

Key Features

- **PyTorch Lightning Framework**: Utilizes PyTorch Lightning for efficient model training and management.
- **Hydra**: dynamic configuration usuing Hydra.
- **Docker Containerization**: Includes Docker setup for easy deployment and reproducibility across different environments.
- **Code Cover & Pytest**: Ensure robust code quality by integrating pytest and Codecov for automated testing and achieving 70% or higher test coverage.
- **Multi Logger** : Loggers used to log model metrics - of train, eval, and infer are
                    - TensorBoard
                    - CSV
                    - Aim
                    - Mflow

- **HyperParameter**: Hyperparameter optimization and loggers.
- All generated reports are saved in dir called as "Plots" and inference images are saved in dir "Validation_results"

Code to run via docker: 
-docker compose build
-docker compose run train
-docker compose run eval_cm
-docker compose run infer
-docker compose run testing 
-docker compose run generate_plots
-docker compose run generate_reults


Code to run via terminal directly:
- activate virtual environment
- pip install requirements.txt (or do - docker build image . - this will activate virtaul env and also install requirements)
- option 2: 
- docker build -t my_training_image .
- run container - docker run -it --name my_training_container my_training_image /bin/bash
- python src/train.py
- python eval_cm.py
- python infer.py
- python scripts/generate_plots.py
- python scripts/generate_results.py 


Metrics Generated: 

confusion matrix for multiclass problem- classes (number of classes 10): 

![confusion_matrix](https://github.com/user-attachments/assets/21bc02ff-f328-4d01-b442-89d7d9442e55)




metric comparision between different epochs:

![image](https://github.com/user-attachments/assets/0a75fe4a-71e1-4e94-a6b6-dbddecfa3852)




Hyperparameter comparison table:

![image](https://github.com/user-attachments/assets/fbf471b6-1b14-47cc-a80c-21718617252a)


test Metrics:

![image](https://github.com/user-attachments/assets/f9c1a8f1-2d24-4ad3-9a02-562faf881941)



Train accuracy step: 

![image](https://github.com/user-attachments/assets/6ecdf054-ab7f-4208-93bf-ab69545cc4bb)



Train Loss step:


![image](https://github.com/user-attachments/assets/efef0960-f27a-4f44-8e9f-701bc5c670e4)




