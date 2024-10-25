**Dog Breed Classification: ** This project aims to classify dog breeds using a convolutional neural network (CNN) trained on image data. The project is built using PyTorch Lightning, and Hydra is used for configuration management. The repo also includes test evaluation using a checkpointed model and inference capabilities. Additionally, code coverage (using Codecov) and Docker containerization are integrated.

Key Features

PyTorch Lightning Framework: Utilizes PyTorch Lightning for efficient model training and management.
Hydra: dynamic configuration usuing Hydra.
Docker Containerization: Includes Docker setup for easy deployment and reproducibility across different environments.
DVC : Data Version control added
generated metrics and PLots

-commands to run in terminal : dvc is integrated with docker compose - " dvc repro" 
- with this the entire project runs - which is pulling data from gdrive and running train, eval and infer.
- to generate metrics : "python scripts/generate_plots.py"

trained model for only 1 epochto save time.
logs are saved in csv 
best model is saved in model checkpoints and infer picks up this saved model from checkpoint.

**test_metrics.md**

![image](https://github.com/user-attachments/assets/8392a05a-4480-46a6-8b53-f01bd2e3183a)

**plots**

![image](https://github.com/user-attachments/assets/66dcd34c-c34b-4c2d-bf06-ce6e4bd21208)


![image](https://github.com/user-attachments/assets/4755d66b-dc57-4062-8419-8e5e6bd00ff0)
