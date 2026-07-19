Data-Driven-Traffic-Prediction-using-Graph-Neural-Networks-AI
------------------------------------------------------------

Traffic congestion is one of the most persistent challenges in modern cities, affecting productivity, increasing fuel consumption, and contributing to pollution.
The Data Driven Traffic Prediction AI project demonstrates how artificial intelligence can forecast traffic speeds using a 
synthetic dataset that simulates real‑world road conditions.
By training advanced AI models and deploying them in a user‑friendly web application, the project makes traffic forecasting accessible to both technical and non‑technical users. The system allows anyone to input or generate sample traffic data and instantly view predictions in both numerical and graphical formats.



##Project  Overview

This project presents a **Data-Driven Traffic Prediction System** powered by **Graph Neural Networks (GNNs)** for forecasting traffic conditions. The system models road networks as graphs to capture both **spatial dependencies** between connected roads and **temporal patterns** in traffic flow.

Unlike many traditional traffic prediction projects that rely on public traffic sensor datasets, this project utilizes a **synthetically generated (Generative) traffic dataset** designed to simulate realistic traffic scenarios. This enables experimentation, rapid prototyping, and model evaluation without requiring access to proprietary transportation data.

The project demonstrates how Graph Neural Networks can learn complex relationships within transportation networks and generate accurate traffic forecasts suitable for Intelligent Transportation Systems (ITS) and Smart City applications.

---

## Features

* Graph-based traffic network modeling
* Synthetic (Generative) traffic dataset
* Data preprocessing and normalization
* Spatial learning using Graph Neural Networks
* Time-series traffic prediction
* Model training and evaluation
* Performance visualization
* Modular and extensible project structure

---

## Project Objectives

The primary objectives of this project are to:

* Predict future traffic conditions using Graph Neural Networks.
* Capture spatial relationships among road segments.
* Learn temporal traffic patterns from sequential data.
* Evaluate prediction performance using standard regression metrics.
* Demonstrate AI applications in intelligent transportation systems.

---

## Technologies Used

| Category                   | Technology                 |
| -------------------------- | -------------------------- |
| Programming Language       | Python                     |
| Deep Learning              | PyTorch                    |
| Graph Learning             | PyTorch Geometric (PyG)    |
| Data Processing            | Pandas, NumPy              |
| Visualization              | Matplotlib, Seaborn        |
| Machine Learning Utilities | Scikit-learn               |
| Development Environment    | Jupyter Notebook / VS Code |

---

## Project Structure

```text
Data-Driven-Traffic-Prediction-using-Graph-Neural-Networks-AI/
│
├── data/
│   ├── synthetic_dataset.csv
│   └── processed_data/
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
│
├── models/
│   ├── gnn_model.py
│   └── saved_models/
│
├── utils/
│   ├── preprocessing.py
│   ├── graph_builder.py
│   └── metrics.py
│
├── results/
│   ├── plots/
│   └── predictions/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Dataset

### Synthetic (Generative) Dataset

This project uses a **synthetically generated traffic dataset** rather than real-world traffic sensor data.

The dataset was created to simulate realistic urban traffic behavior, including:

* Vehicle count
* Average traffic speed
* Traffic density
* Road connectivity
* Time of day
* Day of week
* Weather conditions (optional)
* Congestion level

Using a synthetic dataset allows researchers and students to:

* Experiment safely without privacy concerns.
* Reproduce experiments consistently.
* Scale the dataset as needed.
* Simulate various traffic scenarios.

---

## Methodology

The workflow consists of the following stages:

1. Generate synthetic traffic data.
2. Preprocess and normalize the dataset.
3. Construct the road network graph.
4. Create temporal input sequences.
5. Train the Graph Neural Network.
6. Predict future traffic conditions.
7. Evaluate model performance.
8. Visualize prediction results.

---

## Graph Representation

The transportation network is represented as a graph.

* **Nodes:** Traffic sensors, intersections, or road segments
* **Edges:** Road connectivity between nodes
* **Node Features:** Traffic speed, vehicle count, density, etc.
* **Target:** Future traffic condition prediction

This representation enables the model to capture spatial dependencies that traditional machine learning models often overlook.

---

## Model Architecture

```text
Synthetic Traffic Dataset
            │
            ▼
    Data Preprocessing
            │
            ▼
     Graph Construction
            │
            ▼
 Graph Neural Network Layer(s)
            │
            ▼
 Temporal Feature Learning
            │
            ▼
 Traffic Prediction
            │
            ▼
 Model Evaluation
```

---

## Evaluation Metrics

Model performance is evaluated using standard regression metrics:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)
* Coefficient of Determination (R² Score)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/SohelRana-aiub-Pro/Data-Driven-Traffic-Prediction-using-Graph-Neural-Networks-AI.git

cd Data-Driven-Traffic-Prediction-using-Graph-Neural-Networks-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the preprocessing pipeline:

```bash
python preprocessing.py
```

Train the model:

```bash
python train.py
```

Evaluate the model:

```bash
python evaluate.py
```

Or execute the Jupyter notebooks provided in the `notebooks/` directory.

---

## Results

The Graph Neural Network successfully learns spatial and temporal traffic patterns from the synthetic dataset.

Typical outputs include:

* Predicted traffic flow
* Traffic trend visualization
* Training and validation loss curves
* Model performance metrics
* Prediction vs. actual comparison plots

---

## Applications

Potential applications include:

* Smart City infrastructure
* Intelligent Transportation Systems (ITS)
* Traffic congestion prediction
* Route optimization
* Emergency response planning
* Urban traffic management
* Transportation analytics

---

## Future Improvements

Potential enhancements include:

* Integration with real-world traffic datasets (METR-LA, PEMS-BAY, etc.)
* Graph Attention Networks (GAT)
* Temporal Graph Networks (TGN)
* Spatio-Temporal Graph Convolutional Networks (STGCN)
* Transformer-based traffic forecasting
* Real-time prediction dashboard
* Deployment using FastAPI or Flask
* Interactive visualization dashboard

---

## Acknowledgements

This project was developed as a research and educational implementation to explore Graph Neural Networks for traffic forecasting using synthetic traffic data.

---

## License

This project is licensed under the MIT License.

---

## Author

**Sohel Rana**

* AI & Machine Learning Enthusiast
* Researcher in Data Science and Graph Neural Networks

GitHub: https://github.com/SohelRana-aiub-Pro

---

## Citation

If you use this project in your research or academic work, please cite this repository appropriately.

















For Implement in Local Server/PC , follow the 'Project code structure & Requirements Commands'


<img width="642" height="431" alt="Project Code Structure" src="https://github.com/user-attachments/assets/ca6d9dd2-ffb1-43ac-b741-54542af1abe4" />

Sample Predicted Outputs;
<img width="758" height="563" alt="App-output 1" src="https://github.com/user-attachments/assets/66aedb0c-26fe-4cb3-b8ae-35cc0cbe581a" />



<img width="655" height="343" alt="App-output 2" src="https://github.com/user-attachments/assets/8f68b9ca-aa77-4f2e-87e3-05de58d24931" />


<img width="580" height="477" alt="App-output 3" src="https://github.com/user-attachments/assets/3d586e46-8ebe-45c1-88e6-d572d9c391fd" />

<img width="646" height="388" alt="App-output 4" src="https://github.com/user-attachments/assets/434cb99c-d667-4416-bd70-5216d7560930" />
