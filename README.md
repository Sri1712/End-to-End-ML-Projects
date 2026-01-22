# 🎓 Student Score Prediction – End-to-End Machine Learning Project

This repository contains a complete **end-to-end machine learning project** that predicts a student's exam score based on the number of hours they study. The project follows industry-standard ML engineering practices, including modular code design, logging, exception handling, CI/CD, pipelines, Dockerization, and deployment.

---

## 🚀 Project Overview

The goal of this project is to build a simple and robust ML system that:

* Takes **hours of study** as input
* Predicts the **student's score** as output
* Provides a **production-ready ML pipeline** with:

  * Data ingestion
  * Data validation
  * Data transformation
  * Model training
  * Model evaluation
  * Prediction service
  * User-facing web app

This project is structured to simulate a **real-world ML lifecycle**, from experimentation to deployment.

---

## 🧱 Project Architecture

```
.
├── .github/
│   └── workflows/
│       └── cicd.yaml
├── .venv/
├── artifacts/
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── data_transformation/
│   ├── model_trainer/
│   └── model_evaluation/
├── config/
│   └── config.yaml
├── logs/
│   └── running_log.log
├── research/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_data_validation.ipynb
│   ├── 03_data_transformation.ipynb
│   ├── 04_model_trainer.ipynb
│   ├── 05_model_evaluation.ipynb
│   ├── experiment.ipynb
│   └── trials.ipynb
├── src/
│   ├── bikeSharing/
│   │   ├── __init__.py
│   │   ├── __pycache__/
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── data_ingestion.py
│   │   │   ├── data_validation.py
│   │   │   ├── data_transformation.py
│   │   │   ├── model_trainer.py
│   │   │   └── model_evaluation.py
│   │   ├── config/
│   │   │   └── configuration.py
│   │   ├── constants/
│   │   │   └── __init__.py
│   │   ├── entity/
│   │   │   └── config_entity.py
│   │   ├── pipeline/
│   │   │   ├── prediction.py
│   │   │   ├── stage_01_data_ingestion.py
│   │   │   ├── stage_02_data_validation.py
│   │   │   ├── stage_03_data_transformation.py
│   │   │   ├── stage_04_model_trainer.py
│   │   │   └── stage_05_model_evaluation.py
│   │   └── utils/
│   │       └── common.py
│   └── bikeSharing.egg-info/
├── templates/
│   ├── index.html
│   └── results.html
├── static/
│   └── style.css
├── .gitignore
├── app.py
├── Dockerfile
├── main.py
├── params.yaml
├── requirements.txt
├── schema.yaml
├── setup.py
├── template.py
└── README.md
```

---

## 🔁 End-to-End Workflow

The project follows these implementation steps:

1. **Introduction** <br>
   Defined problem statement and project goals.

2. **GitHub Repo Setup** <br>
   Repository structure, `.gitignore`, and version control.

3. **Project Template Creation** <br>
   Standardized folder layout and boilerplate files.

4. **Project Setup** <br>
   Environment configuration, packaging, and dependency management.

5. **Project Utilities** <br>

   * Centralized logging
   * Custom exception handling
   * Common reusable utility functions

6. **Project Workflow Design** <br>
   Defined ML flow from raw data to predictions.

7. **Notebook Experiments** <br>
   EDA and baseline model experimentation (`research/`).

8. **Component Implementation** <br>

   * **Data Ingestion:**
     Reads raw dataset and stores it in `artifacts/data_ingestion/`.

   * **Data Validation:**
     Performs schema checks and data integrity validation.

   * **Data Transformation:**
     Feature preprocessing and scaling.

   * **Model Training:**
     Trains regression model and stores artifacts.

   * **Model Evaluation:**
     Evaluates model using standard metrics.

9. **Training Pipeline** <br>
   Orchestrates all components end-to-end.

10. **Prediction Pipeline** <br>
    Loads trained model and generates predictions.

11. **User App Implementation** <br>
    Web interface for real-time predictions.

12. **Dockerization** <br>
    Containerized application using Docker.

13. **Deployment** <br>
    Ready for cloud/server deployment.

---

## 📊 Dataset

* Input Feature: `Hours`
* Target Variable: `Score`

This is a simple regression dataset mapping study hours to exam scores.

---

## 🧠 Model

* Type: Regression
* Algorithm: Linear Regression
* Evaluation Metrics:

  * R² Score
  * Mean Squared Error (MSE)
  * Root Mean Squared Error (RMSE)

---

## 🧪 Notebook Experiments

All experiments and initial model prototyping are done in the `research/` directory.

Includes:

* Data exploration
* Visualization
* Feature analysis
* Model training

---

## 🔄 CI/CD Pipeline

A GitHub Actions workflow is defined in:

```
.github/workflows/cicd.yaml
```

This workflow:

* Installs dependencies
* Runs tests (if added)
* Validates code quality
* Builds the application

---

## 📦 Artifacts

All generated outputs are stored in the `artifacts/` directory:

* `data_ingestion/` – raw and split datasets
* `data_validation/` – validation reports
* `data_transformation/` – transformed datasets and preprocessors
* `model_trainer/` – trained model files
* `model_evaluation/` – evaluation metrics

---

## 📝 Logging & Exception Handling

* Centralized logging stored in `logs/`
* Custom exception class for detailed tracebacks

---

## 🔮 Future Improvements

* Add automated tests
* Add model versioning
* Add more features (attendance, sleep hours, etc.)
* Improve UI
* Add monitoring and alerting

---

## 🙌 Acknowledgements

This project was built as a learning-focused end-to-end ML implementation following real-world best practices.

--

## 📫 Contact

**Author:** Srilathaa Vasu <br>
**Email:** [sri1712lathaa@gmail.com](mailto:sri1712lathaa@gmail.com) <br>
**LinkedIn:** Srilathaa vasu <br>

--

⭐ If you find this project helpful, consider giving it a star on GitHub!
