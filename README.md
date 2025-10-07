# Green-Scan 🌿

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27%2B-red.svg)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered web application for instant plant disease detection, providing diagnoses, remedies, and potential yield impact estimates.



---

## Table of Contents
- [About The Project](#about-the-project)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [File Structure](#file-structure)
- [License](#license)
- [Contact](#contact)

---

## About The Project

Green-Scan is a user-friendly tool designed to help farmers, gardeners, and agriculture enthusiasts quickly identify plant diseases from leaf images. By leveraging a powerful deep learning model, this application provides an accurate diagnosis within seconds. Beyond identification, Green-Scan offers actionable advice, including recommended remedies and estimates of the potential impact on crop yield, empowering users to make informed decisions and protect their plants.

This project aims to democratize precision agriculture, making advanced AI tools accessible to everyone for healthier crops and more sustainable farming practices.

---

## Key Features

* **Accurate Disease Classification:** Utilizes a Convolutional Neural Network (CNN) trained on over 87,000 images to identify 38 different plant diseases.
* **Dual Image Input:** Users can either upload an image file directly from their device or paste a URL for analysis.
* **Confidence Score:** Displays the model's confidence level for each prediction.
* **Actionable Insights:** Provides recommended remedies and potential yield impact information for diagnosed diseases.
* **PDF Report Generation:** Users can download a professional, timestamped PDF report summarizing the diagnosis and recommendations.
* **Session History:** The sidebar keeps a running list of the predictions made during the current session for easy reference.

---

## Tech Stack

This project is built with the following technologies:

* **Backend:** Python
* **Web Framework:** Streamlit
* **Deep Learning:** TensorFlow & Keras
* **Image Processing:** Pillow
* **PDF Generation:** FPDF2

---

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Python 3.9 or higher
* `pip` package manager

### Installation

1.  **Clone the repository**
    ```sh
    git clone [https://github.com/your-username/Green-Scan.git](https://github.com/your-username/Green-Scan.git)
    cd Green-Scan
    ```
2.  **Create and activate a virtual environment** (recommended)
    ```sh
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install the required packages**
    ```sh
    pip install -r requirements.txt
    ```
4.  **Download the model file**
    Ensure you have the trained model file `trained_plant_disease_model.keras` in the root directory of the project.

5.  **Run the Streamlit app**
    ```sh
    streamlit run main.py
    ```
    Your browser should open with the application running.

---

## Usage

1.  Navigate to the **Disease Recognition** page from the sidebar.
2.  Choose your input method: **Upload an Image** or **Provide Image URL**.
3.  Upload your image or paste the URL. The image will be displayed.
4.  Click the **Predict** button.
5.  View the diagnosis, confidence score, recommended action, and potential yield impact.
6.  You can download a PDF report or view your prediction in the sidebar history.

---

## File Structure
```
PLANT-DISEASE-CLASSIFICATION/
│
├── .venv/                      # Virtual environment folder
├── data/                       # (Optional) Folder for your dataset
│
├── trained_plant_disease_model.keras  # The trained model file
├── main.py                     # Main Streamlit application script
├── requirements.txt            # List of Python dependencies
├── home_page.jpeg              # Image for the home page
└── README.md                   # This file
```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Contact

Harsh Srivastava - [Your LinkedIn URL] - [Your Email]

Project Link: [https://github.com/your-username/Green-Scan](https://github.com/your-username/Green-Scan)