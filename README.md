# RSPP - Real Estate Sales Price Prediction

Welcome to the Real Estate Sales Price Prediction (RSPP) repository. This project hosts a machine learning model that predicts real estate sales prices based on various property features. The model is deployed on Hugging Face and can be accessed via an API.

## Features

The model predicts the sales price of a property using the following features:

- **Number of bedrooms** (`feature1`)
- **Number of bathrooms** (`feature2`)
- **Square footage of the living area** (`feature3`)
- **Square footage of the lot** (`feature4`)
- **Number of floors** (`feature5`)
- **Condition of the house** (`feature6`)
- **Square footage of the basement** (`feature7`)
- **Year built** (`feature8`)
- **Whether it has been renovated** (`feature9`)
- **City name** (`feature10`)

## Getting Started

### Installation

To interact with the API, you'll need to install the `gradio_client` Python package. If you don't have it installed, use the following command:

```bash
pip install gradio_client
