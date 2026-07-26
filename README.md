# showcase-time-series
A small code base for showing time series prediction. 
To do this, we use Walmart sales data: 
<https://github.com/RawatMeghna/Walmart-Sales-Forecasting-using-Best-ML-algorithms/tree/main/Data%20Sources>

Note that we do not replicate the work from the source above and instead opt for another solution
for a simple and clean demonstration of data processing and machine learning.

## Instructions
### 1. Clone the repository
Download the codebase locally:
```
git clone https://github.com/cruzas/showcase-time-series
cd showcase-time-series
```

### 2. Create an isolated virtual environment
```
python3 -m venv .venv
```

### 3. Activate the environment and install packages
Activate the virtual workspace and install all required libraries:
- Mac/Linux: `source .venv/bin/activate`
- Windows: `.venv\Scripts\activate`

Then, run: 
```
pip3 install -r requirements.txt
```

### 4. Launch the Jupyter notebook
Start the interactive session within the isolated environment:
```
jupyter lab notebooks/sales_forecasting.ipynb
```

