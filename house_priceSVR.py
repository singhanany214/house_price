{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# House Price Prediction using SVR\n",
    "\n",
    "Fix: scale the target `y` along with `X`. SVR is sensitive to target scale."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.svm import SVR\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load dataset\n",
    "df = pd.read_csv(\"house_price.csv\")\n",
    "df = df.dropna()\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Features and target\n",
    "X = df[['area', 'bedrooms', 'bathrooms']]\n",
    "y = df['price']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Scale X\n",
    "scaler_X = StandardScaler()\n",
    "X_train = scaler_X.fit_transform(X_train)\n",
    "X_test  = scaler_X.transform(X_test)\n",
    "\n",
    "# Scale y  <-- this is the fix\n",
    "scaler_y = StandardScaler()\n",
    "y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Basic SVR model\n",
    "model = SVR()\n",
    "model.fit(X_train, y_train_scaled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Predict, then inverse-transform back to original price units\n",
    "y_pred_scaled = model.predict(X_test)\n",
    "y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluation\n",
    "mae  = mean_absolute_error(y_test, y_pred)\n",
    "mse  = mean_squared_error(y_test, y_pred)\n",
    "rmse = np.sqrt(mse)\n",
    "r2   = r2_score(y_test, y_pred)\n",
    "\n",
    "print(\"MAE:\", mae)\n",
    "print(\"MSE:\", mse)\n",
    "print(\"RMSE:\", rmse)\n",
    "print(\"R2 Score:\", r2)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
