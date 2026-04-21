 ⚽ FIFA Transfer Market Value Predictor

A full-stack Machine Learning web app that predicts the **transfer market value of football players** using multiple models trained on the FIFA 19 dataset.

🚀 Features

 Predict player market value using ML models
 Select **famous players** or enter **custom stats**
 8+ dynamic visualizations for analysis
 Model comparison:
    Linear Regression
    Random Forest
    Gradient Boosting (Best)
    PCA-based dimensionality reduction
    Classification into value categories (Low → Elite)

🖥️ Tech Stack

Frontend
HTML, CSS, JavaScript

Backend
Python, Flask

ML & Data**
scikit-learn
pandas, numpy

Visualization**
matplotlib, seaborn

📂 Project Structure

```
├── app.py
├── data.csv
├── templates/
│   └── index.html
```

⚙️ How It Works

1. Loads and cleans dataset (currency, missing values) 
2. Uses features like age, overall, skills, etc.
3. Trains regression + classification models
4. Applies PCA for dimensionality reduction
5. Displays predictions + graphs on UI


📊 Models

| Model             | Purpose            |
| ----------------- | ------------------ |
| Linear Regression | Baseline           |
| Random Forest     | Better accuracy    |
| Gradient Boosting | ⭐ Best performance |


▶️ Run Locally

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
python app.py
```
Open in browser:
```
http://127.0.0.1:5000
```

📈 Visualizations

* Overall vs Value
* Skill Radar
* Similar Players
* Age vs Value
* Model Comparison
* Feature Importance
* Position vs Player
* Value Distribution

📌 Dataset

* FIFA 19 dataset (~18,000 players)

💡 Learning Outcomes

* End-to-end ML project
* Regression vs Classification
* PCA
* Model evaluation (R², MAE, Kappa, MCC)

🔮 Future Improvements

* Deploy online
* Add real-time data
* Improve UI (React)
* Add deep learning

👨‍💻 Author

Sanket Kumar Singh
