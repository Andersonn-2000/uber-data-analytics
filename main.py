from src.scripts.clean_data import Preprocess

from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

preprocess = Preprocess("/home/anderson/Documents/uber_dataset/data/ncr_ride_bookings.csv")

drop_original_cols = [
                'Booking ID', 'Customer ID', 'Pickup Location', 'Drop Location',
                'Cancelled Rides by Customer', 'Reason for cancelling by Customer',
                'Cancelled Rides by Driver', 'Driver Cancellation Reason',
                'Incomplete Rides', 'Incomplete Rides Reason', 'Date', 'Time', 'datetime'
            ]

categorical_cols = ['Vehicle Type', 'Payment Method', 'Booking Status']

X, y, _ = preprocess.run_pipeline(
    drop_original_cols=drop_original_cols,
    categorical_cols=categorical_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Balanceamento
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=neg/pos,
        random_state=42,
        n_jobs=-1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
print(f"Classification Report:\n{classification_report(y_test, y_pred, digits=3)}")
print(f"Average Precision Score:\n{average_precision_score(y_test, y_pred)}")
