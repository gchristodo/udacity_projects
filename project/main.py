"""
This is the main/wrapper function.
"""
import time
import joblib
from churn_library import (import_data,
                           perform_eda,
                           encoder_helper,
                           perform_feature_engineering,
                           train_models,
                           classification_report_image,
                           feature_importance_plot
                           )
from constants import constants


if __name__ == "__main__":
    cat_columns = constants['cat_columns']
    quant_columns = constants['quant_columns']
    PATH = constants['path']
    df = import_data(PATH)
    perform_eda(df)
    df = encoder_helper(df, cat_columns, 'Churn')
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')
    train_models(X_train, X_test, y_train, y_test)
    time.sleep(10)
    rfc_model = joblib.load('./models/new/rfc_model.pkl')
    lr_model = joblib.load('./models/new/logistic_model.pkl')
    y_test_preds_rf = rfc_model.predict(X_test)
    y_train_preds_rf = rfc_model.predict(X_train)
    y_train_preds_lr = lr_model.predict(X_train)
    y_test_preds_lr = lr_model.predict(X_test)
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    dataframe = df[constants['keep_cols']]
    feature_importance_plot(rfc_model, dataframe, "images/results/Feat_Imp_RFC.png")
    feature_importance_plot(lr_model, dataframe, "images/results/Feat_Imp_LR.png")
