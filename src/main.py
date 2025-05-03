from model_driver import LSTMDriver, RNNDriver, SVMDriver, RandomForestDriver

def main():
    model = "rf"
    action = "eval"

    drivers = {
        "lstm": LSTMDriver,
        "rnn": RNNDriver,
        "svm": SVMDriver,
        "rf": RandomForestDriver,
    }

    if model not in drivers:
        raise ValueError(f"❌ Model không hợp lệ: {model}")
    if action not in ["train", "eval"]:
        raise ValueError(f"❌ Action không hợp lệ: {action}")
    driver = drivers[model]()  # tạo instance

    if action == "train":
        driver.train()
        driver.evaluate()
    else:
        driver.evaluate()

if __name__ == "__main__":
    main()
