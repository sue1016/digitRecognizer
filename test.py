from digit_module import digitRecognizer

digitRecognizer = digitRecognizer.DigitRecognizer()
# digitRecognizer.train_cnn_model(mode="USER_DEFINED")
# digitRecognizer.test_cnn_model(mode="MINST")
digitRecognizer.recognize_cell("/Users/yuandarong/mycode/handwriteTableImageToExcel/papersPic/origin/20190708085219/cell6.jpg")