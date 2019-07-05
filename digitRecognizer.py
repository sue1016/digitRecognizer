import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
from .cnn import DigitCNN
from PIL import Image
# third-party library
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision

class DigitRecognizer():
    def __init__(self):
        self.data_tf_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [-0.25])])
        self.cnn = DigitCNN()
        self.cnn.load_state_dict(torch.load("digit_model.pkl"))
        self.shrink_size = {"height_up": 10, "height_down": 10, "width_up": 10, "width_down": 10}

    def recognize_cells(self, cellPaths):
        cellReturns = []
        if cellPaths == None:
            return None
        for cellPath in cellPaths:
            cellReturn = {}
            result, confidence = self.recognize_cell(cellPath)
            cellReturn['path'] = cellPath
            cellReturn['result'] = result
            cellReturn['confidence'] = confidence
            cellReturns.append(cellReturn)
        return cellReturns

    def recognize_cell(self, cellPath):
        cell_result = None
        cell_confidence = None
        the_most_possible_digits = []
        confidences = []
        # 1.cell_image_to_normalized_digit_images
        normalized_digit_images = self.cell_image_to_normalized_digit_images(cellPath)
        for normalized_digit_image in normalized_digit_images:
            # 2.normalized_digit_image_to_preprocessed_digit_image
            preprocessed_digit_image = self.normalized_digit_image_to_preprocessed_digit_image(normalized_digit_image)
            # 3.use cnn model to  predict preprocessed digit image
            the_most_possible_digit, confidence_of_the_most_possible_digit = self.cnn_predict_preprocessed_digit_image(
                preprocessed_digit_image)
            the_most_possible_digits.append(int(the_most_possible_digit))
            confidences.append(confidence_of_the_most_possible_digit)

        cell_result, cell_confidence = self.predicted_digit_images_results_to_final_cell_result(
            the_most_possible_digits, confidences)
        return cell_result, cell_confidence

    def cell_image_to_normalized_digit_images(self, cell_image_file):
        # 切出单个数字并补成正方形，返回得到的图像（ndarray)的list
        cell_image = cv2.imread(cell_image_file)
        grayed_cell_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        height, width = len(grayed_cell_image), len(grayed_cell_image[0])
        grayed_cell_image = grayed_cell_image[self.shrink_size["height_down"]:height - self.shrink_size["height_up"],
                            self.shrink_size["width_down"]:width - self.shrink_size["width_up"]]
        ret, thresh = cv2.threshold(grayed_cell_image, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        raw_digit_images_rec_cordinates = self.get_digits_rectangles(contours, hierarchy)
        normalized_digit_images = []
        for raw_digit_image_rec_cordinate in raw_digit_images_rec_cordinates:
            x, y, w, h = raw_digit_image_rec_cordinate
            raw_digit_image = grayed_cell_image[y:y + h, x:x + w]
            if h > w:
                length = h
            else:
                length = w
            normalized_digit_image = np.zeros((length, length))
            # normailized:to be squared and centered
            for i in range(length):
                for j in range(length):
                    normalized_digit_image[i, j] = 255
            for i in range(h):
                for j in range(w):
                    normalized_digit_image[int((length - h) / 2) + i, int((length - w) / 2) + j] = raw_digit_image[i, j]
            normalized_digit_images.append(normalized_digit_image)
        return normalized_digit_images

    def normalized_digit_image_to_preprocessed_digit_image(self, normalized_digit_image):
        preprocessed_digit_image = Image.fromarray(normalized_digit_image)
        preprocessed_digit_image = preprocessed_digit_image.convert('L')
        preprocessed_digit_image = preprocessed_digit_image.resize((28, 28))
        return preprocessed_digit_image

    def cnn_predict_preprocessed_digit_image(self, preprocessed_digit_image):
        # image to tensor
        to_tensor = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [-0.25])])
        cnn_input_image_as_tensor = to_tensor(preprocessed_digit_image).float()
        cnn_input_image_as_tensor = cnn_input_image_as_tensor.unsqueeze_(0)
        # 预测分类label
        cnn_predict_returns, _ = self.cnn(cnn_input_image_as_tensor)
        the_most_possible_digit = torch.max(cnn_predict_returns, 1)[1].data.numpy().squeeze()

        # 预测各类别概率
        confidences_for_all_possible_digits = torch.nn.functional.softmax(cnn_predict_returns, dim=1)
        # 预测各类别概率，寻找目标概率
        confidences_for_all_possible_digits = confidences_for_all_possible_digits.data[0]
        confidence_of_the_most_possible_digit = 0
        for confidence_for_each_possible_digit in confidences_for_all_possible_digits:
            if confidence_for_each_possible_digit > confidence_of_the_most_possible_digit:
                confidence_of_the_most_possible_digit = confidence_for_each_possible_digit

        return the_most_possible_digit, confidence_of_the_most_possible_digit

    def predicted_digit_images_results_to_final_cell_result(self, the_most_possible_digits, confidences):
        if len(the_most_possible_digits) > 1:
            cell_result = 10 * the_most_possible_digits[0] + the_most_possible_digits[1]
        else:
            try:
                cell_result = the_most_possible_digits[0]
            except:
                cell_result = 0
        # 返回最小置信度
        min_confidence = 1
        for confidence in confidences:
            if confidence < min_confidence:
                min_confidence = confidence
        cell_confidence = min_confidence
        # temp 解决单元格为空的情况
        if min_confidence < 0.25:
            cell_result = 0
        # confidence是一个tensor(float)
        return cell_result, float('%.4f' % cell_confidence)

    def set_train_params(self,epoch=1,batch_size=50,learning_rate=0.001,seed=1):
        torch.manual_seed(seed)
        return epoch,batch_size,learning_rate,seed
    def get_train_data_loader(self,batch_size,mode="MINST"):
        if mode == "MINST":
            # todo: 通过查看指定目录是否有MINST数据集来决定这个参数
            DOWNLOAD_MNIST = True
            train_data = torchvision.datasets.MNIST(
                root='./mnist/',
                train=True,  # this is training data
                transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
                # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
                download=DOWNLOAD_MNIST,
            )
            # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
            train_data_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        elif mode == "USER_DEFINED":
            pass

        return train_data_loader
    def get_test_data(self,mode="MINST"):
        if mode == "MINST":
            test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
            test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[
                     :10000] / 255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
            test_y = test_data.test_labels[:10000]
        elif mode =="USER_DEFINED":
            pass
        return test_x,test_y
    def test_cnn_model(self,mode="MINST"):
        cnn_model = self.cnn
        test_x, test_y = self.get_test_data(mode)
        test_output, last_layer = cnn_model(test_x)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        accuracy = sum(pred_y == test_y) / float(test_y.size(0))
        return accuracy

    def train_cnn_model(self,mode):
        cnn_model = self.cnn
        epoch, batch_size, learning_rate, seed = self.set_train_params(epoch=1,batch_size=50,learning_rate=0.001,seed=1)
        train_data_loader = self.get_train_data_loader(batch_size=batch_size,mode=mode)
        optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)  # optimize all cnn parameters
        loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
        # training and testing
        for epoch in range(epoch):
            for step, (x, y) in enumerate(train_data_loader):  # gives batch data, normalize x when iterate train_loader
                b_x = Variable(x)  # batch x
                b_y = Variable(y)  # batch y

                output = cnn_model(b_x)[0]  # cnn output
                loss = loss_func(output, b_y)  # cross entropy loss
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                # for testing the accuracy on the test set
                if step % 50 == 0:
                    accuracy = self.test_cnn_model(mode=mode)
                    print('Epoch: ' + str(epoch) + '| train loss: %.4f' + str(loss) + '| test accuracy: %.2f' + str(
                        accuracy))
        # 保存训练模型
        torch.save(cnn_model.state_dict(), "digit_model_4_64.pkl")




