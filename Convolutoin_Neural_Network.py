import dataloader
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import datetime

class Convolution():
    def __init__(self, input_shape, kernel_size, depth): # Convolutional((50, 1, 28, 28), 5, 6)
        batch_size, input_depth, input_height, input_width = input_shape 
        self.batch_size = batch_size
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (batch_size, depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size) 
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)     
        self.learning_rate = 0.005
        self.rho = 0.5 # 0.99 
        self.vx_1 = 0 # kernel momentum
        self.vx_2 = 0 # bias momentum
        self.stride = 1
        self.pad = 0

    def im2col(self, input_data, filter_h, filter_w, stride=1, pad=0): 
        batch, channel, height, weight = input_data.shape 
        out_h = int((height + 2 * pad - filter_h)/stride + 1) 
        out_w = int((weight + 2 * pad - filter_w)/stride + 1)  
        img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant', constant_values=0) 
        col = np.zeros((batch, channel, filter_h, filter_w, out_h, out_w)) 
        for y in range(filter_h): 
            y_max = y + stride * out_h 
            for x in range(filter_w): 
                x_max = x + stride * out_w 
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride] 
        
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch * out_h * out_w, -1) 
        return col

    def col2im(self, col, input_shape, filter_h, filter_w, stride=1, pad=0):
        batch, channel, height, weight = input_shape
        out_h = (height + 2*pad - filter_h)//stride + 1
        out_w = (weight + 2*pad - filter_w)//stride + 1
        col = col.reshape(batch, out_h, out_w, channel, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
        img = np.zeros((batch, channel, height + 2*pad + stride - 1, weight + 2*pad + stride - 1))

        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

        return img[:, :, pad:height + pad, pad:weight + pad]

    def forward(self, input): # input = (8, 1, 28, 28), (8, 6, 12, 12)
        self.input = input
        batch, channel, height, weight = self.input_shape # input_shape = (8, 1, 28, 28) / (8, 6, 12, 12)
        f_batch, channel, f_height, f_weight  = self.kernels_shape
        self.output = np.copy(self.biases)
        #self.output2 = np.copy(self.biases)

        out_h = int(1 + (height + 2 * self.pad - f_height) / self.stride)
        out_w = int(1 + (weight + 2 * self.pad - f_weight) / self.stride)

        col = self.im2col(input, f_height, f_weight, self.stride, self.pad)
        col_W = self.kernels.reshape(f_batch, -1).T
        out = np.dot(col, col_W)
        out = out.reshape(batch, out_h, out_w, -1).transpose(0,3,1,2)
        out += self.biases
        return out

    def backward(self, output_gradient): # output_gradient : (8, 6, 8, 8)
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape) # input : (8, 6, 12, 12)

        for i in range(self.batch_size):
            for j in range(self.depth):  
                kernels_gradient[j] = self.conv_correlate_2d(self.input[i], output_gradient[i]) # self.input = (8,6,12,12) output_gradoent = (8,6,8,8)
                input_gradient[i] -= self.conv_full_convolution_2d(output_gradient[i], self.kernels[j]) # output_gradoent = (8,6,8,8) self.kernels = (8,6,5,5)

            kernels_gradient = kernels_gradient / self.batch_size
            input_gradient = input_gradient / self.batch_size
            
        self.kernels -= self.learning_rate * kernels_gradient
        self.biases -= self.learning_rate * output_gradient

        return input_gradient

    def conv_correlate_2d(self, prev_input, W): # output : (5, 5)
        image_depth, input_height, input_width = prev_input.shape # 1*28*28 input_depth,
        kernel_depth, kernel_size, kernel_size = W.shape

        # calculate output size
        output_height = input_height - kernel_size + 1
        output_weight = input_width - kernel_size + 1   
        
        output_shape = (output_height, output_weight) #(self.depth, output_height, output_weight)
        output = np.zeros(output_shape) # 26*26

        for i in range(0, output_height):
            for j in range(0, output_weight):
                h_start, w_start = i, j
                h_end, w_end = h_start + kernel_size, w_start + kernel_size
                correlation = np.multiply(prev_input[:, h_start:h_end, w_start:w_end], W)
                output[i, j] = np.sum(correlation)
        return output 

    def conv_full_convolution_2d(self, output_gradient, kernels): 

        image_depth, input_height, input_width = output_gradient.shape 
        kernel_depth, kernel_size, kernel_size = kernels.shape 
        
        # zero padding -> output_gradient // 6, 8 ,8 -> 6, 16, 16
        npad= ((0, 0), (kernel_size - 1, kernel_size - 1), (kernel_size - 1, kernel_size - 1))
        zero_padding= np.pad(output_gradient, npad,'constant', constant_values=(0))
        
        _, input_height, input_width = zero_padding.shape

        # calculate output size
        output_height = input_height - kernel_size + 1 
        output_weight = input_width - kernel_size + 1
        
        output_shape = (output_height, output_weight) 
        output = np.zeros(output_shape)

        # rotate kernel 180
        np.rot90(kernels)
        np.rot90(kernels)

        for i in range(0, output_height): 
            for j in range(0, output_weight):
                h_start, w_start = i, j
                h_end, w_end = h_start + kernel_size, w_start + kernel_size            
                correlation = np.multiply(zero_padding[:, h_start:h_end, w_start:w_end], kernels) # (6, 5, 4) (6, 5, 5) (6, 8, 8)
                output[i, j] = np.sum(correlation)

        return output 

class CNN(Convolution):
    def __init__(self):
        self.height = 0
        self.weight = 0
        self.depth_1 = 5
        self.depth_2 = 10
        self.kernel_size = 5
        self.batch_size = 50 # 30
        self.W3 = self.linear_layer(80, 10) # depth 6 -> (96, 10) / depth 3 -> (48, 10) / depth 10 -> (160, 10)
        self.b3 = self.linear_layer(self.batch_size, 10) # 8, 10
        self.max_pooling_stride = 2
        self.max_pooling_kernel_size = 2
        self.train_cost = []
        self.test_cost = []
        self.train_cost_epoch = []
        self.test_cost_epoch = []
        self.learning_rate = 0.005
        self.rho = 0.5 # 0.99 
        self.vx_3 = 0
        self.vx_4 = 0

        # using feed forward
        self.output_relu_1 = np.zeros(self.depth_1) # (self.depth, 28 - self.kernel_size + 1, 28 - self.kernel_size + 1)
        self.output_relu_2 = np.zeros(self.depth_2)
        self.d_output_relu_1 = np.zeros(self.depth_1)
        self.d_output_relu_2 = np.zeros(self.depth_2)

        self.convolution_layer_1 = Convolution((self.batch_size, 1, 28, 28), self.kernel_size, self.depth_1) # (input_shape, kernel_size, depth)
        self.convolution_layer_2 = Convolution((self.batch_size, 5, 12, 12), self.kernel_size, self.depth_1)

        #accuracy
        self.train_accuracy = 0
        self.test_accuracy = 0

        # visualization
        self.y_predic = []
        self.y_true = []

        # top 3 list
        self.map_list = []
        self.top_3 = []

        # use max pooling using im2col and col2im
        self.max_pool_h = 2
        self.max_pool_w = 2
        self.max_pool_stride = 2
        self.pad = 0

    def train(self):
        for epoch in range(80):
            data = dataloader.Dataloader(".", is_train = True, shuffle=True, batch_size=self.batch_size)
            for i in range(0, int(len(data.images) / self.batch_size)): # 60000/8 = 7500번 iteration = 1 epoch 60000/100 = 600
               
                data = dataloader.Dataloader(".", is_train = True, shuffle=True, batch_size=self.batch_size)
                _, _, self.height, self.weight = data.images.shape
                input_images, input_labels = data.__getitem__(i)
                loss = self.feed_forward(input_images, input_labels, is_train = True)
                self.back_propagation()

                if i % 10 == 0:
                    now = datetime.datetime.now()
                    print(now, " ,", i, " 번째 iteration -> train loss : ", round(loss/self.batch_size, 2))
        
            # calculate test loss & accuracy 
            now = datetime.datetime.now()
            temp_train = sum(self.train_cost) / (len(data.images) / self.batch_size)
            self.train_cost = []
            self.train_cost_epoch.append(round(temp_train, 2)) # temp는 np array이다! pop이 없음
            print(now, " ", epoch, " 번째 epoch -> loss : ", round(temp_train, 2), end=' ')
            print("train_accuracy : ", round(self.train_accuracy/len(data.images) * 100, 2), "% ")

            # test and accuracy initialization
            self.test()
            self.train_accuracy = 0
            self.test_accuracy = 0

            # confusion matrix 
            if epoch >= 0:
                confusion = confusion_matrix(self.y_true, self.y_predic, normalize = "pred")
                plt.figure(figsize=(16,16))
                sns.heatmap(confusion, annot=True, cmap = 'Blues')
                plt.title("Normalized CONFUSION MATRIX : Convolution_Neural_Network_3_layer")
                plt.ylabel("True label")
                plt.xlabel("Predicted label")
                plt.show()

            # show top 3 accuracy
            if epoch >= 0:
                for i in range(10):
                    self.top_3.append(sorted(self.map_list[i].items(), reverse = True))

                for i in range(10):
                    print("picture:", i , "Top 3", end = " ")

                    for j in range(1, 4):
                        plt.subplot(1, 3, j)
                        percentage, image = self.top_3[i][j]
                        plt.imshow(image, cmap='Greys_r')
                        plt.axis('off')
                        print(round(percentage*100, 2), "%", end = " ")
                    plt.show()

            # train & test loss graph
            if epoch >= 0:
                print("train_cost_epoch : ", self.train_cost_epoch)
                print("test_cost_epoch : ", self.test_cost_epoch)
                plt.plot(range(0, epoch+1), self.train_cost_epoch, 'b', label='train')
                plt.plot(range(0, epoch+1), self.test_cost_epoch, 'r', label='test')
                plt.ylabel('Cost')
                plt.xlabel('Epochs')
                plt.legend(loc='upper right')
                plt.show()

    
    def feed_forward(self, input_images, input_labels, is_train) -> float:

                                #########################
                                ####### 1st layer #######
                                #########################
        # 1st convolution layer                                
        output_convolution_layer_1 = self.convolution_layer_1.forward(input_images) # (8, 6, 24, 24)
        
        # 1st relu
        self.output_relu_1 = self.ReLU(output_convolution_layer_1.copy())
        self.d_output_relu_1 = self.derivation_ReLU(self.output_relu_1.copy())

        # 1st max pooling
        output_max_pooling_1 = self.max_pooling(self.output_relu_1.copy(), 0)
        #print("output_max_pooling_1", output_max_pooling_1[0])

        #self.d_output_max_pooling_1 = self.derivation_max_pooling(output_max_pooling_1.copy(), self.output_relu_1.copy())
        self.d_output_max_pooling_1 = self.derivation_max_pooling(output_max_pooling_1.copy(), 0)

                                #########################
                                ####### 2nd layer #######
                                #########################

        # 2nd convolution layer
        output_convolution_layer_2 = self.convolution_layer_2.forward(output_max_pooling_1)
        
        # 2nd relu
        self.output_relu_2 = self.ReLU(output_convolution_layer_2.copy())
        self.d_output_relu_2 = self.derivation_ReLU(self.output_relu_2.copy())
        
        # 2nd max pooling
        output_max_pooling_2 = self.max_pooling(self.output_relu_2.copy(), 1) 
        self.d_output_max_pooling_2 = self.derivation_max_pooling(output_max_pooling_2.copy(), 1)
                                
                                #########################
                                ####### 3rd layer #######
                                #########################

        # 3rd linear layer
        batch, depth, height, weight = output_max_pooling_2.shape
        linear_temp = np.reshape(output_max_pooling_2, (batch, depth, height * weight)) 
        self.linear_input = np.reshape(linear_temp, (batch, depth * height * weight))
        linear_output = np.dot(self.linear_input, self.W3) + self.b3 
        
        # 3rd relu
        output_relu_3 = self.ReLU(linear_output)
        d_output_relu_3 = self.derivation_ReLU(output_relu_3.copy())

        # softmax
        self.soft = self.Softmax(output_relu_3.copy())
        loss = self.Cross_Entropy_Loss(self.soft, input_labels)
        self.d_softmax_cross = self.derivation_softmax_cross(self.soft, input_labels) # d_softmax_cross = 8*10

        if is_train == True:
            # calculate accuracy
            self.train_cost.append(loss/self.batch_size)
            for i in range(self.batch_size):
                prediction = np.argmax(self.soft[i])
                label_answer = np.argmax(input_labels[i])
                #self.y_predic.append(prediction)
                #self.y_true.append(label_answer)
                if prediction == label_answer:
                    self.train_accuracy += 1

        elif is_train == False:
            # calculate accuracy & top 3 accuracy
            self.test_cost.append(loss/self.batch_size)
            for i in range(self.batch_size):
                prediction = np.argmax(self.soft[i]) # prediction은 index 이다!!
                label_answer = np.argmax(input_labels[i])
                self.y_predic.append(prediction)
                self.y_true.append(label_answer)
                
                if prediction == label_answer:
                    self.test_accuracy += 1    
                    for _ in range(10):
                        temp_map = {}
                        self.map_list.append(temp_map) # map_list = [] 안에는 dictionary로 구성되어 있다
                    temp_image = np.reshape(input_images[i]*255, (28, 28))
                    self.map_list[label_answer][self.soft[i][prediction]] = temp_image

        return loss

    def back_propagation(self):

                                ##########################
                                #### back propagation ####
                                ##########################

        temp = np.dot(self.d_softmax_cross, self.W3.T) # 32, 16*depth = 48
        _, size = temp.shape
        size_2 = int((size / self.depth_1) ** (1/2)) # size_2 = 4
        temp_2 = np.reshape(temp, (self.batch_size, self.depth_1, size_2, size_2)) # 32,6,4,4 생성
        temp_3 = self.derivation_max_pooling(temp_2, 1)
        temp_5 = np.multiply(temp_3, self.d_output_max_pooling_2.copy()) 
        temp_6 = np.multiply(temp_5, self.d_output_relu_2) # temp_6 -100 ~ 100 사이값
        grad = self.convolution_layer_2.backward(temp_6) # grad, temp_7이 좀 커서 둘이 곱하면 커진다.
        temp_7 = self.derivation_max_pooling(grad, 0)
        temp_9 = np.multiply(temp_7, self.d_output_max_pooling_1)
        temp_10 = np.multiply(temp_9, self.d_output_relu_1) # temp_10 도 1000대의 값이 있음
        grad2 = self.convolution_layer_1.backward(temp_10)
        dW3 = np.dot(self.d_softmax_cross.T, self.linear_input)
        db3 = self.d_softmax_cross 

        self.vx_3 = self.rho * self.vx_3 - self.learning_rate * dW3.T
        self.W3 += self.vx_3
        self.vx_4 = self.rho * self.vx_4 - self.learning_rate * db3
        self.b3 += self.vx_4

    def test(self):
        data = dataloader.Dataloader(".", is_train = False, shuffle = True, batch_size=self.batch_size) 
        for i in range(0, int(len(data.images) / self.batch_size)):
            input_images, input_labels = data.__getitem__(i)
            self.feed_forward(input_images, input_labels, is_train = False)
        
        temp_test = sum(self.test_cost) / (len(data.images) / self.batch_size)
        self.test_cost_epoch.append(round(temp_test, 2))
        self.test_cost = []
        now = datetime.datetime.now()
        print(now, ".. 번째 epoch -> test  loss : ", round(temp_test, 2), end=' ')
        print("test accuracy : ", round(self.test_accuracy/len(data.images) * 100, 2), "%")

    def max_pooling(self, x, flag): # max pooling using im2col -> faster
            batch, channel, height, weight = x.shape
            stride = self.max_pooling_stride

            out_h=int(1+(height-self.max_pool_h)/self.max_pool_stride)
            out_w=int(1+(weight-self.max_pool_w)/self.max_pool_stride)
        
            col=super().im2col(x, self.max_pool_h, self.max_pool_w, self.max_pool_stride, self.pad)  #전개
            col=col.reshape(-1, self.max_pool_h*self.max_pool_w)
            arg_max = np.argmax(col, axis=1)
            out=np.max(col, axis=1) 
            
            out=out.reshape(batch, out_h, out_w, channel).transpose(0, 3, 1, 2)
            if flag == 0:
                self.x_1 = x
                self.arg_max_1 = arg_max
            elif flag == 1:
                self.x_2 = x
                self.arg_max_2 = arg_max
            return out

    def derivation_max_pooling(self, dout, flag): # derivation max pooling using im2col -> faster

        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.max_pool_h * self.max_pool_w
        dmax = np.zeros((dout.size, pool_size))

        if flag == 0:
            dmax[np.arange(self.arg_max_1.size), self.arg_max_1.flatten()] = dout.flatten()
            dmax = dmax.reshape(dout.shape + (pool_size,)) 
            dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
            dx = super().col2im(dcol, self.x_1.shape, self.max_pool_h, self.max_pool_w, self.max_pool_stride, self.pad)

        elif flag == 1:
            dmax[np.arange(self.arg_max_2.size), self.arg_max_2.flatten()] = dout.flatten()
            dmax = dmax.reshape(dout.shape + (pool_size,)) 
            dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
            dx = super().col2im(dcol, self.x_2.shape, self.max_pool_h, self.max_pool_w, self.max_pool_stride, self.pad)
            
        return dx

    def max_pooling_2(self, input): # max pooling using for loop -> slower..
        
        (batch_size, channel, height_prev, weight_prev) = input.shape
        
        stride = self.max_pooling_stride
        filter_size = self.max_pooling_kernel_size

        height = int(1 + (height_prev - filter_size) / stride)
        weight = int(1 + (weight_prev - filter_size) / stride)             
        output = np.zeros((batch_size, channel, height, weight)) 

        for batch in range(batch_size):
            for i in range(channel):                         
                for j in range(height):                    
                    for k in range(weight):                     
                        vert_start = j * stride
                        vert_end = j * stride +filter_size
                        horiz_start = k * stride      
                        horiz_end = k * stride + filter_size
                        a_prev_slice = input[batch, i, vert_start:vert_end, horiz_start:horiz_end] 
                        output[batch, i, j, k] = np.max(a_prev_slice) 
                
        return output

    def derivation_max_pooling_2(self, dA, output_relu): # derivation max pooling using for loop -> slower..
        
        A_prev = output_relu # (8, 5, 24, 24)
        stride = self.max_pooling_stride
        f = self.max_pooling_kernel_size
       
        batch_size, channel, height_prev, weight_prev = A_prev.shape
        batch_size, channel, height, weight = dA.shape
        
        dA_prev = np.zeros(A_prev.shape) # (8, 5, 24, 24)
        
        for batch in range(batch_size):
            for i in range(channel):                      
                a_prev = A_prev[batch, i]
                for j in range(height):                   
                    for k in range(weight):               
                        vert_start = j
                        vert_end = vert_start + f
                        horiz_start = k
                        horiz_end = horiz_start + f
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end]
                        mask = self.max_pooling_mask(a_prev_slice)
                        dA_prev[batch, i, vert_start:vert_end, horiz_start:horiz_end] += np.multiply(mask, dA[batch, i, j, k])
                            
        return dA_prev
        
    def max_pooling_mask(self, x):     
        mask = x == np.max(x)       
        return mask

    def linear_layer(self, row, column):
        np.random.seed(0)
        linear_layer = np.random.randn(row, column)
        return linear_layer

    def ReLU(self, x):
        return np.maximum(0, x)

    def derivation_ReLU(self, x):
        dRelu_dx = x
        dRelu_dx[dRelu_dx <= 0] = 0
        dRelu_dx[dRelu_dx > 0] = 1
        return dRelu_dx

    def Leaky_ReLU(self, x):
        return np.maximum(0.01*x, x)

    def derivation_Leaky_ReLU(self, x):
        dRelu_dx = x
        dRelu_dx[dRelu_dx < 0] = 0.01
        dRelu_dx[dRelu_dx > 0] = 1
        return dRelu_dx

    def Softmax(self, x): # x : 8*10
        s = np.exp(x)
        total = np.sum(s, axis=1).reshape(-1,1)
        return s/total

    def Cross_Entropy_Loss(self, softmax_matrix, label_matrix):
        # delta -> 아주 작은 값 (y가 0인 경우 -inf 값을 예방) 
        delta = 1e-7 
        return -np.sum(label_matrix*np.log(softmax_matrix+delta))

    def derivation_softmax_cross(self, softmax_matrix, label_matrix):
        return softmax_matrix - label_matrix

if __name__ == "__main__":
    cnn = CNN()
    cnn.train()
