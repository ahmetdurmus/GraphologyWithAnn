# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 11:26:05 2019

@author: Ahmet
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from colorutils import Color
import cv2
from flask import Flask, render_template
from flask import request
import math

app = Flask(__name__)

app.static_folder = 'static'

def Sigmoid(Z):
    return 1/(1+np.exp(-Z))

def Relu(Z):
    return np.maximum(0,Z)

def dRelu(Z):
    Z[Z<=0] = 0
    Z[Z>0] = 1
    return Z

def dSigmoid(Z):
    s = 1/(1+np.exp(-Z))
    dZ = s * (1-s)
    return dZ

class Analysis:
    def __init__(self, x):
        self.X = x
        self.param = {}
        self.param['W1'] = np.array(([]), dtype=float)
        self.param['b1'] = np.array(([]), dtype=float)
        self.param['W2'] = np.array(([]), dtype=float)
        self.param['b2'] = np.array(([]), dtype=float)
        self.param['W3'] = np.array(([]), dtype=float)
        self.param['b3'] = np.array(([]), dtype=float)
        
    def readWeight(self):
        Array = []
        f=open('W1.txt', 'r').readlines()
        N = len(f)
        for i in range(0,N):
            Array.append(f[i].split())
            
        for i in range(0,N):
            for j in range(0,1024):
                Array[i][j] = float(Array[i][j])
                
        self.param['W1'] = np.array(Array)
        
        Array = []
        
        f=open('W2.txt', 'r').readlines()
        N = len(f)
        for i in range(0,N):
            Array.append(f[i].split())
            
        for i in range(0,N):
            for j in range(0,50):
                Array[i][j] = float(Array[i][j])
                
        self.param['W2'] = np.array(Array)
        
        Array = []
        
        f=open('W3.txt', 'r').readlines()
        N = len(f)
        for i in range(0,N):
            Array.append(f[i].split())
            
        for i in range(0,N):
            for j in range(0,50):
                Array[i][j] = float(Array[i][j])
                
        self.param['W3'] = np.array(Array)
        
        Array = []
        
        f=open('b1.txt', 'r').readlines()
        N = len(f)
        for i in range(0,N):
            Array.append(f[i].split())
            
        for i in range(0,N):
            for j in range(0,1):
                Array[i][j] = float(Array[i][j])
                
        self.param['b1'] = np.array(Array)
        
        Array = []
        
        f=open('b2.txt', 'r').readlines() 
        N = len(f)
        for i in range(0,N):
            Array.append(f[i].split())
            
        for i in range(0,N):
            for j in range(0,1):
                Array[i][j] = float(Array[i][j])
                
        self.param['b2'] = np.array(Array)
        
        Array = []
        
        f=open('b3.txt', 'r').readlines()
        N = len(f)
        for i in range(0,N):
            Array.append(f[i].split())
            
        for i in range(0,N):
            for j in range(0,1):
                Array[i][j] = float(Array[i][j])
                
        self.param['b3'] = np.array(Array)
        
        Array = []     
    def forward(self):
        
        Z1 = self.param['W1'].dot(self.X.T) + self.param['b1']
        A1 = Relu(Z1)
        
        Z2 = self.param['W2'].dot(A1) + self.param['b2'] 
        A2 = Relu(Z2)
        
        Z3 = self.param['W3'].dot(A2) + self.param['b3']
        A3 = Sigmoid(Z3)
        
        self.Yh=A3
        
        return self.Yh
class ann:
    def __init__(self, x, y, L, enter_number, first_intermediate_layer_number, second_intermediate_layer_number, output_layer_number):
        self.debug = 0;
        self.X=x
        self.Y=y
        self.Yh=np.zeros((1,self.Y.shape[1])) 
        self.L=L + 1
        if self.L == 3:
            self.dims = [enter_number, first_intermediate_layer_number, second_intermediate_layer_number, output_layer_number]
        else:
            self.dims = [enter_number, first_intermediate_layer_number, output_layer_number]
        self.param = {}
        self.ch = {}
        self.grad = {}
        self.loss = []
        self.lr=0.01
        self.sam = self.Y.shape[1]
        self.threshold=0.5
        
    def nInit(self):    
        np.random.seed(1)
        self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0]) 
        self.param['b1'] = np.zeros((self.dims[1], 1))        
        self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1]) 
        self.param['b2'] = np.zeros((self.dims[2], 1))
        if self.L == 3:
            self.param['W3'] = np.random.randn(self.dims[3], self.dims[2]) / np.sqrt(self.dims[2]) 
            self.param['b3'] = np.zeros((self.dims[3], 1))                  
        return

    def forward(self):    
        Z1 = self.param['W1'].dot(self.X) + self.param['b1'] 
        A1 = Relu(Z1)
        #print("A1",A1)
        self.ch['Z1'],self.ch['A1']=Z1,A1
        
        if self.L == 3:
            Z2 = self.param['W2'].dot(A1) + self.param['b2'] 
            A2 = Relu(Z2)
            #print("A2",A2)
            self.ch['Z2'],self.ch['A2']=Z2,A2
            
            Z3 = self.param['W3'].dot(A2) + self.param['b3']  
            A3 = Sigmoid(Z3)
            self.ch['Z3'],self.ch['A3']=Z3,A3
            
            self.Yh=A3
            loss=self.nloss(A3)
            #print(loss)

        elif self.L == 2:
            Z2 = self.param['W2'].dot(A1) + self.param['b2']  
            A2 = Sigmoid(Z2)
            self.ch['Z2'],self.ch['A2']=Z2,A2
            
            self.Yh=A2
            loss=self.nloss(A2)
        
        return self.Yh, loss

    def nloss(self,Yh):
        squared_errors = (self.Yh - self.Y) ** 2
        self.Loss= np.sum(squared_errors)
        return self.Loss

    def backward(self):
        dLoss_Yh = - (np.divide(self.Y, self.Yh ) - np.divide(1 - self.Y, 1 - self.Yh))    
        
        if self.L == 3:
            dLoss_Z2 = dLoss_Yh * dSigmoid(self.ch['Z3'])    
            dLoss_A2 = np.dot(self.param["W3"].T,dLoss_Z2)
            dLoss_W3 = 1./self.ch['A2'].shape[1] * np.dot(dLoss_Z2,self.ch['A2'].T)
            dLoss_b3 = 1./self.ch['A2'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1],1])) 
            
            dLoss_Z1 = dLoss_A2 * dRelu(self.ch['Z2'])
            dLoss_A1 = np.dot(self.param["W2"].T,dLoss_Z1)
            dLoss_W2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z1,self.ch['A1'].T)
            dLoss_b2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1],1]))
                        
            dLoss_Z0 = dLoss_A1 * dRelu(self.ch['Z1'])
            dLoss_A0 = np.dot(self.param["W1"].T,dLoss_Z0)
            dLoss_W1 = 1./self.X.shape[1] * np.dot(dLoss_Z0,self.X.T)
            dLoss_b1 = 1./self.X.shape[1] * np.dot(dLoss_Z0, np.ones([dLoss_Z0.shape[1],1]))  
            
            self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
            self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
            self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
            self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2
            self.param["W3"] = self.param["W3"] - self.lr * dLoss_W3
            self.param["b3"] = self.param["b3"] - self.lr * dLoss_b3
        
        elif self.L == 2: 
            
            dLoss_Z2 = dLoss_Yh * dSigmoid(self.ch['Z2'])    
            dLoss_A1 = np.dot(self.param["W2"].T,dLoss_Z2)
            dLoss_W2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2,self.ch['A1'].T)
            dLoss_b2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1],1])) 
                                
            dLoss_Z1 = dLoss_A1 * dRelu(self.ch['Z1'])        
            dLoss_A0 = np.dot(self.param["W1"].T,dLoss_Z1)
            dLoss_W1 = 1./self.X.shape[1] * np.dot(dLoss_Z1,self.X.T)
            dLoss_b1 = 1./self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1],1]))  
            
            self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
            self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
            self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
            self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2
        
        return
    
    def gd(self,X, Y, max_error_limit, iter = 3000):
        
        np.random.seed(1)            
    
        self.nInit()
    
        for i in range(0, iter):
            Yh, loss=self.forward()
            self.backward()
        
            #if i % 500 == 0:
            print ("Cost after iteration %i: %f" %(i, loss))
            self.loss.append(loss)
                
            if loss <= max_error_limit:
                break

        plt.plot(np.squeeze(self.loss))
        plt.ylabel('Loss')
        plt.xlabel('Iter')
        plt.title("Lr =" + str(self.lr))
        plt.show()
        
        print(Yh)
        
        return i, self.loss
    
    def saveWeights(self):
        np.savetxt("W1.txt", self.param["W1"], fmt="%s")
        np.savetxt("b1.txt", self.param["b1"], fmt="%s")
        np.savetxt("W2.txt", self.param["W2"], fmt="%s")
        np.savetxt("b2.txt", self.param["b2"], fmt="%s")
        if self.L == 3:
            np.savetxt("W3.txt", self.param["W3"], fmt="%s")
            np.savetxt("b3.txt", self.param["b3"], fmt="%s")
        

src_path = "C:/Users/Ahmet/Desktop/Bitirme Tezi/YSA/ArtificialNeuralNetwork/"
colorX = []

def get_string(img_path):
    
    # Read image with opencv
    img = cv2.imread(img_path)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Write image after removed noise
    cv2.imwrite(src_path + "removed_noise.png", img)

    #  Apply threshold to get image with only black and white
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # Write the image after apply opencv to do some ...
    cv2.imwrite(src_path + "thres.png", img)

    im = Image.open(src_path + "thres.png")
    
    width, height = im.size
    
    rgb_im = im.convert('RGB')
    
    for pixel_x in range(0, width):
        for pixel_y in range(0, height):
            r, g, b = rgb_im.getpixel((pixel_x, pixel_y))
            hexColor = Color((r, g, b))
            colorX.append(hexColor.red) 

@app.route('/', methods=['POST','GET'])
def homepage():

    # when the GET request is thrown
    if request.method == 'GET':
        
        return render_template("index.html")

    # when the POST request is thrown
    if request.method == 'POST':
        """   
        learning_coefficient      = request.form['learning_coefficient']
        hidden_coefficient_number = request.form['hidden_coefficient_number']
        max_error_limit           = request.form['max_error_limit']
        enter_number              = request.form['enter_number']
        first_intermediate_layer_number = request.form['first_intermediate_layer_number']
        second_intermediate_layer_number = request.form['second_intermediate_layer_number']
        output_layer_number       = request.form['output_layer_number']
        
        
        
        df = pd.read_csv('training.csv',header=None)
        df2 = pd.read_csv('output.csv',header=None)
        
        scaled_df=df
        names = df.columns[:]
        scaler = MinMaxScaler()
        scaled_df2=df2
        names2 = df2.columns[:]
        scaled_df = scaler.fit_transform(df.iloc[:,:])
        scaled_df = pd.DataFrame(scaled_df, columns=names)
        scaled_df2 = pd.DataFrame(scaled_df2, columns=names2)
        
        x=scaled_df.iloc[:,:].values.transpose()
        y=scaled_df2.iloc[:,:].values.transpose()
        
        if second_intermediate_layer_number != "":
            nn = ann(x,y, int(hidden_coefficient_number), int(enter_number), int(first_intermediate_layer_number), int(second_intermediate_layer_number), int(output_layer_number))
        
        else: 
            nn = ann(x,y, int(hidden_coefficient_number), int(enter_number), int(first_intermediate_layer_number), 0, int(output_layer_number))
        
        nn.lr = float(learning_coefficient)
        
        if nn.L == 3:
            nn.dims = [int(enter_number), int(first_intermediate_layer_number), int(second_intermediate_layer_number), int(output_layer_number)]
        elif nn.L == 2:
            nn.dims = [int(enter_number), int(first_intermediate_layer_number), int(output_layer_number)]
        
        max_error_limit = float(max_error_limit)
        
        i, lossList = nn.gd(x, y, max_error_limit = max_error_limit, iter = 60000)
        
        return render_template("index.html", 
                           iteration = i, 
                           loss_limit = math.ceil(lossList[0]), 
                           lossList = lossList,
                           learning_coefficient = learning_coefficient,
                           hidden_coefficient_number = hidden_coefficient_number,
                           max_error_limit = max_error_limit, 
                           enter_number = enter_number, 
                           first_intermediate_layer_number = first_intermediate_layer_number,
                           second_intermediate_layer_number = second_intermediate_layer_number
                           )

        """
        
        image_path = request.form['image_path']
        learning_coefficient      = 0.0
        hidden_coefficient_number = 0
        max_error_limit           = 0.0
        enter_number              = 0
        first_intermediate_layer_number = 0
        second_intermediate_layer_number = 0
        output_layer_number       = 0
        
        get_string("Test/" + image_path)
        rgbValue = np.array(([colorX]))
        rgbValueNormalize = []
        for i in range(len(rgbValue)):
            rgbValueNormalize.append(rgbValue[i] / 255)
        
        rgbValueNormalizeNp = np.array(rgbValueNormalize)
        
        analysis = Analysis(rgbValueNormalizeNp)
        analysis.readWeight()
        result = analysis.forward()
        print(result.tolist())
        
        a = []
        state = [["İletişimi Düşük", "bg-danger", "İletişim kurmak bir yetenek, iş dünyasında başarı için olmazsa olmaz bir kural olarak görülür."],
                 ["İletişimi Yüksek", "bg-success", "Kişi daha arkadaş canlısı, yönlendirici, sorumluluk sahibi, girişken olma eğilimi taşıyor."],
                 ["Sabırsız", "bg-secondary", "“Ertelemek, işyerindeki verimliliği ciddi ölçüde etkiliyor ve sabırsız kişiler evrak işlerini sürekli ertelediği için insanlar büyük paralar kaybediyor” Ernesto Reuben"],
                 ["Enerjik", "bg-warning", "Başarılı olmanın sırrının, sürekli daha iyiye ulaşmak için sınırları zorlamak olduğunu bilirler."],
                 ["Hırsı düşük", "bg-info", "Hırs; amaç ve hedefler gerçekçi sınırlar içinde olduğu, kişiyi üretmeye, çeşitlendirmeye, yeniliğe taşıdığı sürece olağan yaşamın bir parçası olarak sayılabilir. Ancak bu sınırlardan taştığı noktada stres kaynağına dönüşür."], 
                 ["Hırsı yüksek", "bg-primary", "Hırs; amaç ve hedefler gerçekçi sınırlar içinde olduğu, kişiyi üretmeye, çeşitlendirmeye, yeniliğe taşıdığı sürece olağan yaşamın bir parçası olarak sayılabilir. Ancak bu sınırlardan taştığı noktada stres kaynağına dönüşür."]]
            
        for i in range(6):
            
            if result[i][0] > 0.7:
                a.append(state[i])

        print(a)
            
        lossList = []
        
        return render_template("index.html", 
                               iteration = 0, 
                               loss_limit = 0, 
                               lossList = lossList,
                               learning_coefficient = learning_coefficient,
                               hidden_coefficient_number = hidden_coefficient_number,
                               max_error_limit = max_error_limit, 
                               enter_number = enter_number, 
                               first_intermediate_layer_number = first_intermediate_layer_number,
                               second_intermediate_layer_number = second_intermediate_layer_number,
                               character_analysis = a
                               )
        
            
if __name__ == '__main__':
    app.run()
