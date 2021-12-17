import csv
import os
import matplotlib.pyplot
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

def read_tsv(filename):
    with open(filename,encoding="utf-8") as f:
        tsvreader = csv.reader(f, delimiter='\t')
        temp = list(tsvreader)
        return temp

#glove向量读取
def get_numpy_word_embed(word2ix,lenn=1e6):
    row = 0
    file = 'glove.6B.50d.txt'
    path = 'D:\\transformerFileDownload\\glove'
    whole = os.path.join(path, file)
    words_embed = {}
    with open(whole, mode='r',encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split()
            word = line_list[0]
            embed = line_list[1:]
            embed = [float(num) for num in embed]
            words_embed[word] = embed
            row += 1
    # word2ix = {}
    ix2word = {ix: w for w, ix in word2ix.items()}

    id2emb = {}
    cls_sep = np.random.normal(scale=0.5,size=100).tolist()
    for ix in range(len(word2ix)):
        if ix2word[ix] in words_embed:
            id2emb[ix] = words_embed[ix2word[ix]]
        else:
            id2emb[ix] = [0.0] * 50
    id2emb[lenn+2] = cls_sep[:50]
    id2emb[lenn+3] = cls_sep[50:]
    data = [id2emb[ix] for ix in range(len(word2ix))]

    return data

def vocab_mask_generation(data):
    m,n = data.shape
    vocab = np.full((m,n),0)
    mask = np.full((m,n),0)
    for i in range(m):
        for j in range(n):
            vocab[i, j] = j
            if data[i,j]>0:
                mask[i,j] = 1
    return vocab,mask

#rnn取最后的隐藏状态效果很差
def rnn_mask_g(data):
    rnn_mask = torch.zeros((len(data),1,64)).long()
    for i,j in enumerate(data):
        rnn_mask[i][0] += j-1
    return rnn_mask

#使用向量，输出词的id
def id_generation(data,vocab,vect_len):
    sentence = []
    for i in data:
        sentence_d = []
        for j in i:
            if j in vocab.keys():
                sentence_d.append(vocab[j])
            else:
                sentence_d.append(vect_len)
        sentence.append(sentence_d)
    return sentence

#解决多标签不平衡
class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss

class CNN_e(nn.Module):
    def __init__(self,vect_len,out_channels,flag_g,weight,label_num):
        super(CNN_e, self).__init__()
        #随机初始化向量与glove向量
        if flag_g == 0:
            weight_r = nn.init.kaiming_normal_(torch.Tensor(vect_len+2,50))
            self.embedding = nn.Embedding(vect_len + 2, 50,_weight=weight_r)
        else:
            self.embedding = nn.Embedding(vect_len+2,50,_weight=weight)
        self.drop = nn.Dropout(0.5)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=50,
                                              out_channels=out_channels,
                                              kernel_size=filter_size)
                                    for filter_size in [3, 4, 5]])

        self.l = nn.Sequential(nn.Linear(out_channels*3,label_num),nn.Dropout(0.5))
        self.cross = nn.CrossEntropyLoss()

    def forward(self,data,mask):
        embed = self.embedding(data)
        embed = self.drop(embed)
        embedded = embed.permute(0, 2, 1)
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[-1]).squeeze(-1) for conv in conved]
        out = torch.cat(pooled, dim=-1)
        out_l = self.l(out)
        return out_l

    def loss(self, pre, label_data):
        loss = self.cross(pre, label_data)
        return loss

    def acc(self,y_pre,label):
        lenn = len(y_pre)
        y_pre = y_pre.argmax(axis=1)
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn
        return y_acc

class RNN_e(nn.Module):
    def __init__(self,vect_len,flag_g,weight,label_num):
        super(RNN_e, self).__init__()
        if flag_g == 0:
            weight_r = nn.init.xavier_normal_(torch.Tensor(vect_len + 2, 50))
            self.embedding = nn.Embedding(vect_len + 2, 50, _weight=weight_r)
        else:
            self.embedding = nn.Embedding(vect_len + 2, 50, _weight=weight)
        self.drop = nn.Dropout(0.5)
        self.rnn = nn.RNN(input_size=50,hidden_size=64,num_layers=1,dropout=0.5,batch_first=True,nonlinearity='tanh')
        self.l = nn.Linear(64,label_num)
        self.loss = nn.CrossEntropyLoss()

    def forward(self,data,mask):

        embed = self.embedding(data)
        embed = self.drop(embed)
        h0 = torch.zeros((1,embed.size()[0],64),requires_grad=False).cuda()
        rec,rnn = self.rnn(embed,h0)
        gather_em = rec.gather(1,mask)
        out = self.l(gather_em).squeeze(1)
        return out

    def loss(self, pre, label_data):
        loss = self.loss(pre, label_data)

        return loss

    def acc(self,y_pre,label):
        lenn = len(y_pre)
        y_pre = y_pre.argmax(axis=1)
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn
        return y_acc

def cnn_embedding_train(X_train_c,y_train,X_test_c,y_test,X_dev_c,y_dev,vect_len,epochs,flag_g,weigth):
    train_data = TensorDataset(X_train_c, y_train)
    dev_data = TensorDataset(X_dev_c,y_dev)
    test_data = TensorDataset(X_test_c, y_test)

    batch = 512
    train_iter = DataLoader(train_data, shuffle=True, batch_size=batch)
    dev_iter = DataLoader(dev_data, shuffle=True, batch_size=batch)
    test_iter = DataLoader(test_data, shuffle=True, batch_size=batch)

    model = CNN_e(vect_len,64,flag_g,weigth,5)
    model.to(torch.device('cuda:0'))

    #Adam来做随机梯度下降
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=0.0001)
    model.train()

    weight_loss = None
    loss_s = FocalLoss(weight=weight_loss)
    max_dev,max_test = 0.0,0.0
    for epoch in range(epochs):
        for batch in train_iter:
            train_id, y_train_t = batch
            train_id, y_train_t = train_id.cuda(), y_train_t.cuda()

            out = model(train_id)
            loss = loss_s(out,y_train_t)
            loss.backward()
            optimizer.step()
            model.zero_grad()
        if epoch%4 == 0:
            model.eval()
            flag = 0
            total_train,total_label = None,None
            for batch in dev_iter:
                train_id, y_train_t = batch
                train_id, y_train_t = train_id.cuda(), y_train_t.cuda()

                out = model(train_id)
                if flag == 0:
                    total_train = out
                    total_label = y_train_t
                    flag = 1
                else:
                    total_train = torch.cat((total_train, out), 0)
                    total_label = torch.cat((total_label,y_train_t),0)
            dev_acc = model.acc(total_train, total_label)
            if max_dev<dev_acc:
                max_dev = dev_acc
                for batch in test_iter:
                    train_id, y_train_t = batch
                    train_id, y_train_t = train_id.cuda(), y_train_t.cuda()

                    out = model(train_id)
                    if flag == 0:
                        total_train = out
                        total_label = y_train_t
                        flag = 1
                    else:
                        total_train = torch.cat((total_train, out), 0)
                        total_label = torch.cat((total_label, y_train_t), 0)
                test_acc = model.acc(total_train, total_label)
                max_test = test_acc
            print("epoch:", epoch, "----dev_acc:", max_dev,"----test_acc:", max_test)
            model.train()
    return max_dev,max_test

def rnn_embedding_train(X_train_c,y_train,X_test_c,y_test,X_dev_c,y_dev,mask_train,mask_test,mask_dev,vect_len,epochs,flag_g,weigth):
    train_data = TensorDataset(X_train_c, y_train,mask_train)
    dev_data = TensorDataset(X_dev_c,y_dev,mask_dev)
    test_data = TensorDataset(X_test_c, y_test,mask_test)

    batch = 512
    train_iter = DataLoader(train_data, shuffle=True, batch_size=batch)
    test_iter = DataLoader(test_data, shuffle=True, batch_size=batch)
    dev_iter = DataLoader(dev_data,shuffle=True,batch_size=batch)

    model = RNN_e(vect_len,flag_g,weigth,5)
    model.to(torch.device('cuda:0'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)
    model.train()
    loss_s = FocalLoss()
    max_dev,max_test = 0.0,0.0
    for epoch in range(epochs):
        for batch in train_iter:
            train_id, y_train_t,mask_train_t = batch
            train_id, y_train_t,mask_train_t = train_id.cuda(), y_train_t.cuda(),mask_train_t.cuda()

            out = model(train_id,mask_train_t)
            loss = loss_s(out,y_train_t)
            loss.backward()
            optimizer.step()
            model.zero_grad()
        if epoch%4 == 0:
            model.eval()
            flag = 0
            total_train,total_label_train = None,None
            for batch in dev_iter:
                train_id, y_train_t, mask_train_t = batch
                train_id, y_train_t, mask_train_t = train_id.cuda(), y_train_t.cuda(), mask_train_t.cuda()

                out = model(train_id, mask_train_t)
                if flag == 0:
                    total_train = out
                    total_label_train = y_train_t
                    flag = 1
                else:
                    total_train = torch.cat((total_train, out), 0)
                    total_label_train = torch.cat((total_label_train,y_train_t),0)
            dev_acc = model.acc(total_train, total_label_train)
            if max_dev<dev_acc:
                max_dev = dev_acc
                flag = 0
                total_test, total_label_test = None, None
                for batch in test_iter:
                    train_id, y_train_t,mask_test_t = batch
                    train_id, y_train_t,mask_test_t = train_id.cuda(), y_train_t.cuda(),mask_test_t.cuda()
                    out = model(train_id,mask_test_t)
                    if flag == 0:
                        total_test = out
                        total_label_test = y_train_t
                        flag = 1
                    else:
                        total_test = torch.cat((total_test, out), 0)
                        total_label_test= torch.cat((total_label_test, y_train_t), 0)

                test_acc = model.acc(total_test, total_label_test)
                max_test = test_acc
            print("epoch:", epoch, "----dev_acc:", max_dev, "----test_acc:", max_test)
            model.train()
    return max_dev,max_test

if __name__ == '__main__':
    print(torch.cuda.is_available())
    # 全部
    data_train = read_tsv('./data/sentiment-analysis-on-movie-reviews/convert_2_train.tsv')
    data_test = read_tsv('./data/sentiment-analysis-on-movie-reviews/convert_2_test.tsv')
    data_dev = read_tsv('./data/sentiment-analysis-on-movie-reviews/convert_2_dev.tsv')
    #重复，懒得写函数
    X_train,X_split_train,y_train,X_test,X_split_test,y_test,X_dev,X_split_dev,y_dev = [],[],[],[],[],[],[],[],[]
    for i in data_train:
        X_train.append(i[2].lower())
        y_train.append(int(i[3]))
        X_split_train.append(i[2].lower().split(' '))
    for i in data_test:
        X_test.append(i[2].lower())
        y_test.append(int(i[3]))
        X_split_test.append(i[2].lower().split(' '))
    for i in data_dev:
        X_dev.append(i[2].lower())
        y_dev.append(int(i[3]))
        X_split_dev.append(i[2].lower().split(' '))
    X_len_train = [len(i) for i in X_split_train]
    X_len_test = [len(i) for i in X_split_test]
    X_len_dev = [len(i) for i in X_split_dev]
    max_len = 0
    for i in X_len_train:
        max_len = max(max_len,i)
    for i in X_len_dev:
        max_len = max(max_len,i)
    for i in X_len_test:
        max_len = max(max_len,i)

    veccc = {}
    vect_len = 0
    for i in X_split_train:
        for j,k in enumerate(i):
            if k not in veccc.keys():
                veccc[k] = vect_len
                vect_len += 1
    for i in X_split_test:
        for j, k in enumerate(i):
            if k not in veccc.keys():
                veccc[k] = vect_len
                vect_len += 1
    for i in X_split_dev:
        for j,k in enumerate(i):
            if k not in veccc.keys():
                veccc[k] = vect_len
                vect_len += 1

    veccc['[UNK]'] = vect_len
    veccc['[PAD]'] = vect_len+1
    glove_embed = get_numpy_word_embed(veccc)
    glove_embed = torch.FloatTensor(glove_embed)

    sentence_id_train = id_generation(X_split_train,veccc,vect_len)
    sentence_id_train = [torch.LongTensor(i) for i in sentence_id_train]
    sentence_id_train = pad_sequence(sentence_id_train,batch_first=True,padding_value=vect_len)
    sentence_id_test = id_generation(X_split_test, veccc, vect_len)
    sentence_id_test = [torch.LongTensor(i) for i in sentence_id_test]
    sentence_id_test = pad_sequence(sentence_id_test, batch_first=True, padding_value=vect_len)
    sentence_id_dev = id_generation(X_split_dev, veccc, vect_len)
    sentence_id_dev = [torch.LongTensor(i) for i in sentence_id_dev]
    sentence_id_dev = pad_sequence(sentence_id_dev, batch_first=True, padding_value=vect_len)
    rnn_mask_train = rnn_mask_g(X_len_train)
    rnn_mask_test = rnn_mask_g(X_len_test)
    rnn_mask_dev = rnn_mask_g(X_len_dev)
    y_train = torch.tensor(y_train).long()
    y_dev = torch.tensor(y_dev).long()
    y_test = torch.tensor(y_test).long()

    #随机向量与glove向量，cnn与rnn比较
    max_cnn_dev,max_cnn_test,max_rnn_dev,max_rnn_test = 0.0,0.0,0.0,0.0
    for i in range(2):
        cnn_dev_acc,cnn_test_acc = cnn_embedding_train(sentence_id_train, y_train, sentence_id_test, y_test,sentence_id_dev,y_dev, vect_len, 100, i, glove_embed)
        rnn_dev_acc, rnn_test_acc = rnn_embedding_train(sentence_id_train, y_train, sentence_id_test, y_test,sentence_id_dev,y_dev, rnn_mask_train, rnn_mask_test,rnn_mask_dev, vect_len, 100, i, glove_embed)
        if i==0:
            print('random')
        else:
            print('glove')
        print("cnn:",cnn_dev_acc,'\t',cnn_test_acc)
        print("rnn:",rnn_dev_acc,'\t',rnn_test_acc)

    # matplotlib.pyplot.subplot(2, 1, 1)
    # matplotlib.pyplot.semilogx(epochs, cnn_acc[0][0], 'r--', label='cnn+random')
    # matplotlib.pyplot.semilogx(epochs, cnn_acc[1][0], 'g--', label='cnn+glove')
    # matplotlib.pyplot.semilogx(epochs, rnn_acc[0][0], 'b--', label='rnn+random')
    # matplotlib.pyplot.semilogx(epochs, rnn_acc[1][0], 'black', label='rnn+glove')
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.title("Training Accuracy")
    # matplotlib.pyplot.xlabel("epochs")
    # matplotlib.pyplot.ylabel("Accuracy")
    # matplotlib.pyplot.ylim(0.4, 1.0)
    # matplotlib.pyplot.subplot(2, 1, 2)
    # matplotlib.pyplot.semilogx(epochs, cnn_acc[0][1], 'r--', label='cnn+random')
    # matplotlib.pyplot.semilogx(epochs, cnn_acc[1][1], 'g--', label='cnn+glove')
    # matplotlib.pyplot.semilogx(epochs, rnn_acc[0][1], 'b--', label='rnn+random')
    # matplotlib.pyplot.semilogx(epochs, rnn_acc[1][1], 'black', label='rnn+glove')
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.title("Test Accuracy")
    # matplotlib.pyplot.xlabel("epochs")
    # matplotlib.pyplot.ylabel("Accuracy")
    # matplotlib.pyplot.ylim(0.4, 1.0)
    # matplotlib.pyplot.tight_layout()
    # matplotlib.pyplot.show()