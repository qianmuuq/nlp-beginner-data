import json
import math
import re
import matplotlib.pyplot
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from ner_task4 import setup_seed
from sentiment_analysis_task2 import get_numpy_word_embed, id_generation, FocalLoss

#这里还是可以优化的，将文中非字母和数字的去除，有些特殊情况如 XX‘s不正确
def read_data(filename):
    with open(filename,encoding='utf-8') as f:
        sentence1,sentence2,label = [],[],[]
        read_nums = 0
        # ss = {}
        for i in f.readlines():
            d = json.loads(i)
            if len(d["sentence1"].lower().split(' ')) > 16 or len(d["sentence2"].lower().split(' ')) > 16:
                continue
            d1 = re.sub(r'[^a-zA-Z0-9 ]','',d["sentence1"].lower())
            d2 = re.sub(r'[^a-zA-Z0-9 ]', '', d["sentence2"].lower())
            sentence1.append(d1.split(' '))
            sentence2.append(d2.split(' '))
            label.append(d["gold_label"])
            read_nums += 1
    return sentence1,sentence2,label

#文本长度掩码
def len_mask_g(data,max_len):
    len_mask = torch.zeros((len(data),1,64)).long()
    softmax_mask = torch.zeros((len(data),max_len)).long()
    for i,j in enumerate(data):
        len_mask[i][0] += j-1
        softmax_mask[i][:j] = 1
    return len_mask,softmax_mask

#注意力机制掩码
def martix_mask(s1,s2,max_len):
    m_mask = torch.zeros((len(s1),max_len,max_len)).long()
    for i,j in enumerate(s1):
        m_mask[i,:s1[i],:s2[i]] = 1
    return m_mask

def token_type(s1_len,s2_len,maxlen):
    token_type_ids = torch.zeros(len(s1_len),maxlen).long()
    for i,_ in enumerate(s1_len):
        token_type_ids[i,2+s1_len[i]:3+s1_len[i]+s2_len[i]] = 1
    return token_type_ids

#下面分别是Reasoning about Entailment with Neural Attention的不同方法
class Conditional_encoding(nn.Module):
    def __init__(self,vect_len,weight,bidi=False):
        super(Conditional_encoding, self).__init__()
        self.embedding = nn.Embedding(vect_len + 4, 50, _weight=weight)

        self.lstm1 = nn.LSTMCell(input_size=50, hidden_size=64)
        self.lstm2 = nn.LSTM(input_size=50,hidden_size=64,num_layers=1,batch_first=True,dropout=0.3,bidirectional=bidi)
        self.l1 = nn.Sequential(nn.Linear(64, 128),nn.Dropout(),nn.Tanh())
        self.l = nn.Sequential(nn.Linear(128,4))

    def forward(self,sentence1,sentence2,s1_len,s2_len,s1_s,s2_s):
        embed1 = self.embedding(sentence1)
        embed2 = self.embedding(sentence2)

        embed1 = embed1.permute(1,0,2)
        h0 = torch.zeros(embed1.size()[1],64,requires_grad=False).cuda()
        c0 = torch.zeros(embed1.size()[1],64,requires_grad=False).cuda()
        h,c = [],[]
        # print(embed1[0].size())
        for i in range(embed1.size()[0]):
            h0,c0 = self.lstm1(embed1[i],(h0,c0))
            h += [h0]
            c += [c0]
        h = torch.stack(h)
        c = torch.stack(c)
        h = h.permute(1,0,2)
        c = c.permute(1,0,2)

        gather_h = h.gather(1, s1_len).view(1,embed2.size()[0],64)
        gather_c = c.gather(1, s1_len).view(1,embed2.size()[0],64)

        out,(_,_) = self.lstm2(embed2,(gather_h,gather_c))
        out = out.gather(1,s2_len).squeeze(1)

        out = self.l1(out)
        out = self.l(out)
        return out

    def acc(self,y_pre,label):
        lenn = len(y_pre)
        y_pre = y_pre.argmax(axis=1)
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn

        return y_acc

class Attenion(nn.Module):
    def __init__(self,vect_len,weight,bidi=False):
        super(Attenion, self).__init__()
        self.embedding = nn.Embedding(vect_len + 4, 50, _weight=weight)

        self.lstm1 = nn.LSTMCell(input_size=50, hidden_size=64)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=64, num_layers=1, batch_first=True, dropout=0.3,bidirectional=bidi)
        self.wy = nn.Parameter( nn.init.kaiming_normal_(torch.Tensor(64,64)))
        self.wh = nn.Parameter( nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.w = nn.Parameter(nn.init.normal_(torch.Tensor(64)))
        self.wp = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.wx = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.l1 = nn.Sequential(nn.Linear(64, 128), nn.Dropout(), nn.Tanh())
        self.l = nn.Sequential(nn.Linear(128, 4))

    def forward(self,sentence1,sentence2,s1_len,s2_len,s1_s,s2_s):
        embed1 = self.embedding(sentence1)
        embed2 = self.embedding(sentence2)
        b,s,_ = embed1.size()
        e = 64

        embed1 = embed1.permute(1, 0, 2)
        h0 = torch.zeros(b, e, requires_grad=False).cuda()
        c0 = torch.zeros(b, e, requires_grad=False).cuda()
        h, c = [], []
        # print(embed1[0].size())
        for i in range(embed1.size()[0]):
            h0, c0 = self.lstm1(embed1[i], (h0, c0))
            h += [h0]
            c += [c0]
        h = torch.stack(h)
        c = torch.stack(c)
        h = h.permute(1, 0, 2)
        c = c.permute(1, 0, 2)

        gather_h = h.gather(1, s1_len).view(1, b, e)
        gather_c = c.gather(1, s1_len).view(1, b, e)

        out, (_, _) = self.lstm2(embed2, (gather_h, gather_c))
        out = out.gather(1, s2_len)

        wyy = torch.matmul(h,self.wy)
        hnel = out.expand(b,s,e)
        whhn = torch.matmul(hnel,self.wh)
        M = self.tanh(wyy+whhn)
        wt = self.w.view(64,1)
        wtm = torch.matmul(M,wt).squeeze(2)
        s1_mask = s1_s*wtm-(1-s1_s)*1e12
        soft = self.softmax(s1_mask).view(b,s,1)
        soft = soft.expand(b,s,e)
        r = h*soft
        r = torch.sum(r,1)
        wpr = torch.matmul(r,self.wp)
        wxhx = torch.matmul(out.squeeze(1),self.wx)
        h_out = self.tanh(wpr+wxhx)
        h_out = self.l1(h_out)
        h_out = self.l(h_out)
        return h_out

    def acc(self, y_pre, label):
        lenn = len(y_pre)
        y_pre = y_pre.argmax(axis=1)
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn

        return y_acc

class Attenion_two_way(nn.Module):
    def __init__(self,vect_len,weight,bidi=False):
        super(Attenion_two_way, self).__init__()
        self.embedding = nn.Embedding(vect_len + 4, 50, _weight=weight)

        self.lstm1 = nn.LSTMCell(input_size=50, hidden_size=64)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=64, num_layers=1, batch_first=True, dropout=0.3,bidirectional=bidi)
        self.wy = nn.Parameter( nn.init.kaiming_normal_(torch.Tensor(64,64)))
        self.wh = nn.Parameter( nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.w = nn.Parameter(nn.init.normal_(torch.Tensor(64)))
        self.wp = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.wx = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.l1 = nn.Sequential(nn.Linear(128, 128), nn.Dropout(), nn.Tanh())
        self.l = nn.Sequential(nn.Linear(128, 4))

    def forward(self,sentence1,sentence2,s1_len,s2_len,s1_s,s2_s):
        embed1 = self.embedding(sentence1)
        embed2 = self.embedding(sentence2)
        b, s, _ = embed1.size()
        e = 64

        embed1 = embed1.permute(1, 0, 2)
        h0 = torch.zeros(b, e, requires_grad=False).cuda()
        c0 = torch.zeros(b, e, requires_grad=False).cuda()
        h, c = [], []
        # print(embed1[0].size())
        for i in range(embed1.size()[0]):
            h0, c0 = self.lstm1(embed1[i], (h0, c0))
            h += [h0]
            c += [c0]
        h = torch.stack(h)
        c = torch.stack(c)
        h = h.permute(1, 0, 2)
        c = c.permute(1, 0, 2)

        gather_h = h.gather(1, s1_len).view(1, b, e)
        gather_c = c.gather(1, s1_len).view(1, b, e)

        out, (_, _) = self.lstm2(embed2, (gather_h, gather_c))
        out = out.gather(1, s2_len)

        wyy = torch.matmul(h, self.wy)
        hnel = out.expand(b, s, e)
        whhn = torch.matmul(hnel, self.wh)
        M = self.tanh(wyy + whhn)
        wt = self.w.view(64, 1)
        wtm = torch.matmul(M, wt).squeeze(2)
        s1_mask = s1_s * wtm - (1 - s1_s) * 1e12
        soft = self.softmax(s1_mask).view(b, s, 1)
        soft = soft.expand(b, s, e)
        r = h * soft
        r = torch.sum(r, 1)
        wpr = torch.matmul(r, self.wp)
        wxhx = torch.matmul(out.squeeze(1), self.wx)
        h_out = self.tanh(wpr + wxhx)
        return h_out

    def lout(self,out1,out2):
        h_out = torch.cat([out1,out2],dim=-1)
        h_out = self.l1(h_out)
        h_out = self.l(h_out)
        return h_out


    def acc(self, y_pre, label):
        lenn = len(y_pre)
        y_pre = y_pre.argmax(axis=1)
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn

        return y_acc

class Wbw_Attenion(nn.Module):
    def __init__(self,vect_len,weight,bidi=False):
        super(Wbw_Attenion, self).__init__()
        self.embedding = nn.Embedding(vect_len + 4, 50, _weight=weight)

        self.lstm1 = nn.LSTMCell(input_size=50, hidden_size=64)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=64, num_layers=1, batch_first=True, dropout=0.3,bidirectional=bidi)
        self.wy = nn.Parameter( nn.init.kaiming_normal_(torch.Tensor(64,64)))
        self.wh = nn.Parameter( nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.w = nn.Parameter(nn.init.normal_(torch.Tensor(64)))
        self.wp = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.wx = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.wr = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.wt = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.l1 = nn.Sequential(nn.Linear(64, 128), nn.Dropout(), nn.Tanh())
        self.l = nn.Sequential(nn.Linear(128, 4))

    def forward(self,sentence1,sentence2,s1_len,s2_len,s1_s,s2_s):
        embed1 = self.embedding(sentence1)
        embed2 = self.embedding(sentence2)
        b,s,_ = embed1.size()
        _,s2,_ = embed2.size()
        e = 64

        embed1 = embed1.permute(1, 0, 2)
        h0 = torch.zeros(b, e, requires_grad=False).cuda()
        c0 = torch.zeros(b, e, requires_grad=False).cuda()
        h, c = [], []
        # print(embed1[0].size())
        for i in range(embed1.size()[0]):
            h0, c0 = self.lstm1(embed1[i], (h0, c0))
            h += [h0]
            c += [c0]
        h = torch.stack(h)
        c = torch.stack(c)
        h = h.permute(1, 0, 2)
        c = c.permute(1, 0, 2)

        gather_h = h.gather(1, s1_len).view(1, b, e)
        gather_c = c.gather(1, s1_len).view(1, b, e)

        out, (_, _) = self.lstm2(embed2, (gather_h, gather_c))
        # out = out.gather(1, s2_len)
        r = torch.zeros(b,e,requires_grad=False).cuda()
        Rt = []
        for i in range(s2):
            wyy = torch.matmul(h,self.wy)
            hnel = out.expand(b,s,e)
            whhn = torch.matmul(hnel,self.wh)
            rtel = r.unsqueeze(1).expand(b,s,e)
            wrrt = torch.matmul(rtel,self.wr)
            M = self.tanh(wyy+whhn+wrrt)

            wt = self.w.view(64,1)
            wtm = torch.matmul(M,wt).squeeze(2)
            s1_mask = s1_s*wtm-(1-s1_s)*1e12
            soft = self.softmax(s1_mask).view(b,s,1)
            soft = soft.expand(b,s,e)

            wtrt = torch.matmul(r,self.wt)
            r = h*soft
            r = torch.sum(r,1)+self.tanh(wtrt)
            Rt += [r]
        Rt = torch.stack(Rt)
        Rt = Rt.permute(1,0,2)
        rn = Rt.gather(1,s2_len).squeeze(1)
        hn_out = out.gather(1, s2_len)
        wpr = torch.matmul(rn,self.wp)
        wxhx = torch.matmul(hn_out.squeeze(1),self.wx)
        h_out = self.tanh(wpr+wxhx)
        h_out = self.l1(h_out)
        h_out = self.l(h_out)
        return h_out

    def acc(self, y_pre, label):
        lenn = len(y_pre)
        y_pre = y_pre.argmax(axis=1)
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn

        return y_acc

#two_way就是不同顺序运行两次
class Wbw_Attenion_two_way(nn.Module):
    def __init__(self,vect_len,weight,bidi=False):
        super(Wbw_Attenion_two_way, self).__init__()
        self.embedding = nn.Embedding(vect_len + 4, 50, _weight=weight)

        self.lstm1 = nn.LSTMCell(input_size=50, hidden_size=64)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=64, num_layers=1, batch_first=True, dropout=0.3,bidirectional=bidi)
        self.wy = nn.Parameter( nn.init.kaiming_normal_(torch.Tensor(64,64)))
        self.wh = nn.Parameter( nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.w = nn.Parameter(nn.init.normal_(torch.Tensor(64)))
        self.wp = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.wx = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.wr = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.wt = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.l1 = nn.Sequential(nn.Linear(128, 128), nn.Dropout(), nn.Tanh())
        self.l = nn.Sequential(nn.Linear(128, 4))

    def forward(self,sentence1,sentence2,s1_len,s2_len,s1_s,s2_s):
        embed1 = self.embedding(sentence1)
        embed2 = self.embedding(sentence2)
        b,s,_ = embed1.size()
        _,s2,_ = embed2.size()
        e = 64

        embed1 = embed1.permute(1, 0, 2)
        h0 = torch.zeros(b, e, requires_grad=False).cuda()
        c0 = torch.zeros(b, e, requires_grad=False).cuda()
        h, c = [], []
        # print(embed1[0].size())
        for i in range(embed1.size()[0]):
            h0, c0 = self.lstm1(embed1[i], (h0, c0))
            h += [h0]
            c += [c0]
        h = torch.stack(h)
        c = torch.stack(c)
        h = h.permute(1, 0, 2)
        c = c.permute(1, 0, 2)

        gather_h = h.gather(1, s1_len).view(1, b, e)
        gather_c = c.gather(1, s1_len).view(1, b, e)

        out, (_, _) = self.lstm2(embed2, (gather_h, gather_c))
        # out = out.gather(1, s2_len)
        r = torch.zeros(b,e,requires_grad=False).cuda()
        Rt = []
        for i in range(s2):
            wyy = torch.matmul(h,self.wy)
            hnel = out.expand(b,s,e)
            whhn = torch.matmul(hnel,self.wh)
            rtel = r.unsqueeze(1).expand(b,s,e)
            wrrt = torch.matmul(rtel,self.wr)
            M = self.tanh(wyy+whhn+wrrt)

            wt = self.w.view(64,1)
            wtm = torch.matmul(M,wt).squeeze(2)
            s1_mask = s1_s*wtm-(1-s1_s)*1e12
            soft = self.softmax(s1_mask).view(b,s,1)
            soft = soft.expand(b,s,e)

            wtrt = torch.matmul(r,self.wt)
            r = h*soft
            r = torch.sum(r,1)+self.tanh(wtrt)
            Rt += [r]
        Rt = torch.stack(Rt)
        Rt = Rt.permute(1,0,2)
        rn = Rt.gather(1,s2_len).squeeze(1)
        hn_out = out.gather(1, s2_len)
        wpr = torch.matmul(rn,self.wp)
        wxhx = torch.matmul(hn_out.squeeze(1),self.wx)
        h_out = self.tanh(wpr+wxhx)

        return h_out

    def lout(self,out1,out2):
        h_out = torch.cat([out1,out2],-1)
        h_out = self.l1(h_out)
        h_out = self.l(h_out)
        return h_out

    def acc(self, y_pre, label):
        lenn = len(y_pre)
        y_pre = y_pre.argmax(axis=1)
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn

        return y_acc

class ESIM(nn.Module):
    def __init__(self,vect_len,weight):
        super(ESIM, self).__init__()
        self.embedding = nn.Embedding(vect_len + 4, 50, _weight=weight)
        self.lstm1 = nn.LSTM(input_size=50, hidden_size=64, num_layers=1, batch_first=True, dropout=0.3,bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=64, num_layers=1, batch_first=True, dropout=0.3,bidirectional=True)
        self.soft2 = nn.Softmax(dim=2)
        self.lstm3 = nn.LSTM(input_size=512, hidden_size=64, num_layers=1, batch_first=True, dropout=0.3,bidirectional=True)
        self.l1 = nn.Sequential(nn.Linear(512, 128), nn.Dropout(), nn.Tanh())
        self.l = nn.Sequential(nn.Linear(128, 4))

    def forward(self,sentence1,sentence2,s1_s,s2_s,m_mask):
        embed1 = self.embedding(sentence1)
        embed2 = self.embedding(sentence2)
        b, s, _ = embed1.size()
        _, s2, _ = embed2.size()
        em = 64

        lstm1,_ = self.lstm1(embed1)
        lstm2,_ = self.lstm2(embed2)

        s1_s = s1_s.view(b,s,1).expand(b,s,em*2)
        s2_s = s2_s.view(b,s2,1).expand(b,s2,em*2)
        lstm1 = lstm1*s1_s
        lstm2 = lstm2*s2_s

        e = torch.matmul(lstm1,lstm2.permute(0,2,1))
        e = e*m_mask-(1-m_mask)*1e12
        a_t = self.soft2(e)
        a_t = torch.matmul(a_t,lstm2)
        e = e.permute(0,2,1)
        b_t = self.soft2(e)
        b_t = torch.matmul(b_t,lstm1)

        ma = torch.cat([lstm1,a_t,lstm1-a_t,lstm1*a_t],dim=-1)
        mb = torch.cat([lstm2,b_t,lstm2-b_t,lstm2*b_t],dim=-1)

        va,_ = self.lstm3(ma)
        vb,_ = self.lstm3(mb)
        va = va.permute(0,2,1)
        vb = vb.permute(0,2,1)
        va_vag = F.avg_pool1d(va,va.size()[-1]).squeeze(2)
        va_max = F.max_pool1d(va,va.size()[-1]).squeeze(2)
        vb_vag = F.avg_pool1d(vb,vb.size()[-1]).squeeze(2)
        vb_max = F.max_pool1d(vb,vb.size()[-1]).squeeze(2)
        out = torch.cat([va_vag,va_max,vb_vag,vb_max],dim=-1)

        out = self.l1(out)
        out = self.l(out)

        return out

    def acc(self, y_pre, label):
        lenn = len(y_pre)
        y_pre = y_pre.argmax(axis=1)
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn

        return y_acc

#test应与dev对调
def trainning(s1_train, s1_test,s1_val,s2_train, s2_test,s2_val, y_train, y_test,y_val,s1_len_train,s1_len_test,s1_len_val,s2_len_train,s2_len_test,s2_len_val,s1_s_train,s1_s_test,s1_s_val,s2_s_train,s2_s_test,s2_s_val,epochs,model_t,batchs):
    train_data = TensorDataset(s1_train,s2_train, y_train,s1_len_train,s2_len_train,s1_s_train,s2_s_train)
    test_data = TensorDataset(s1_test,s2_test, y_test,s1_len_test,s2_len_test,s1_s_test,s2_s_test)
    val_data = TensorDataset(s1_val,s2_val, y_val,s1_len_val,s2_len_val,s1_s_val,s2_s_val)

    train_iter = DataLoader(train_data, shuffle=True, batch_size=batchs)
    test_iter = DataLoader(test_data, shuffle=True, batch_size=batchs)
    val_iter = DataLoader(val_data, shuffle=True, batch_size=batchs)

    model = model_t
    model.to(torch.device('cuda:0'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0006, weight_decay=0.0003)
    model.train()
    loss_s = FocalLoss()

    max_train_acc, max_test_acc, val_acc = 0, 0, 0
    for epoch in range(epochs):
        for batch in train_iter:
            s1_train_t,s2_train_t, y_train_t,s1_len_train_t,s2_len_train_t,s1_s_train_t,s2_s_train_t = batch
            s1_train_t,s2_train_t, y_train_t,s1_len_train_t,s2_len_train_t,s1_s_train_t,s2_s_train_t = \
                s1_train_t.cuda(),s2_train_t.cuda(), y_train_t.cuda(),s1_len_train_t.cuda(),s2_len_train_t.cuda(),s1_s_train_t.cuda(),s2_s_train_t.cuda()

            out = model(s1_train_t, s2_train_t,s1_len_train_t,s2_len_train_t,s1_s_train_t,s2_s_train_t)
            loss = loss_s(out, y_train_t)
            loss.backward()
            optimizer.step()
            model.zero_grad()
        if epoch % 2 == 0 and epoch != 0:
            model.eval()
            with torch.no_grad():
                flag = 0
                total_test, total_label_test = None, None
                for batch in test_iter:
                    s1_train_t, s2_train_t, y_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t = batch
                    s1_train_t, s2_train_t, y_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t = \
                        s1_train_t.cuda(), s2_train_t.cuda(), y_train_t.cuda(), s1_len_train_t.cuda(), s2_len_train_t.cuda(), s1_s_train_t.cuda(), s2_s_train_t.cuda()

                    out = model(s1_train_t, s2_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t)
                    if flag == 0:
                        total_test = out
                        total_label_test = y_train_t
                        flag = 1
                    else:
                        total_test = torch.cat((total_test, out), 0)
                        total_label_test = torch.cat((total_label_test, y_train_t), 0)
                test_acc = model.acc(total_test, total_label_test)
                flag = 0
                total_test, total_label_test = None, None
                if max_test_acc < test_acc:
                    max_test_acc = test_acc
                    for batch in val_iter:
                        s1_train_t, s2_train_t, y_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t = batch
                        s1_train_t, s2_train_t, y_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t = \
                            s1_train_t.cuda(), s2_train_t.cuda(), y_train_t.cuda(), s1_len_train_t.cuda(), s2_len_train_t.cuda(), s1_s_train_t.cuda(), s2_s_train_t.cuda()

                        out = model(s1_train_t, s2_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t)
                        if flag == 0:
                            total_test = out
                            total_label_test = y_train_t
                            flag = 1
                        else:
                            total_test = torch.cat((total_test, out), 0)
                            total_label_test = torch.cat((total_label_test, y_train_t), 0)
                    val_acc = model.acc(total_test, total_label_test)

            model.train()
    return max_test_acc,val_acc

def two_way_trainning(s1_train, s1_test,s2_train, s2_test, y_train, y_test,s1_len_train,s1_len_test,s2_len_train,s2_len_test,s1_s_train,s1_s_test,s2_s_train,s2_s_test,epochs,model_t,batchs):
    train_data = TensorDataset(s1_train, s2_train, y_train, s1_len_train, s2_len_train, s1_s_train, s2_s_train)
    test_data = TensorDataset(s1_test, s2_test, y_test, s1_len_test, s2_len_test, s1_s_test, s2_s_test)

    batch = batchs
    train_iter = DataLoader(train_data, shuffle=True, batch_size=batch)
    test_iter = DataLoader(test_data, shuffle=True, batch_size=batch)

    model = model_t
    model.to(torch.device('cuda:0'))
    print("train:", torch.cuda.memory_allocated())

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0006, weight_decay=0.0003)
    model.train()
    loss_s = FocalLoss()

    train_acc_total, test_acc_total = [], []
    for epoch in range(epochs):
        for batch in train_iter:
            s1_train_t, s2_train_t, y_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t = batch
            s1_train_t, s2_train_t, y_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t = \
                s1_train_t.cuda(), s2_train_t.cuda(), y_train_t.cuda(), s1_len_train_t.cuda(), s2_len_train_t.cuda(), s1_s_train_t.cuda(), s2_s_train_t.cuda()

            print("train:", torch.cuda.memory_allocated())
            out1 = model(s1_train_t, s2_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t)
            out2 = model(s2_train_t, s1_train_t, s2_len_train_t, s1_len_train_t, s2_s_train_t, s1_s_train_t)
            print("train——out12:", torch.cuda.memory_allocated())
            out = model.lout(out1,out2)
            print("train--out:",torch.cuda.memory_allocated())
            loss = loss_s(out, y_train_t)
            print("train--loss:", torch.cuda.memory_allocated())
            loss.backward()
            print("train--back:", torch.cuda.memory_allocated())
            optimizer.step()
            print("train--optimizer:", torch.cuda.memory_allocated())
            model.zero_grad()
        if epoch % 2 == 0 and epoch != 0:
            model.eval()
            with torch.no_grad():
                flag = 0
                total_train, total_label_train = None, None
                # if epoch>40:
                #     for name, parms in model.named_parameters():
                #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->parm_value:', parms.data)
                for batch in train_iter:
                    s1_train_t, s2_train_t, y_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t = batch
                    s1_train_t, s2_train_t, y_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t = \
                        s1_train_t.cuda(), s2_train_t.cuda(), y_train_t.cuda(), s1_len_train_t.cuda(), s2_len_train_t.cuda(), s1_s_train_t.cuda(), s2_s_train_t.cuda()
                    print("test", torch.cuda.memory_allocated())
                    out1 = model(s1_train_t, s2_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t)
                    out2 = model(s2_train_t, s1_train_t, s2_len_train_t, s1_len_train_t, s2_s_train_t, s1_s_train_t)
                    out = model.lout(out1, out2)
                    print("test--out",torch.cuda.memory_allocated())
                    if flag == 0:
                        total_train = out
                        total_label_train = y_train_t
                        flag = 1
                    else:
                        total_train = torch.cat((total_train, out), 0)
                        total_label_train = torch.cat((total_label_train, y_train_t), 0)
                train_acc = model.acc(total_train, total_label_train)
                flag = 0
                total_test, total_label_test = None, None
                for batch in test_iter:
                    s1_train_t, s2_train_t, y_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t = batch
                    s1_train_t, s2_train_t, y_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t = \
                        s1_train_t.cuda(), s2_train_t.cuda(), y_train_t.cuda(), s1_len_train_t.cuda(), s2_len_train_t.cuda(), s1_s_train_t.cuda(), s2_s_train_t.cuda()

                    out1 = model(s1_train_t, s2_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t)
                    out2 = model(s2_train_t, s1_train_t, s2_len_train_t, s1_len_train_t, s2_s_train_t, s1_s_train_t)
                    out = model.lout(out1, out2)
                    if flag == 0:
                        total_test = out
                        total_label_test = y_train_t
                        flag = 1
                    else:
                        total_test = torch.cat((total_test, out), 0)
                        total_label_test = torch.cat((total_label_test, y_train_t), 0)

                test_acc = model.acc(total_test, total_label_test)
            train_acc_total.append(train_acc)
            test_acc_total.append(test_acc)
            print("epoch:", epoch, "----train_acc:", train_acc, "----test_acc:", test_acc)
            model.train()
    return train_acc_total, test_acc_total

def ESIM_trainning(s1_train, s1_test,s1_val,s2_train, s2_test,s2_val, y_train, y_test,y_val,s1_s_train,s1_s_test,s1_s_val,s2_s_train,s2_s_test,s2_s_val,train_m_mask,test_m_mask,val_m_mask,epochs,model_t,batchs):
    train_data = TensorDataset(s1_train,s2_train, y_train,s1_s_train,s2_s_train,train_m_mask)
    test_data = TensorDataset(s1_test,s2_test, y_test,s1_s_test,s2_s_test,test_m_mask)
    val_data = TensorDataset(s1_val, s2_val, y_val, s1_s_val, s2_s_val, val_m_mask)

    train_iter = DataLoader(train_data, shuffle=True, batch_size=batchs)
    test_iter = DataLoader(test_data, shuffle=True, batch_size=batchs)
    val_iter = DataLoader(val_data, shuffle=True, batch_size=batchs)

    model = model_t
    model.to(torch.device('cuda:0'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0006, weight_decay=0.0003)
    model.train()
    loss_s = FocalLoss()

    train_acc_total, test_acc_total = [], []
    max_train_acc,max_test_acc, val_acc = 0,0, 0
    for epoch in range(epochs):
        for batch in train_iter:
            s1_train_t,s2_train_t, y_train_t,s1_s_train_t,s2_s_train_t,train_m_mask_t = batch
            s1_train_t,s2_train_t, y_train_t,s1_s_train_t,s2_s_train_t,train_m_mask_t = \
                s1_train_t.cuda(),s2_train_t.cuda(), y_train_t.cuda(),s1_s_train_t.cuda(),s2_s_train_t.cuda(),train_m_mask_t.cuda()

            out = model(s1_train_t, s2_train_t,s1_s_train_t,s2_s_train_t,train_m_mask_t)
            loss = loss_s(out, y_train_t)
            loss.backward()
            optimizer.step()
            model.zero_grad()
        if epoch % 2 == 0 and epoch != 0:
            model.eval()
            flag = 0
            total_train, total_label_train = None, None
            with torch.no_grad():
                for batch in train_iter:
                    s1_train_t, s2_train_t, y_train_t, s1_s_train_t, s2_s_train_t, train_m_mask_t = batch
                    s1_train_t, s2_train_t, y_train_t, s1_s_train_t, s2_s_train_t, train_m_mask_t = \
                        s1_train_t.cuda(), s2_train_t.cuda(), y_train_t.cuda(), s1_s_train_t.cuda(), s2_s_train_t.cuda(), train_m_mask_t.cuda()

                    out = model(s1_train_t, s2_train_t, s1_s_train_t, s2_s_train_t, train_m_mask_t)
                    if flag == 0:
                        total_train = out
                        total_label_train = y_train_t
                        flag = 1
                    else:
                        total_train = torch.cat((total_train, out), 0)
                        total_label_train = torch.cat((total_label_train, y_train_t), 0)
                train_acc = model.acc(total_train, total_label_train)
                flag = 0
                total_test, total_label_test = None, None
                for batch in test_iter:
                    s1_train_t, s2_train_t, y_train_t, s1_s_train_t, s2_s_train_t, train_m_mask_t = batch
                    s1_train_t, s2_train_t, y_train_t, s1_s_train_t, s2_s_train_t, train_m_mask_t = \
                        s1_train_t.cuda(), s2_train_t.cuda(), y_train_t.cuda(), s1_s_train_t.cuda(), s2_s_train_t.cuda(), train_m_mask_t.cuda()

                    out = model(s1_train_t, s2_train_t, s1_s_train_t, s2_s_train_t, train_m_mask_t)
                    if flag == 0:
                        total_test = out
                        total_label_test = y_train_t
                        flag = 1
                    else:
                        total_test = torch.cat((total_test, out), 0)
                        total_label_test = torch.cat((total_label_test, y_train_t), 0)
                test_acc = model.acc(total_test, total_label_test)
                flag = 0
                total_test, total_label_test = None, None
                if max_test_acc < test_acc:
                    max_train_acc = train_acc
                    max_test_acc = test_acc
                    for batch in val_iter:
                        s1_train_t, s2_train_t, y_train_t, s1_s_train_t, s2_s_train_t, train_m_mask_t = batch
                        s1_train_t, s2_train_t, y_train_t, s1_s_train_t, s2_s_train_t, train_m_mask_t = \
                            s1_train_t.cuda(), s2_train_t.cuda(), y_train_t.cuda(), s1_s_train_t.cuda(), s2_s_train_t.cuda(), train_m_mask_t.cuda()

                        out = model(s1_train_t, s2_train_t, s1_s_train_t, s2_s_train_t, train_m_mask_t)
                        if flag == 0:
                            total_test = out
                            total_label_test = y_train_t
                            flag = 1
                        else:
                            total_test = torch.cat((total_test, out), 0)
                            total_label_test = torch.cat((total_label_test, y_train_t), 0)
                    val_acc = model.acc(total_test, total_label_test)
            train_acc_total.append(train_acc)
            test_acc_total.append(test_acc)
            print("epoch:", epoch, "----train_acc:", max_test_acc, "----test_acc:", val_acc)
            model.train()
    return max_test_acc,val_acc

def self_attention_trainning(train_bert_s_id,test_bert_s_id,val_bert_s_id,train_bert_mask,test_bert_mask,val_bert_mask,train_token_type,test_token_type,val_token_type,y_train, y_test,y_val,epochs,model_t,batchs):
    train_data = TensorDataset(train_bert_s_id,train_bert_mask,train_token_type,y_train)
    test_data = TensorDataset(test_bert_s_id,test_bert_mask,test_token_type,y_test)
    val_data = TensorDataset(val_bert_s_id, val_bert_mask, val_token_type, y_val)

    train_iter = DataLoader(train_data, shuffle=True, batch_size=batchs)
    test_iter = DataLoader(test_data, shuffle=True, batch_size=batchs)
    val_iter = DataLoader(val_data, shuffle=True, batch_size=batchs)

    model = model_t
    model.to(torch.device('cuda:0'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.00001)
    model.train()
    loss_s = FocalLoss()

    train_acc_total, test_acc_total = [], []
    max_train_acc,max_test_acc,val_acc = 0,0,0
    for epoch in range(epochs):
        for batch in train_iter:
            train_bert_s_id_t,train_bert_mask_t,train_token_type_t,y_train_t = batch
            train_bert_s_id_t, train_bert_mask_t, train_token_type_t,y_train_t = train_bert_s_id_t.cuda(),train_bert_mask_t.cuda(),train_token_type_t.cuda(),y_train_t.cuda()

            out = model(train_bert_s_id_t,train_token_type_t,train_bert_mask_t)
            loss = loss_s(out, y_train_t)
            loss.backward()
            # if epoch>40:
            for name, parms in model.named_parameters():
                if parms.requires_grad is True:
                    if isinstance(parms.grad,torch.Tensor):
                        print('-->name:', name, ' -->grad_mean:', parms.grad.mean(),' -->grad_mean:', parms.grad.std())
            optimizer.step()
            model.zero_grad()
        if epoch % 2 == 0 and epoch != 0:
            model.eval()
            flag = 0
            total_train, total_label_train = None, None
            with torch.no_grad():
                for batch in train_iter:
                    train_bert_s_id_t, train_bert_mask_t, train_token_type_t, y_train_t = batch
                    train_bert_s_id_t, train_bert_mask_t, train_token_type_t, y_train_t = train_bert_s_id_t.cuda(), train_bert_mask_t.cuda(), train_token_type_t.cuda(), y_train_t.cuda()

                    out = model(train_bert_s_id_t, train_token_type_t, train_bert_mask_t)
                    if flag == 0:
                        total_train = out
                        total_label_train = y_train_t
                        flag = 1
                    else:
                        total_train = torch.cat((total_train, out), 0)
                        total_label_train = torch.cat((total_label_train, y_train_t), 0)
                train_acc = model.acc(total_train, total_label_train)
                flag = 0
                total_test, total_label_test = None, None
                for batch in test_iter:
                    train_bert_s_id_t, train_bert_mask_t, train_token_type_t, y_train_t = batch
                    train_bert_s_id_t, train_bert_mask_t, train_token_type_t, y_train_t = train_bert_s_id_t.cuda(), train_bert_mask_t.cuda(), train_token_type_t.cuda(), y_train_t.cuda()

                    out = model(train_bert_s_id_t, train_token_type_t, train_bert_mask_t)
                    if flag == 0:
                        total_test = out
                        total_label_test = y_train_t
                        flag = 1
                    else:
                        total_test = torch.cat((total_test, out), 0)
                        total_label_test = torch.cat((total_label_test, y_train_t), 0)
                test_acc = model.acc(total_test, total_label_test)
                flag = 0
                total_test, total_label_test = None, None
                if max_test_acc<test_acc:
                    max_train_acc = train_acc
                    max_test_acc = test_acc
                    for batch in val_iter:
                        train_bert_s_id_t, train_bert_mask_t, train_token_type_t, y_train_t = batch
                        train_bert_s_id_t, train_bert_mask_t, train_token_type_t, y_train_t = train_bert_s_id_t.cuda(), train_bert_mask_t.cuda(), train_token_type_t.cuda(), y_train_t.cuda()

                        out = model(train_bert_s_id_t, train_token_type_t, train_bert_mask_t)
                        if flag == 0:
                            total_test = out
                            total_label_test = y_train_t
                            flag = 1
                        else:
                            total_test = torch.cat((total_test, out), 0)
                            total_label_test = torch.cat((total_label_test, y_train_t), 0)
                    val_acc = model.acc(total_test, total_label_test)

            train_acc_total.append(train_acc)
            test_acc_total.append(test_acc)
            print("epoch:", epoch, "----train_acc:", train_acc, "----test_acc:", test_acc)
            model.train()
    return train_acc_total, test_acc_total,max_train_acc,max_test_acc,val_acc

if __name__=='__main__':
    setup_seed(123)
    train_sen1,train_sen2,train_label = read_data('./data/snli_1.0/convert_train.jsonl')
    test_sen1, test_sen2, test_label = read_data('./data/snli_1.0/convert_test.jsonl')
    dev_sen1, dev_sen2, dev_label = read_data('./data/snli_1.0/convert_dev.jsonl')

    #同，懒
    train_s1_len = [len(i) for i in train_sen1]
    train_s2_len = [len(i) for i in train_sen2]
    test_s1_len = [len(i) for i in test_sen1]
    test_s2_len = [len(i) for i in test_sen2]
    dev_s1_len = [len(i) for i in dev_sen1]
    dev_s2_len = [len(i) for i in dev_sen2]

    max_len = 0
    for i in range(len(train_sen1)):
        max_len = max(max_len, train_s1_len[i],train_s2_len[i])
    for i in range(len(test_sen1)):
        max_len = max(max_len, test_s1_len[i], test_s2_len[i])
    for i in range(len(dev_sen1)):
        max_len = max(max_len, dev_s1_len[i], dev_s2_len[i])

    veccc = {}
    vect_len = 0
    for i in range(len(train_sen1)):
        for j, k in enumerate(train_sen1[i]):
            # print(k)
            if k not in veccc.keys():
                veccc[k] = vect_len
                vect_len += 1
        for j, k in enumerate(train_sen2[i]):
            # print(k)
            if k not in veccc.keys():
                veccc[k] = vect_len
                vect_len += 1
    for i in range(len(test_sen1)):
        for j, k in enumerate(test_sen1[i]):
            # print(k)
            if k not in veccc.keys():
                veccc[k] = vect_len
                vect_len += 1
        for j, k in enumerate(test_sen2[i]):
            # print(k)
            if k not in veccc.keys():
                veccc[k] = vect_len
                vect_len += 1
    for i in range(len(dev_sen1)):
        for j, k in enumerate(dev_sen1[i]):
            # print(k)
            if k not in veccc.keys():
                veccc[k] = vect_len
                vect_len += 1
        for j, k in enumerate(dev_sen2[i]):
            # print(k)
            if k not in veccc.keys():
                veccc[k] = vect_len
                vect_len += 1

    train_labels,test_labels,dev_labels = [],[],[]
    for i in train_label:
        if i=="entailment":
            train_labels.append(0)
        elif i=="neutral":
            train_labels.append(1)
        elif i=="-":
            train_labels.append(2)
        elif i=="contradiction":
            train_labels.append(3)
    for i in test_label:
        if i=="entailment":
            test_labels.append(0)
        elif i=="neutral":
            test_labels.append(1)
        elif i=="-":
            test_labels.append(2)
        elif i=="contradiction":
            test_labels.append(3)
    for i in dev_label:
        if i=="entailment":
            dev_labels.append(0)
        elif i=="neutral":
            dev_labels.append(1)
        elif i=="-":
            dev_labels.append(2)
        elif i=="contradiction":
            dev_labels.append(3)

    veccc['[UNK]'] = vect_len
    veccc['[PAD]'] = vect_len + 1
    veccc['[CLS]'] = vect_len + 2
    veccc['[SEP]'] = vect_len + 3
    glove_embed = get_numpy_word_embed(veccc,vect_len)
    glove_embed = torch.FloatTensor(glove_embed)
    train_m_mask = martix_mask(train_s1_len, train_s2_len, max_len)
    test_m_mask = martix_mask(test_s1_len, test_s2_len, max_len)
    dev_m_mask = martix_mask(dev_s1_len, dev_s2_len, max_len)
    #使用lstm会有rnn一样的问题，需要取该长度的掩码；第二个为注意力机制的掩码
    train_s1_len_trans,train_s1_softmax = len_mask_g(train_s1_len,max_len)
    train_s2_len_trans,train_s2_softmax = len_mask_g(train_s2_len,max_len)
    test_s1_len_trans, test_s1_softmax = len_mask_g(test_s1_len, max_len)
    test_s2_len_trans, test_s2_softmax = len_mask_g(test_s2_len, max_len)
    dev_s1_len_trans, dev_s1_softmax = len_mask_g(dev_s1_len, max_len)
    dev_s2_len_trans, dev_s2_softmax = len_mask_g(dev_s2_len, max_len)

    #句子中每个词向量的id
    train_sentence1_id = id_generation(train_sen1, veccc,vect_len)
    train_sentence2_id = id_generation(train_sen2, veccc,vect_len)
    test_sentence1_id = id_generation(test_sen1, veccc, vect_len)
    test_sentence2_id = id_generation(test_sen2, veccc, vect_len)
    dev_sentence1_id = id_generation(dev_sen1, veccc, vect_len)
    dev_sentence2_id = id_generation(dev_sen2, veccc, vect_len)

    train_sentence1_id = [torch.LongTensor(i) for i in train_sentence1_id]
    train_sentence2_id = [torch.LongTensor(i) for i in train_sentence2_id]
    train_sentence1_id = pad_sequence(train_sentence1_id, batch_first=True, padding_value=vect_len)
    train_sentence2_id = pad_sequence(train_sentence2_id, batch_first=True, padding_value=vect_len)
    train_y = torch.tensor(train_labels).long()
    test_sentence1_id = [torch.LongTensor(i) for i in test_sentence1_id]
    test_sentence2_id = [torch.LongTensor(i) for i in test_sentence2_id]
    test_sentence1_id = pad_sequence(test_sentence1_id, batch_first=True, padding_value=vect_len)
    test_sentence2_id = pad_sequence(test_sentence2_id, batch_first=True, padding_value=vect_len)
    test_y = torch.tensor(test_labels).long()
    dev_sentence1_id = [torch.LongTensor(i) for i in dev_sentence1_id]
    dev_sentence2_id = [torch.LongTensor(i) for i in dev_sentence2_id]
    dev_sentence1_id = pad_sequence(dev_sentence1_id, batch_first=True, padding_value=vect_len)
    dev_sentence2_id = pad_sequence(dev_sentence2_id, batch_first=True, padding_value=vect_len)
    dev_y = torch.tensor(dev_labels).long()

    model = Conditional_encoding(vect_len, glove_embed, False)
    con_dev,con_test = trainning(train_sentence1_id, dev_sentence1_id, test_sentence1_id, train_sentence2_id, dev_sentence2_id, test_sentence2_id, train_y, dev_y, test_y, train_s1_len_trans, dev_s1_len_trans,
              test_s1_len_trans, train_s2_len_trans, dev_s2_len_trans, test_s2_len_trans, train_s1_softmax, dev_s1_softmax, test_s1_softmax, train_s2_softmax, dev_s2_softmax,
              test_s2_softmax, epochs=60, model_t=model, batchs=256)

    model = Attenion(vect_len, glove_embed, False)
    attention_dev, attention_test = trainning(train_sentence1_id, dev_sentence1_id, test_sentence1_id, train_sentence2_id,
                                  dev_sentence2_id, test_sentence2_id, train_y, dev_y, test_y, train_s1_len_trans,dev_s1_len_trans,test_s1_len_trans, train_s2_len_trans, dev_s2_len_trans, test_s2_len_trans,
                                  train_s1_softmax, dev_s1_softmax, test_s1_softmax, train_s2_softmax, dev_s2_softmax,test_s2_softmax, epochs=60, model_t=model, batchs=256)
    model = Wbw_Attenion(vect_len, glove_embed, False)
    wbw_dev, wbw_test = trainning(train_sentence1_id, dev_sentence1_id, test_sentence1_id,train_sentence2_id,
                                              dev_sentence2_id, test_sentence2_id, train_y, dev_y, test_y,train_s1_len_trans, dev_s1_len_trans, test_s1_len_trans,
                                              train_s2_len_trans, dev_s2_len_trans, test_s2_len_trans,train_s1_softmax, dev_s1_softmax, test_s1_softmax, train_s2_softmax,dev_s2_softmax, test_s2_softmax, epochs=60, model_t=model, batchs=64)
    model = ESIM(vect_len, glove_embed)
    esim_dev,esim_test = ESIM_trainning(train_sentence1_id, dev_sentence1_id, test_sentence1_id,train_sentence2_id,
                                              dev_sentence2_id, test_sentence2_id, train_y, dev_y, test_y,train_s1_softmax, dev_s1_softmax,
                                            test_s1_softmax, train_s2_softmax,dev_s2_softmax, test_s2_softmax, train_m_mask, dev_m_mask,test_m_mask, epochs=60, model_t=model,batchs=256)
    print("con:",con_dev,con_test)
    print("attention:",attention_dev,attention_test)
    print("wbw:",wbw_dev,wbw_test)
    print("esim:",esim_dev,esim_test)

    # two_way比较，函数参数要按上面修改一下
    # epochs = []
    # for i in range(2,50,2):
    #     epochs.append(i)
    #
    # attention,attention_two_way,wbw_attention,wbw_attention_two_way = [[],[]],[[],[]],[[],[]],[[],[]]
    #
    # model = Attenion(vect_len, glove_embed, False)
    # attention[0],attention[1] = trainning(s1_train, s1_test,s2_train, s2_test, y_train, y_test,s1_len_train,s1_len_test,s2_len_train,s2_len_test,s1_s_train,s1_s_test,s2_s_train,s2_s_test,epochs=50,model_t=model,batchs=256)
    # model = Attenion_two_way(vect_len, glove_embed, False)
    # attention_two_way[0], attention_two_way[1] = two_way_trainning(s1_train, s1_test,s2_train, s2_test, y_train, y_test,s1_len_train,s1_len_test,s2_len_train,s2_len_test,s1_s_train,s1_s_test,s2_s_train,s2_s_test,epochs=50,model_t=model,batchs=128)
    # model = Wbw_Attenion(vect_len, glove_embed, False)
    # wbw_attention[0], wbw_attention[1] = trainning(s1_train, s1_test, s2_train, s2_test, y_train, y_test, s1_len_train,s1_len_test, s2_len_train, s2_len_test, s1_s_train, s1_s_test,s2_s_train, s2_s_test, epochs=50, model_t=model, batchs=64)
    # model = Wbw_Attenion_two_way(vect_len, glove_embed, False)
    # wbw_attention_two_way[0], wbw_attention_two_way[1] = two_way_trainning(s1_train, s1_test, s2_train, s2_test, y_train, y_test,s1_len_train, s1_len_test, s2_len_train, s2_len_test,s1_s_train, s1_s_test, s2_s_train, s2_s_test,epochs=50, model_t=model, batchs=32)
