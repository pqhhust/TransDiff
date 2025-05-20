import csv
from collections import Counter
import random
import math
import torch
import torch.nn.functional as F

def process_batch(data, word_to_int, device): 
    batch_size = len(data)       
    dd_temp,data_temp,ans_temp,max_len=[],[],[],0
    for sentence in data:
        temp=[]
        sent=sentence[0]
        dd_temp.append(sent)
        answer=sentence[1]
        for word in sent:                               
            if word in word_to_int:  
                temp.append(word_to_int[word])
            else:
                temp.append(word_to_int['@unk'])
        data_temp.append(temp)
        ans_temp.append(answer)
        max_len=max(max_len,len(temp))
    input=torch.zeros(batch_size,max_len).long().to(device)
    input_mask=torch.zeros(batch_size,max_len).long().to(device)
    pos=torch.zeros(batch_size,max_len).long().to(device)
    answers=torch.tensor(ans_temp).long().to(device)
    for i,sentence in enumerate(data_temp):
        input_mask[i][:len(sentence)]=1
        for j,word in enumerate(sentence):
            input[i][j]=word
            pos[i][j]=j
    return dd_temp,input,input_mask,pos,answers


def get_data(path_list, ood_path, seed):
    data_gold=[]
    for path in path_list:
        with open(path) as fd:
            rd=csv.reader(fd, delimiter="\t", quotechar='"')
            for line in rd:
                li=line[-1].split()
                data_gold.append((li, int(line[1])))
    N = len(data_gold)
    train_size = math.ceil(N* 0.7)
    test_size = math.ceil(N* 0.2)
    val_size = N-train_size-test_size
    # random.seed(seed)
    random.Random(42).shuffle(data_gold)
    data_train = []
    gold_train = []
    data_test = []
    gold_test = []
    for dg in data_gold[:train_size+val_size]:
        data_train.append(dg[0])
        gold_train.append(dg[1])
    for dg in data_gold[train_size+val_size:]:
        data_test.append(dg[0])
        gold_test.append(dg[1])
    data_gold_ood=[]
    for path in ood_path:
        with open(path) as fd:
            rd=csv.reader(fd, delimiter="\t", quotechar='"')
            for line in rd:
                li=line[-1].split()
                data_gold_ood.append((li, int(line[1])))
    data_ood = []
    gold_ood = []
    for dg in data_gold_ood:
        data_ood.append(dg[0])
        gold_ood.append(dg[1])
    return data_train,gold_train,data_test,gold_test,data_ood,gold_ood


def get_vocab(data,min_count):
    word_to_int,int_to_word={},{}
    word_count=Counter()
    for sentence in data:
        for word in sentence:
            word_count[word]+=1
    res=[]
    for word in word_count:
        if word_count[word]>min_count:
            res.append(word)
    word_to_int['@pad'],word_to_int['@unk']=0,1
    int_to_word[0],int_to_word[1]='@pad','@unk'
    index=2
    for word in res:
        int_to_word[index]=word
        word_to_int[word]=index
        index+=1
    return word_to_int,int_to_word
    

def accuracy_cal(output,answer):
    pred=F.softmax(output,dim=-1)
    _,pred=pred.max(dim=-1)
    count=0
    if pred[0]==answer[0]:
        count+=1         
    return count


class DataLoader(object):
    def __init__(self,data,gold,batch_size,word_to_int,device,shuffle=True):
        self.data=data
        self.data_len = len(data)
        self.batch_size=batch_size
        self.word_to_int=word_to_int
        self.gold=gold
        self.device = device
        self.count = 0
        self.data_list = list(zip(self.data,self.gold))
        self.num_batches = math.ceil(self.data_len / self.batch_size)
        self.shuffle=shuffle
            
    def get_data(self):
        return self.data_list[self.count*self.batch_size : min((self.count+1)* self.batch_size, self.data_len)]
        
    def __load_next__(self):
        data=self.get_data()
        self.count += 1
        if self.count == self.num_batches:
            if self.shuffle:
                random.shuffle(self.data_list)
            self.count = 0
        dd_temp,data_temp,ans_temp,max_len=[],[],[],0
        for sentence in data:
            temp=[]
            sent=sentence[0]
            dd_temp.append(sent)
            answer=sentence[1]
            for word in sent:                               
                if word in self.word_to_int:  
                    temp.append(self.word_to_int[word])
                else:
                    temp.append(self.word_to_int['@unk'])
            data_temp.append(temp)
            ans_temp.append(answer)
            max_len=max(max_len,len(temp))
        if max_len == 5: # 10
            max_len = 6  # 11
            dd_temp2 = []
            for sent in dd_temp:
                sent.append('@pad')
                dd_temp2.append(sent)
            dd_temp = dd_temp2
        input=torch.zeros(len(data_temp),max_len).long().to(self.device)
        input_mask=torch.zeros(len(data_temp),max_len).long().to(self.device)
        pos=torch.zeros(len(data_temp),max_len).long().to(self.device)
        answers=torch.tensor(ans_temp).long().to(self.device)
        for i,sentence in enumerate(data_temp):
            input_mask[i][:len(sentence)]=1
            for j,word in enumerate(sentence):
                input[i][j]=word
                pos[i][j]=j
        return dd_temp,input,input_mask,pos,answers
            
import math
import random
import torch
from collections.abc import Mapping

class BatchContainer(Mapping):
    """Custom container that supports .to(device) operation"""
    def __init__(self, data):
        self._data = data
        self.shape = torch.tensor([len(data)])
        
    def __getitem__(self, key):
        if key == 0:
            return self
        return self._data[key]
        
    def __iter__(self):
        return iter(self._data)
        
    def __len__(self):
        return len(self._data)
        
    def to(self, device):
        """Recursively move all tensors to the specified device"""
        moved_data = {}
        for key, value in self._data.items():
            if isinstance(value, torch.Tensor):
                moved_data[key] = value.to(device)
            elif isinstance(value, (list, tuple)):
                # Handle lists/tuples of tensors
                moved_data[key] = [v.to(device) if isinstance(v, torch.Tensor) else v 
                                 for v in value]
            else:
                moved_data[key] = value
        return BatchContainer(moved_data)
        
    def __repr__(self):
        return f"BatchContainer({self._data})"

class DataLoader_KFLLA:
    def __init__(self, data, gold, batch_size, word_to_int, device, shuffle=True):
        self.data = data
        self.gold = gold
        self.batch_size = batch_size
        self.word_to_int = word_to_int
        self.device = device
        self.shuffle = shuffle
        
        self.data_len = len(data)
        self.data_list = list(zip(self.data, self.gold))
        self.num_batches = self.data_len // self.batch_size
        self.reset()
        self.dataset = data
    
    def reset(self):
        self.count = 0
        if self.shuffle:
            random.shuffle(self.data_list)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        batch_data = self.data_list[
            self.count * self.batch_size : 
            min((self.count + 1) * self.batch_size, self.data_len)
        ]
        self.count += 1
        if self.count == self.num_batches:
            self.reset()
            raise StopIteration
        
        # actual_batch_size = min((self.count + 1) * self.batch_size, self.data_len) - self.count * self.batch_size + 1 
        # Process batch
        dd_temp,data_temp,ans_temp,max_len=[],[],[],0
        for sentence in batch_data:
            temp=[]
            sent=sentence[0]
            dd_temp.append(sent)
            answer=sentence[1]
            for word in sent:                               
                if word in self.word_to_int:  
                    temp.append(self.word_to_int[word])
                else:
                    temp.append(self.word_to_int['@unk'])
            data_temp.append(temp)
            ans_temp.append(answer)
            max_len=max(max_len,len(temp))
        if max_len == 5: # 10
            max_len = 6  # 11
            dd_temp2 = []
            for sent in dd_temp:
                sent.append('@pad')
                dd_temp2.append(sent)
            dd_temp = dd_temp2
        input=torch.zeros(len(data_temp),max_len).long().to(self.device)
        input_mask=torch.zeros(len(data_temp),max_len).long().to(self.device)
        pos=torch.zeros(len(data_temp),max_len).long().to(self.device)
        answers=torch.tensor(ans_temp).long().to(self.device)
        for i,sentence in enumerate(data_temp):
            input_mask[i][:len(sentence)]=1
            for j,word in enumerate(sentence):
                input[i][j]=word
                pos[i][j]=j
        
        # Create container
        batch = BatchContainer({
            'sentences': dd_temp,  # Original sentences
            'input_ids': input.to(self.device),
            'attention_mask': input_mask.to(self.device),
            'position_ids': pos.to(self.device),
            'labels': answers.to(self.device)
        })
        
        return batch, batch['labels']

class TestLoader(DataLoader):
    def __init__(self,data,gold,word_to_int,device):
        self.data=data
        self.batch_size=1
        self.word_to_int=word_to_int
        self.gold=gold
        self.device=device
        self.count=0
        self.len=len(data)
            
    def reset_count(self):
        self.count=0
            
    def get_data(self):
        data=self.data[self.count]
        ans=self.gold[self.count]   
        final=[(data,ans)]
        self.count+=1
        if self.count==len(self.data):
            self.reset_count()
        return final
