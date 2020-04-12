import torch
import pickle
import torch.utils.data as Data
def load_data(num_classes):
    with open('./temp/ctx600_order.pkl', 'rb') as f:
        data = pickle.load(f)
    train_data = data[0].transpose(0,3,1,2)
    train_label = data[1]
    test_data = data[2].transpose(0,3,1,2)
    test_label = data[3]
    
    #取num_classes个汉字进行分类
    train_count = 200 * num_classes
    test_count = 40 * num_classes
    
    train_data = train_data[:train_count]
    train_label = train_label[:train_count]
    test_data = test_data[:test_count]
    test_label = test_label[:test_count]
    
    return train_data,train_label,test_data,test_label

def load_tensor_data(num_classes):
    train_data,train_label,test_data,test_label = load_data(num_classes)

    print('训练数据大小：',train_data.shape)
    print('测试数据大小：',test_data.shape)

    X_train = torch.from_numpy(train_data).float()
    y_train = torch.from_numpy(train_label).long()
    X_test = torch.from_numpy(test_data).float()
    y_test = torch.from_numpy(test_label).long()

    train_dataset = Data.TensorDataset(X_train, y_train)
    test_dataset = Data.TensorDataset(X_test, y_test)

    return train_dataset, test_dataset