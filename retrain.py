import random

from utils.models import *
from utils.dataset import *
import torch

dataset_name = 'MTFL'
model_name_list = ['TCDCNN']
fault_type_list = ['RandomLabelNoise',  'RandomDataNoise']
image_size = None
epoches = 15
batch_size = 64
remove_ratio = 0.05


def data_slice(path_dir):
    slice_num = int(1)
    random.seed(2023)
    with open(class_path, 'r') as f:
        classes = json.load(f)
    class_keys = list(classes.keys())
    result = {i: {} for i in range(slice_num)}
    for name in class_keys:
        img_list_dir = os.listdir(os.path.join(path_dir, name))
        img_list = []
        for img in img_list_dir:
            path = os.path.join(name, img)
            img_list.append(path)
        random.shuffle(img_list)
        slice_len = len(img_list) // slice_num
        for i in range(slice_num):
            result[i][name] = img_list[i * slice_len:(i + 1) * slice_len]
    print('data slice done with slice num: ', slice_num)
    return result


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class_path = './dataset/' + dataset_name.lower() + '_classes.json'
MethodList = ['Manual','Uncertainty']

for model_name in model_name_list:
    for fault_type in fault_type_list:
        datasetroot = './dataset/' + fault_type + '/' + dataset_name
        classes = load_json(class_path)
        class_keys = list(classes.keys())
        if model_name != 'TCDCNN':
            name2label = {}
            for label in class_keys:
                image_names = os.listdir(datasetroot + '/train/' + label)
                for image_name in image_names:
                    name2label[image_name] = label

        model_args = './dataset/' + dataset_name.lower() + '_model_args.pth'
        modelargs = torch.load(model_args)
        model_path = './dataset/' + fault_type + '/' + dataset_name + '/' + model_name + '.pth'

        reatrain_save_path = datasetroot + '/retrain/' + model_name + '/'
        os.makedirs(reatrain_save_path, exist_ok=True)

        if os.path.exists(reatrain_save_path + 'results_acc.json'):
            results_acc = load_json(reatrain_save_path + 'results_acc.json')
        else:
            results_acc = {Method: -1 for Method in MethodList}
        for Method in MethodList:
            if os.path.exists(reatrain_save_path + Method + '_retrain.pth'):
                print('skip '+reatrain_save_path + Method + '_retrain.pth')
                continue
            print('runing on '+model_name+' with '+fault_type+' and '+Method)
            seed_torch(2023)

            if model_name != 'TCDCNN':
                image_list_path = datasetroot + '/results/' + model_name + '/' + Method + '_results_list.json'
                image_list = load_json(image_list_path)
                image_list = image_list[int(len(image_list) * remove_ratio):]
                print('re image_list length: ', len(image_list))
                data_s = {label: [] for label in class_keys}
                for image_name in image_list:
                    data_s[name2label[image_name]].append(os.path.join(name2label[image_name], image_name))
                ignore_list=[]
            else:
                image_list_path = datasetroot + '/results/' + model_name + '/' + Method + '_results_list.json'
                image_list = load_json(image_list_path)
                ignore_list = image_list[:int(len(image_list) * remove_ratio)]
                print('re ignore_list length: ', len(ignore_list))
                data_s = [None]
            model = eval(model_name)()
            model.load_state_dict(torch.load(model_path))
            loss_fn = nn.CrossEntropyLoss()
            if modelargs['optimizer'] == 'SGD':
                if model_name == 'ResNet' or model_name == 'VGG':
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
                elif model_name == 'TCDCNN':
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            elif modelargs['optimizer'] == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            train_data = dataset(root=datasetroot,
                                 classes_path=class_path,
                                 transform=modelargs['transform'],
                                 image_size=image_size,
                                 image_set='train',
                                 specific_label=None,
                                 ignore_list=ignore_list,
                                 data_s=data_s)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
            if model_name != 'TCDCNN':
                test_data_s = data_slice('./dataset/OriginalTestData/' + dataset_name + '/test')[0]
            else:
                test_data_s = [None]

            test_data = dataset(root='./dataset/OriginalTestData/' + dataset_name,
                                classes_path=class_path,
                                transform=modelargs['transform'],
                                image_size=image_size,
                                image_set='test',
                                specific_label=None,
                                ignore_list=[],
                                data_s=test_data_s)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.train()
            for epoch in range(epoches):
                for i, data in enumerate(train_loader):
                    images, labels, image_paths = data
                    if model_name == 'TCDCNN':
                        out = model(images.to(device).float())
                    else:
                        out = model(images.to(device))
                    labels = labels.to(device)
                    if model_name == 'TCDCNN':
                        loss = model.loss([out],
                                          [labels.float()])
                    else:
                        loss = loss_fn(out, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print('\r', 'epoch: ', epoch, 'processing batch: ', i, end='')
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                mse_list = []
                for data in test_loader:
                    images, labels, _ = data
                    if model_name == 'TCDCNN':
                        outputs = model(images.float().to(device))
                    else:
                        outputs = model(images.to(device))
                    if model_name == 'TCDCNN':
                        accuracy = model.accuracy(outputs, labels.float().to(device))
                        mse_list.append(accuracy)
                    else:
                        _, predicted = torch.max(outputs.data, 1)
                        labels = labels.to(device)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                if model_name == 'TCDCNN':
                    print('Test Accuracy before mutation on test set: {} %'.format(sum(mse_list) / len(mse_list)))
                else:
                    print('Test Accuracy after mutation on test set: {} %'.format(correct / total))
            save_as_json(results_acc, reatrain_save_path + 'results_acc.json')
            # save model
            torch.save(model, reatrain_save_path + Method + '_retrain.pth')
