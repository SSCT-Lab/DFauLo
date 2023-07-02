import torch
from torch import nn
from torchvision.transforms import transforms

from utils.dataset import dataset
import json
import random

from tqdm import tqdm

from utils.models import *
from utils.dataset import *
import torch
from wavemix.classification import WaveMix
# args

dataset_name = 'MNIST'
model_name = 'LeNet5'
fault_type = 'OriginalTrainData'
class_path = './dataset/mnist_classes.json'
image_size = (28, 28, 1)
lr = 0.001
epoches = 20
batch_size = 64
#

if model_name == 'WaveMix':
    model = WaveMix(
        num_classes=26,
        depth=4,
        mult=2,
        ff_channel=48,
        final_dim=48,
        dropout=0.3,
        level=3,
        patch_size=4,
    )
else:
    model = eval(model_name)()


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


transform = transforms.Compose([
    # MNIST transform
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# train the model
train_data_s = data_slice('./dataset/' + fault_type + '/' + dataset_name+'/train')[0]
train_data = dataset(root='./dataset/' + fault_type + '/' + dataset_name + '/',
                     classes_path=class_path,
                     transform=transform,
                     image_size=image_size,
                     image_set='train',
                     specific_label=None,
                     ignore_list=[],
                     data_s=train_data_s)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_data_s = data_slice('./dataset/OriginalTestData/'+dataset_name+'/test')[0]
test_data = dataset(root='./dataset/OriginalTestData/'+dataset_name+'/',
                    classes_path=class_path,
                    transform=transform,
                    image_size=image_size,
                    image_set='test',
                    specific_label=None,
                    ignore_list=[],
                    data_s=test_data_s)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9, 13], gamma=0.1)
assert os.path.exists('./dataset/' + fault_type + '/' + dataset_name + '/')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
for epoch in range(epoches):
    model.train()
    for i, (images, labels, _) in enumerate(train_loader):
        outputs = model(images.to(device))
        loss = criterion(outputs, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, epoches, i + 1, len(train_data) // batch_size, loss.item()))
    # lr_scheduler.step()
    # test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels, _ in test_loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            labels = labels.to(device)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the test set: {} %'.format(100 * correct / total))

# save the model
torch.save(model.state_dict(), './dataset/' + fault_type + '/' + dataset_name + '/' + model_name + '.pth')
