'''
This code is used to evaluate the classification accuracy of the trained model.
You should at least guarantee this code can run without any error on validation set.
And whether this code can run is the most important factor for grading.
We provide the remaining code, all you should do are, and you can't modify the remaining code:
1. Replace the random classifier with your trained model.(line 64-68)
2. modify the get_label function to get the predicted label.(line 18-24)(just like Leetcode solutions)
'''
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import csv
import os
NUM_CLASSES = len(my_bidict)

# Write your code here
# And get the predicted label, which is a tensor of shape (batch_size,)
# Begin of your code
def get_label(model, model_input, device):
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake, sum_all= False)
    loss_all = torch.empty(NUM_CLASSES, model_input.shape[0])
    for i in range(NUM_CLASSES):
        model_out = model(model_input,  torch.full((model_input.shape[0],), i).to(device))
        result= loss_op(model_input, model_out)
        loss_all[i] = result
    # print(loss_all)
    # print(torch.argmin(loss_all, dim=0))
    return torch.argmin(loss_all, dim=0).to(device)
# End of your code

def classifier(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        original_label = [my_bidict[item] for item in categories]
        original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        answer = get_label(model, model_input, device)
        correct_num = torch.sum(answer == original_label)
        acc_tracker.update(correct_num.item(), model_input.shape[0])
    
    return acc_tracker.get_ratio()
        
def evaluate_and_save_predictions(model, data_dir, device, output_csv):
    model.eval()
    predictions = []
    with torch.no_grad():
        for filename in os.listdir(data_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):  # Assuming images are JPEG or PNG
                img_path = os.path.join(data_dir, filename)
                img = Image.open(img_path)
                img = transforms.Resize((32, 32))(img)
                img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)  # Convert to tensor and move to device
                print(img_tensor.size())
                predicted_label = get_label(model, img_tensor, device).item()
                predictions.append((filename, predicted_label))

    # Save predictions to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'label'])
        writer.writerows(predictions)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='validation', help='Mode for the dataset')
    parser.add_argument('-o', '--output_csv', type=str,
                        default='predictions3.csv', help='Output CSV file for predictions')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)

    #Write your code here
    #You should replace the random classifier with your trained model
    #Begin of your code
    model = PixelCNN(nr_resnet=2, nr_filters=60, nr_logistic_mix=5)
    #End of your code
    
    model = model.to(device)
    #Attention: the path of the model is fixed to 'models/conditional_pixelcnn.pth'
    #You should save your model to this path
    model.load_state_dict(torch.load('models\pcnn_cpen455_from_scratch_99.pth'))
    model.eval()
    print('model parameters loaded')
    # acc = classifier(model = model, data_loader = dataloader, device = device)
    # print(f"Accuracy: {acc}")
    evaluate_and_save_predictions(model=model, data_dir='data/test', device=device, output_csv=args.output_csv)
        