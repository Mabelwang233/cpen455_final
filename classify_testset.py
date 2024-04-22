'''
script to classify the test set and save result to result.csv, logits to logits.pt
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
logits = torch.empty(519, 4)

def get_test_label(model, model_input, device, img_idx = None):
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake, sum_all= False)
    ans = []
    #feed the every images with 4 labels repectively
    for image_in in model_input:
        loss_all = []
        image_in = image_in.unsqueeze(0).to(device)
        for i in range(NUM_CLASSES):
            image_out = model(image_in,  torch.tensor([i]).to(device))
            loss_all.append(loss_op(image_in, image_out))
        logits[img_idx, :] = torch.tensor(loss_all)
        #the label makes the smallest loss is the most likely label
        ans.append(np.argmin([loss.detach().cpu().numpy() for loss in loss_all]))
    return torch.tensor(ans).to(device) 

#helper function to classify the test set, and save result to result.csv   
def evaluate_and_save_predictions(model, data_dir, device, output_csv):
    model.eval()
    predictions = []
    with torch.no_grad():
        i = 0
        for filename in os.listdir(data_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):  # Assuming images are JPEG or PNG
                img_path = os.path.join(data_dir, filename)
                img = Image.open(img_path)
                img = transforms.Resize((32, 32))(img)
                img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)  # Convert to tensor and move to device
                predicted_label = get_test_label(model, img_tensor, device, i).item()
                predictions.append((filename, predicted_label))
                i = i + 1

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
                        default='test', help='Mode for the dataset')
    
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
    #load model
    model = PixelCNN(nr_resnet=1, nr_filters=80, nr_logistic_mix=5)
    model = model.to(device)

    model.load_state_dict(torch.load('models\conditional_pixelcnn.pth'))
    model.eval()
    print('model parameters loaded')
    #Test set does not have known label, so can not get accuracy

    #Added to save .pt and .csv file for submission
    evaluate_and_save_predictions(model=model, data_dir='data/test', device=device, output_csv='result.csv')
    print(logits)
    torch.save(logits, 'logits.pt')
        